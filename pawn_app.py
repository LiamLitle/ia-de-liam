"""P.A.W.N. : mini application pour jouer contre lui dans le navigateur.

Mettre ce fichier dans le même dossier que pawn.onnx, puis :
    pip install python-chess onnxruntime numpy
    python pawn_app.py
"""
import json
import os
import random
import sys
import threading
import webbrowser
from datetime import date, datetime
from urllib.parse import parse_qs, urlparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import chess
import chess.pgn
import chess.polyglot
import chess.svg
import numpy as np
import onnxruntime as ort

DOSSIER = os.path.dirname(os.path.abspath(__file__))
PORT = 8765
MATE = 100_000

# les niveaux "arbre" : profondeur (coups regardés à l'avance), bruit (hésitation),
# part de coups au hasard. Mêmes réglages pour les trois modèles à arbre complet.
_NIVEAUX_ARBRE = {
    1: dict(label="1 · Très facile", depth=1, noise=300, p_random=0.25),
    2: dict(label="2 · Facile", depth=1, noise=100, p_random=0.05),
    3: dict(label="3 · Moyen", depth=2, noise=150, p_random=0.10),
    4: dict(label="4 · Difficile", depth=2, noise=8, p_random=0.0),
    5: dict(label="5 · Très difficile (lent)", depth=3, noise=8, p_random=0.0),
}

# quatre versions de P.A.W.N. : trois anciennes (arbre complet, toutes les feuilles
# évaluées d'un coup) et Soluce avec une recherche alpha-bêta + quiescence, où la
# profondeur devient le "type de réflexion" (elle élague les branches inutiles et
# prolonge les captures, donc à profondeur égale elle est plus forte ET plus rapide).
MODELES = {
    "ancien": dict(fichier="pawn.onnx", libelle="Ancien P.A.W.N.", recherche="arbre", niveaux=_NIVEAUX_ARBRE),
    "big": dict(fichier="pawn_big.onnx", libelle="P.A.W.N. big", recherche="arbre", niveaux=_NIVEAUX_ARBRE),
    "soluce_arbre": dict(fichier="pawn_soluce.onnx", libelle="Soluce (arbre complet)", recherche="arbre", niveaux=_NIVEAUX_ARBRE),
    "soluce_ab": dict(
        fichier="pawn_soluce.onnx", libelle="Soluce + alpha-bêta/quiescence", recherche="alphabeta",
        niveaux={
            1: dict(label="1 · Très facile", depth=1, prof_q=2, noise=250, p_random=0.20),
            2: dict(label="2 · Facile", depth=1, prof_q=3, noise=80, p_random=0.05),
            3: dict(label="3 · Moyen", depth=2, prof_q=4, noise=40, p_random=0.0),
            4: dict(label="4 · Difficile", depth=2, prof_q=4, noise=0, p_random=0.0),
            5: dict(label="5 · Très difficile", depth=3, prof_q=4, noise=0, p_random=0.0),
            6: dict(label="6 · Expert (très lent, ~40s/coup)", depth=4, prof_q=4, noise=0, p_random=0.0),
        },
    ),
}
MODELE_DEFAUT, NIVEAU_DEFAUT = "soluce_ab", 3

SESSIONS = {}


def session_pour(modele):
    s = SESSIONS.get(modele)
    if s is None:
        fichier = MODELES[modele]["fichier"]
        chemin = os.path.join(DOSSIER, fichier)
        if not os.path.exists(chemin):
            sys.exit(f"{fichier} introuvable : {chemin}\nMets ce fichier à côté de pawn_app.py.")
        s = ort.InferenceSession(chemin, providers=["CPUExecutionProvider"])
        SESSIONS[modele] = s
    return s


session_pour(MODELE_DEFAUT)  # on vérifie tout de suite que le modèle par défaut est là

PIECES_ID = {c: i + 1 for i, c in enumerate("PNBRQK")}


def enc_fen(fen):
    f = fen.split(" ")
    noir = f[1] == "b"
    out = bytearray(69)
    r = c = 0
    for ch in f[0]:
        if ch == "/":
            r += 1
            c = 0
        elif ch.isdigit():
            c += int(ch)
        else:
            nous = ch.isupper() != noir
            out[(r if noir else 7 - r) * 8 + c] = PIECES_ID[ch.upper()] + (0 if nous else 6)
            c += 1
    ours, theirs = ("kq", "KQ") if noir else ("KQ", "kq")
    out[64] = ours[0] in f[2]
    out[65] = ours[1] in f[2]
    out[66] = theirs[0] in f[2]
    out[67] = theirs[1] in f[2]
    if f[3] != "-":
        out[68] = ord(f[3][0]) - 96
    return bytes(out)


def evals(fens, modele):
    if not fens:
        return np.zeros(0, dtype=np.float32)
    s = session_pour(modele)
    res = []
    for i in range(0, len(fens), 4096):
        x = np.frombuffer(b"".join(map(enc_fen, fens[i:i + 4096])), np.uint8).reshape(-1, 69).copy()
        res.append(s.run(None, {"board": x})[0])
    return np.concatenate(res)


def eval1(fen, modele):
    return float(evals([fen], modele)[0])


# -- recherche "arbre complet" (ancien P.A.W.N., P.A.W.N. big, Soluce sans alpha-bêta) --

def arbre(board, depth, ply, feuilles):
    if not any(board.legal_moves):
        return ("t", -(MATE - ply) if board.is_check() else 0)
    if ply > 0 and (board.is_insufficient_material() or board.is_repetition(2) or board.halfmove_clock >= 100):
        return ("t", 0)
    if depth == 0:
        feuilles.append(board.fen())
        return ("f", len(feuilles) - 1)
    fils = []
    for mv in list(board.legal_moves):
        board.push(mv)
        fils.append((mv, arbre(board, depth - 1, ply + 1, feuilles)))
        board.pop()
    return ("n", fils)


def valeur(noeud, v):
    genre, x = noeud
    if genre == "t":
        return x
    if genre == "f":
        return v[x]
    return max(-valeur(c, v) for _, c in x)


def choisir_arbre(board, modele, cfg):
    feuilles = []
    racine = arbre(board, cfg["depth"], 0, feuilles)
    v = evals(feuilles, modele) if feuilles else []
    notes = [(-valeur(c, v) + random.uniform(-cfg["noise"], cfg["noise"]), mv) for mv, c in racine[1]]
    return max(notes, key=lambda t: t[0])[1]


# -- recherche alpha-bêta + quiescence (Soluce) --------------------------------
# évalue une position à la fois (pas en lot), mais élague les branches inutiles et
# prolonge les captures jusqu'à ce que la position soit calme, pour éviter l'effet
# d'horizon (une pièce perdue juste après la profondeur regardée).

VAL_PIECE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def trier_coups(board, coups, priorite):
    def cle(mv):
        if mv == priorite:
            return 10_000
        if board.is_capture(mv):
            prise = board.piece_type_at(mv.to_square) or chess.PAWN  # None = prise en passant
            attaquant = board.piece_type_at(mv.from_square)
            return 1000 + VAL_PIECE[prise] * 10 - VAL_PIECE[attaquant]  # MVV-LVA
        if board.gives_check(mv):
            return 500
        return 0
    return sorted(coups, key=cle, reverse=True)


def quiescence(board, modele, alpha, beta, prof_q):
    stand_pat = eval1(board.fen(), modele)
    if stand_pat >= beta:
        return beta
    alpha = max(alpha, stand_pat)
    if prof_q <= 0:
        return alpha
    captures = [m for m in board.legal_moves if board.is_capture(m)]
    for mv in trier_coups(board, captures, None):
        board.push(mv)
        score = -quiescence(board, modele, -beta, -alpha, prof_q - 1)
        board.pop()
        if score >= beta:
            return beta
        alpha = max(alpha, score)
    return alpha


def negamax(board, modele, profondeur, alpha, beta, ply, prof_q, tt, coup_tt):
    alpha0 = alpha
    if not any(board.legal_moves):
        return -(MATE - ply) if board.is_check() else 0
    if ply > 0 and (board.is_insufficient_material() or board.is_repetition(2) or board.halfmove_clock >= 100):
        return 0
    hash_ = chess.polyglot.zobrist_hash(board)
    entree = tt.get(hash_)
    if entree and entree[0] >= profondeur:
        d, v, drapeau, _ = entree
        if drapeau == "exact":
            return v
        if drapeau == "min":
            alpha = max(alpha, v)
        elif drapeau == "max":
            beta = min(beta, v)
        if alpha >= beta:
            return v
    if profondeur <= 0:
        return quiescence(board, modele, alpha, beta, prof_q)
    meilleur, meilleur_coup = -MATE - 1, None
    for mv in trier_coups(board, list(board.legal_moves), coup_tt.get(hash_)):
        board.push(mv)
        score = -negamax(board, modele, profondeur - 1, -beta, -alpha, ply + 1, prof_q, tt, coup_tt)
        board.pop()
        if score > meilleur:
            meilleur, meilleur_coup = score, mv
        alpha = max(alpha, score)
        if alpha >= beta:
            break  # élagage : les coups restants ne changeraient pas la décision de l'adversaire
    drapeau = "exact" if alpha0 < meilleur < beta else ("min" if meilleur >= beta else "max")
    tt[hash_] = (profondeur, meilleur, drapeau, meilleur_coup)
    if meilleur_coup is not None:
        coup_tt[hash_] = meilleur_coup
    return meilleur


def choisir_alphabeta(board, modele, cfg):
    tt, coup_tt = {}, {}
    meilleur_coup = None
    for d in range(1, cfg["depth"] + 1):  # approfondissement itératif : trie mieux à chaque tour
        alpha, beta = -MATE - 1, MATE + 1
        hash_ = chess.polyglot.zobrist_hash(board)
        meilleur_score, meilleur_du_tour = -MATE - 1, None
        for mv in trier_coups(board, list(board.legal_moves), coup_tt.get(hash_)):
            board.push(mv)
            score = -negamax(board, modele, d - 1, -beta, -alpha, 1, cfg["prof_q"], tt, coup_tt)
            board.pop()
            bruit = random.uniform(-cfg["noise"], cfg["noise"]) if cfg["noise"] else 0
            if score + bruit > meilleur_score:
                meilleur_score, meilleur_du_tour = score + bruit, mv
            alpha = max(alpha, score)
        meilleur_coup = meilleur_du_tour
        coup_tt[hash_] = meilleur_coup
    return meilleur_coup


def choisir(board, modele, niveau):
    cfg = MODELES[modele]["niveaux"][niveau]
    if random.random() < cfg["p_random"]:
        return random.choice(list(board.legal_moves))
    if MODELES[modele]["recherche"] == "alphabeta":
        return choisir_alphabeta(board, modele, cfg)
    return choisir_arbre(board, modele, cfg)


FIN = {
    chess.Termination.CHECKMATE: "échec et mat",
    chess.Termination.STALEMATE: "pat",
    chess.Termination.INSUFFICIENT_MATERIAL: "matériel insuffisant",
    chess.Termination.THREEFOLD_REPETITION: "répétition de position",
    chess.Termination.FIVEFOLD_REPETITION: "répétition de position",
    chess.Termination.FIFTY_MOVES: "règle des 50 coups",
    chess.Termination.SEVENTYFIVE_MOVES: "règle des 75 coups",
}


class Partie:
    def __init__(self):
        self.verrou = threading.RLock()
        self.nouvelle(chess.WHITE, MODELE_DEFAUT, NIVEAU_DEFAUT)

    def nouvelle(self, humain, modele, niveau):
        with self.verrou:
            self.gid = getattr(self, "gid", 0) + 1
            self.board = chess.Board()
            self.humain = humain
            self.modele = modele if modele in MODELES else MODELE_DEFAUT
            self.niveau = min(max(MODELES[self.modele]["niveaux"]), max(1, niveau))
            self.san = []
            self.reflechit = False
        self.penser()

    def penser(self):
        with self.verrou:
            if self.reflechit or self.board.is_game_over(claim_draw=True) or self.board.turn == self.humain:
                return
            self.reflechit = True
            gid, copie, modele, niveau = self.gid, self.board.copy(), self.modele, self.niveau
        threading.Thread(target=self._calcul, args=(gid, copie, modele, niveau), daemon=True).start()

    def _calcul(self, gid, copie, modele, niveau):
        mv = choisir(copie, modele, niveau)
        with self.verrou:
            if gid != self.gid:
                return
            self.san.append(self.board.san(mv))
            self.board.push(mv)
            self.reflechit = False

    def jouer(self, depart, arrivee, promo):
        with self.verrou:
            if self.reflechit or self.board.turn != self.humain or self.board.is_game_over(claim_draw=True):
                return False
            try:
                mv = chess.Move.from_uci(depart + arrivee + (promo or ""))
            except ValueError:
                return False
            if mv not in self.board.legal_moves:
                return False
            self.san.append(self.board.san(mv))
            self.board.push(mv)
        self.penser()
        return True

    def annuler(self):
        with self.verrou:
            self.gid += 1
            self.reflechit = False
            while self.board.move_stack:
                self.board.pop()
                self.san.pop()
                if self.board.turn == self.humain:
                    break
        self.penser()

    def regler_niveau(self, niveau):
        with self.verrou:
            maxi = max(MODELES[self.modele]["niveaux"])
            self.niveau = min(maxi, max(1, niveau))

    def regler_modele(self, modele):
        if modele not in MODELES:
            return
        with self.verrou:
            self.modele = modele
            maxi = max(MODELES[modele]["niveaux"])
            self.niveau = min(self.niveau, maxi)

    def position(self, ply):
        with self.verrou:
            pile = self.board.move_stack[:ply]
            b = chess.Board()
            for mv in pile:
                b.push(mv)
            return {
                "pieces": {chess.square_name(s): p.symbol() for s, p in b.piece_map().items()},
                "dernier": [chess.square_name(pile[-1].from_square), chess.square_name(pile[-1].to_square)] if pile else None,
                "echec": chess.square_name(b.king(b.turn)) if b.is_check() else None,
            }

    def pgn(self):
        with self.verrou:
            jeu = chess.pgn.Game.from_board(self.board)
            moi, lui = "Toi", f"{MODELES[self.modele]['libelle']} (niveau {self.niveau})"
            jeu.headers["Event"] = "Partie contre P.A.W.N."
            jeu.headers["Date"] = date.today().strftime("%Y.%m.%d")
            jeu.headers["White"] = moi if self.humain == chess.WHITE else lui
            jeu.headers["Black"] = lui if self.humain == chess.WHITE else moi
            jeu.headers["Result"] = self.board.result(claim_draw=True)
            return str(jeu)

    def export(self):
        with self.verrou:
            racine = self.board.root()
            b = racine.copy()
            valeur = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
            fens, coups = [b.fen()], []
            for k, mv in enumerate(self.board.move_stack):
                trait = b.turn
                san, prise = b.san(mv), None
                if b.is_capture(mv):
                    pc = chess.Piece(chess.PAWN, not trait) if b.is_en_passant(mv) else b.piece_at(mv.to_square)
                    prise = pc.symbol().lower()
                b.push(mv)
                fens.append(b.fen())
                mat = sum(valeur.get(p.piece_type, 0) * (1 if p.color else -1) for p in b.piece_map().values())
                coups.append({
                    "ply": k + 1, "numero": k // 2 + 1,
                    "couleur": "blancs" if trait else "noirs",
                    "joueur": "toi" if trait == self.humain else "pawn",
                    "san": san, "uci": mv.uci(),
                    "prise": prise, "echec": b.is_check(), "mat": b.is_checkmate(),
                    "fen_apres": fens[-1], "materiel_blancs_moins_noirs": mat,
                })
            # évaluation du modèle après chaque coup, du point de vue des blancs (en centipions)
            vivantes = [i for i in range(len(fens)) if any(chess.Board(fens[i]).legal_moves)]
            if vivantes:
                ev = evals([fens[i] for i in vivantes], self.modele)
                for i, v in zip(vivantes, ev):
                    signe = 1 if chess.Board(fens[i]).turn else -1
                    if i > 0:
                        coups[i - 1]["eval_blancs_cp"] = int(round(float(v) * signe))
            fini = b.is_game_over(claim_draw=True)
            res = b.outcome(claim_draw=True) if fini else None
            return {
                "format": "pawn-partie-v1",
                "date": datetime.now().isoformat(timespec="seconds"),
                "modele": MODELES[self.modele]["fichier"],
                "modele_libelle": MODELES[self.modele]["libelle"],
                "niveau_pawn": self.niveau,
                "reglages_niveau": MODELES[self.modele]["niveaux"][self.niveau],
                "tu_jouais": "blancs" if self.humain == chess.WHITE else "noirs",
                "position_depart": racine.fen(),
                "resultat": b.result(claim_draw=True) if fini else "*",
                "fin": FIN.get(res.termination, "fin de partie") if res else "partie en cours",
                "nb_coups_joues": len(coups),
                "coups": coups,
                "pgn": self.pgn(),
            }

    def prises(self):
        b = self.board.root()
        out = []
        for k, mv in enumerate(self.board.move_stack):
            if b.is_capture(mv):
                if b.is_en_passant(mv):
                    pc = chess.Piece(chess.PAWN, not b.turn)
                    case = chess.square_name(mv.to_square - 8 if b.turn else mv.to_square + 8)
                else:
                    pc = b.piece_at(mv.to_square)
                    case = chess.square_name(mv.to_square)
                out.append({"ply": k + 1, "par": "w" if b.turn else "b", "piece": pc.symbol(), "case": case})
            b.push(mv)
        return out

    def etat(self):
        with self.verrou:
            b = self.board
            fini = b.is_game_over(claim_draw=True)
            legaux = {}
            if not fini and not self.reflechit and b.turn == self.humain:
                for mv in b.legal_moves:
                    legaux.setdefault(chess.square_name(mv.from_square), set()).add(chess.square_name(mv.to_square))
            if fini:
                res = b.outcome(claim_draw=True)
                cause = FIN.get(res.termination, "fin de partie")
                if res.winner is None:
                    texte = f"Partie nulle ({cause})."
                elif res.winner == self.humain:
                    texte = f"Tu as gagné ! ({cause})"
                else:
                    texte = f"P.A.W.N. a gagné ({cause})."
            elif self.reflechit:
                texte = "P.A.W.N. réfléchit…"
            else:
                texte = "Échec ! À toi de jouer." if b.is_check() else "À toi de jouer."
            return {
                "pieces": {chess.square_name(s): p.symbol() for s, p in b.piece_map().items()},
                "legaux": {k: sorted(v) for k, v in legaux.items()},
                "dernier": [chess.square_name(b.peek().from_square), chess.square_name(b.peek().to_square)] if b.move_stack else None,
                "echec": chess.square_name(b.king(b.turn)) if b.is_check() else None,
                "humain": "b" if self.humain == chess.BLACK else "w",
                "modele": self.modele,
                "niveau": self.niveau,
                "coups": self.san,
                "prises": self.prises(),
                "texte": texte,
                "fini": fini,
                "reflechit": self.reflechit,
            }


partie = Partie()

PAGE = r"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>P.A.W.N.</title>
<style>
  :root { --fond:#f6f5f1; --carte:#fff; --texte:#1c1b19; --doux:#6b6a65; --trait:#dcdad2; --clair:#ebecd0; --fonce:#779556; --accent:#2a78d6; --choix:rgba(42,120,214,.45); --gain:#2f9e5b; --perte:#d94a48; --dernier:rgba(235,200,60,.55); --echec:rgba(227,73,72,.75); }
  @media (prefers-color-scheme: dark) { :root { --fond:#141413; --carte:#1e1e1c; --texte:#f1f0ec; --doux:#a3a29b; --trait:#33322f; --accent:#4a90e2; } }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--fond); color:var(--texte); font:16px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif; }
  main { max-width:1000px; margin:0 auto; padding:20px 16px 40px; }
  h1 { margin:0 0 4px; font-size:26px; letter-spacing:.5px; }
  .sous { color:var(--doux); margin:0 0 18px; }
  .jeu { display:flex; gap:24px; flex-wrap:wrap; align-items:flex-start; }
  .col { width:min(92vw,560px); }
  .rel { position:relative; }
  .bande { display:flex; align-items:center; gap:8px; min-height:34px; font-size:13px; color:var(--doux); }
  .bande .ico { display:flex; flex-wrap:wrap; align-items:center; }
  .bande .ico svg { width:24px; height:24px; margin-right:-6px; }
  .bande .aucune { font-style:italic; opacity:.7; }
  .bande .solde { font-weight:700; color:var(--texte); background:var(--trait); border-radius:10px; padding:0 7px; }
  #toast { position:absolute; left:50%; top:10px; transform:translate(-50%,-14px); opacity:0; pointer-events:none; padding:8px 14px; border-radius:10px; color:#fff; font-weight:600; font-size:15px; text-align:center; box-shadow:0 3px 14px rgba(0,0,0,.35); transition:opacity .25s, transform .25s; z-index:3; max-width:92%; }
  #toast.on { opacity:1; transform:translate(-50%,0); }
  #toast.gain { background:var(--gain); } #toast.perte { background:var(--perte); }
  .prise { font-size:14px; border-radius:8px; padding:7px 10px; margin:-4px 0 12px; border-left:4px solid var(--trait); background:var(--fond); }
  .prise.gain { border-color:var(--gain); } .prise.perte { border-color:var(--perte); }
  .plateau { display:grid; grid-template-columns:repeat(8,1fr); grid-template-rows:repeat(8,1fr); width:100%; aspect-ratio:1; border-radius:6px; overflow:hidden; box-shadow:0 2px 12px rgba(0,0,0,.18); user-select:none; }
  .case { position:relative; min-width:0; min-height:0; display:flex; align-items:center; justify-content:center; cursor:default; }
  .case.c { background:var(--clair); } .case.f { background:var(--fonce); }
  .case.dernier::before, .case.choisie::before, .case.echec::before { content:""; position:absolute; inset:0; background:var(--dernier); }
  .case.choisie::before { background:var(--choix); } .case.echec::before { background:var(--echec); }
  .case.cible::after { content:""; position:absolute; width:28%; height:28%; border-radius:50%; background:rgba(0,0,0,.25); }
  .case.cible.prise::after { width:88%; height:88%; background:none; border:5px solid rgba(0,0,0,.25); }
  .case.jouable { cursor:pointer; }
  .case svg { position:relative; width:92%; height:92%; pointer-events:none; }
  .coord { position:absolute; font-size:11px; font-weight:600; opacity:.7; pointer-events:none; }
  .coord.n { top:2px; left:3px; } .coord.l { bottom:1px; right:4px; }
  .case.c .coord { color:var(--fonce); } .case.f .coord { color:var(--clair); }
  .panneau { flex:1; min-width:260px; background:var(--carte); border:1px solid var(--trait); border-radius:10px; padding:16px; }
  .statut { font-size:19px; font-weight:600; min-height:28px; margin-bottom:14px; }
  label { display:block; font-size:13px; color:var(--doux); margin:12px 0 4px; }
  select, button { font:inherit; padding:8px 10px; border-radius:8px; border:1px solid var(--trait); background:var(--fond); color:var(--texte); }
  select { width:100%; }
  .rang { display:flex; gap:8px; margin-top:14px; flex-wrap:wrap; }
  button { cursor:pointer; } button.p { background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600; }
  button:disabled { opacity:.5; cursor:default; }
  .legende { font-size:13px; color:var(--doux); margin:-8px 0 6px; }
  .histo { margin-top:18px; border-top:1px solid var(--trait); padding-top:10px; }
  .histo-tete { display:flex; justify-content:space-between; align-items:center; font-weight:600; margin-bottom:8px; }
  .histo-tete { flex-wrap:wrap; gap:6px; }
  .histo-tete .btns { display:flex; gap:6px; }
  .histo-tete button { white-space:nowrap; }
  .histo-tete button { font-size:13px; padding:4px 10px; }
  .vue { display:none; font-size:13px; background:var(--choix); border-radius:8px; padding:8px 10px; margin-bottom:8px; }
  .vue button { font-size:13px; padding:3px 8px; margin-left:6px; }
  .coups { max-height:230px; overflow:auto; font-variant-numeric:tabular-nums; font-size:14px; }
  .coups table { width:100%; border-collapse:collapse; }
  .coups th { position:sticky; top:0; background:var(--carte); text-align:left; font-size:12px; font-weight:600; color:var(--doux); padding:4px 6px; }
  .coups td { padding:3px 6px; }
  .coups td.n { color:var(--doux); width:2.2em; }
  .coups td[data-ply] { cursor:pointer; border-radius:5px; }
  .coups td[data-ply]:hover { background:var(--trait); }
  .coups td.moi { font-weight:600; }
  .coups td.gain { box-shadow:inset 3px 0 var(--gain); }
  .coups td.perte { box-shadow:inset 3px 0 var(--perte); }
  .cle { font-size:12px; color:var(--doux); margin-top:6px; }
  .coups td.sel { background:var(--accent); color:#fff; }
  .vide { color:var(--doux); font-size:14px; }
  #promo { position:fixed; inset:0; display:none; align-items:center; justify-content:center; background:rgba(0,0,0,.45); z-index:5; }
  #promo div { background:var(--carte); padding:16px; border-radius:10px; display:flex; gap:10px; }
  #promo button { width:70px; height:70px; padding:4px; }
  #promo svg { width:100%; height:100%; }
</style>
</head>
<body>
<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>%%PIECES%%</defs></svg>
<main>
  <h1>P.A.W.N.</h1>
  <p class="sous">Poorly Automated Weak Noob : joue contre lui.</p>
  <div class="jeu">
    <div class="col">
      <div class="bande" id="haut"></div>
      <div class="rel"><div class="plateau" id="plateau"></div><div id="toast"></div></div>
      <div class="bande" id="bas"></div>
    </div>
    <div class="panneau">
      <div class="statut" id="statut">…</div>
      <div class="legende" id="legende"></div>
      <div class="prise" id="prise" style="display:none"></div>
      <label for="modele">Modèle de P.A.W.N.</label>
      <select id="modele">%%MODELES_OPTIONS%%</select>
      <label for="niveau">Type de réflexion</label>
      <select id="niveau"></select>
      <label for="couleur">Tu joues</label>
      <select id="couleur">
        <option value="w">les blancs</option>
        <option value="b">les noirs</option>
        <option value="r">au hasard</option>
      </select>
      <div class="rang">
        <button class="p" id="nouvelle">Nouvelle partie</button>
        <button id="annuler">Annuler mon coup</button>
      </div>
      <div class="histo">
        <div class="histo-tete"><span>Historique de la partie</span><span class="btns"><button id="copier">Copier la partie</button> <button id="exporter">Exporter (JSON)</button></span></div>
        <div class="vue" id="vue"></div>
        <div class="coups" id="coups"></div>
        <div class="cle">Trait vert : tu as pris une pièce · trait rouge : tu en as perdu une.</div>
      </div>
    </div>
  </div>
</main>
<div id="promo"><div id="promoChoix"></div></div>
<script>
const NOMS = {p:"pawn", n:"knight", b:"bishop", r:"rook", q:"queen", k:"king"};
const MODELES = %%MODELES_JSON%%;
let etat = null, choisie = null, enAttente = null, vue = null, posVue = null, modeleAffiche = null;

function remplirNiveaux(modele, niveauActuel) {
  const niv = document.getElementById("niveau");
  niv.innerHTML = MODELES[modele].niveaux.map(n => `<option value="${n.valeur}">${n.label}</option>`).join("");
  niv.value = niveauActuel;
}

const VAL = {p:1, n:3, b:3, r:5, q:9, k:0};
const NOM_FR = {p:["pion","m"], n:["cavalier","m"], b:["fou","m"], r:["tour","f"], q:["dame","f"], k:["roi","m"]};
let vusPrises = null, minuteur = null;
function phrase(p, moi) {
  const [nom, g] = NOM_FR[p.piece.toLowerCase()];
  return p.par === moi
    ? {gain:true, texte:`Tu as pris ${g === "f" ? "sa" : "son"} ${nom} en ${p.case} !`}
    : {gain:false, texte:`Tu as perdu ${g === "f" ? "ta" : "ton"} ${nom} en ${p.case} (pris par P.A.W.N.)`};
}
function annoncer(p, moi) {
  const f = phrase(p, moi), t = document.getElementById("toast");
  t.textContent = f.texte; t.className = "on " + (f.gain ? "gain" : "perte");
  clearTimeout(minuteur);
  minuteur = setTimeout(() => t.classList.remove("on"), 3200);
}
function bande(id, titre, liste, solde) {
  const tri = [...liste].sort((a, b) => VAL[b.toLowerCase()] - VAL[a.toLowerCase()]);
  document.getElementById(id).innerHTML = `<span>${titre}</span>` +
    (tri.length ? `<span class="ico">${tri.map(piece).join("")}</span>` : `<span class="aucune">rien pour l'instant</span>`) +
    (solde > 0 ? `<span class="solde">+${solde}</span>` : "");
}

const piece = s => `<svg viewBox="0 0 45 45"><use href="#${s === s.toUpperCase() ? "white" : "black"}-${NOMS[s.toLowerCase()]}"/></svg>`;

async function appel(url, corps) {
  const r = await fetch(url, corps ? {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(corps)} : {});
  return r.json();
}

function dessiner() {
  const e = etat, noir = e.humain === "b";
  if (vusPrises !== null && e.prises.length > vusPrises) annoncer(e.prises[e.prises.length - 1], e.humain);
  vusPrises = e.prises.length;
  const aff = vue === null ? e : {pieces: posVue.pieces, dernier: posVue.dernier, echec: posVue.echec, legaux: {}};
  const plateau = document.getElementById("plateau");
  const cibles = vue === null && choisie && e.legaux[choisie] ? e.legaux[choisie] : [];
  let html = "";
  for (let i = 0; i < 64; i++) {
    const rang = noir ? i >> 3 : 7 - (i >> 3);
    const col = noir ? 7 - (i & 7) : i & 7;
    const nom = "abcdefgh"[col] + (rang + 1);
    const cls = ["case", (rang + col) % 2 ? "c" : "f"];
    if (aff.dernier && aff.dernier.includes(nom)) cls.push("dernier");
    if (nom === choisie) cls.push("choisie");
    if (nom === aff.echec) cls.push("echec");
    if (cibles.includes(nom)) { cls.push("cible"); if (aff.pieces[nom]) cls.push("prise"); }
    if (aff.legaux[nom] || cibles.includes(nom)) cls.push("jouable");
    let coord = "";
    if ((i & 7) === 0) coord += `<span class="coord n">${rang + 1}</span>`;
    if (i >> 3 === 7) coord += `<span class="coord l">${"abcdefgh"[col]}</span>`;
    html += `<div class="${cls.join(" ")}" data-c="${nom}">${coord}${aff.pieces[nom] ? piece(aff.pieces[nom]) : ""}</div>`;
  }
  plateau.innerHTML = html;
  document.getElementById("statut").textContent = e.texte;
  if (modeleAffiche !== e.modele) {
    document.getElementById("modele").value = e.modele;
    remplirNiveaux(e.modele, e.niveau);
    modeleAffiche = e.modele;
  }
  document.getElementById("niveau").value = e.niveau;
  if (!coteInit) { document.getElementById("couleur").value = e.humain; coteInit = true; }
  const moi = e.humain, n = e.coups.length, courant = vue === null ? n : vue;
  const cote = c => (c === "w" ? "Blancs" : "Noirs") + (c === moi ? " · toi" : " · P.A.W.N.");
  document.getElementById("legende").textContent = `${MODELES[e.modele].libelle} · niveau ${e.niveau} · toi : ${moi === "w" ? "les blancs" : "les noirs"}`;
  const jusque = e.prises.filter(p => p.ply <= courant);
  const miennes = jusque.filter(p => p.par === moi).map(p => p.piece);
  const perdues = jusque.filter(p => p.par !== moi).map(p => p.piece);
  const somme = l => l.reduce((t, x) => t + VAL[x.toLowerCase()], 0);
  const solde = somme(miennes) - somme(perdues);
  bande("haut", "P.A.W.N. a pris :", perdues, -solde);
  bande("bas", "Tu as pris :", miennes, solde);
  const parPly = {};
  e.prises.forEach(p => parPly[p.ply] = p);
  const dern = jusque[jusque.length - 1], zp = document.getElementById("prise");
  if (dern) {
    const f = phrase(dern, moi);
    zp.style.display = "block"; zp.className = "prise " + (f.gain ? "gain" : "perte");
    zp.textContent = `${dern.ply === courant ? "Dernier coup" : "Dernière prise (coup " + Math.ceil(dern.ply / 2) + ")"} : ${f.texte}`;
  } else zp.style.display = "none";
  const zone = document.getElementById("coups");
  if (!n) {
    zone.innerHTML = '<div class="vide">Aucun coup pour l\'instant.</div>';
  } else {
    let t = `<table><thead><tr><th></th><th>${cote("w")}</th><th>${cote("b")}</th></tr></thead><tbody>`;
    for (let i = 0; i < n; i += 2) {
      const cel = (k, c) => k < n ? `<td data-ply="${k + 1}" class="${c === moi ? "moi" : ""} ${k + 1 === courant ? "sel" : ""} ${parPly[k + 1] ? (parPly[k + 1].par === moi ? "gain" : "perte") : ""}">${e.coups[k]}</td>` : "<td></td>";
      t += `<tr><td class="n">${i / 2 + 1}.</td>${cel(i, "w")}${cel(i + 1, "b")}</tr>`;
    }
    zone.innerHTML = t + "</tbody></table>";
    const sel = zone.querySelector("td.sel");
    if (vue === null) zone.scrollTop = zone.scrollHeight;
    else if (sel) sel.scrollIntoView({block: "nearest"});
  }
  const banniere = document.getElementById("vue");
  if (vue === null) banniere.style.display = "none";
  else {
    const nom = vue === 0 ? "le début" : `${Math.ceil(vue / 2)}${vue % 2 ? "." : "..."} ${e.coups[vue - 1]}`;
    banniere.style.display = "block";
    banniere.innerHTML = `Tu regardes la position après ${nom}. <button id="retour">Revenir à la partie</button>`;
  }
}

let coteInit = false;
async function voir(ply) {
  const n = etat.coups.length;
  if (ply >= n) { vue = null; posVue = null; }
  else { vue = Math.max(0, ply); posVue = await appel("/api/position?ply=" + vue); }
  choisie = null;
  dessiner();
}

async function rafraichir() {
  const nouveau = await appel("/api/etat");
  const change = !etat || JSON.stringify(nouveau) !== JSON.stringify(etat);
  etat = nouveau;
  if (vue !== null && vue > etat.coups.length) { vue = null; posVue = null; }
  if (change) dessiner();
}

async function jouer(depart, arrivee, promo) {
  const r = await appel("/api/jouer", {depart, arrivee, promo});
  choisie = null;
  await rafraichir();
  if (!r.ok) dessiner();
}

function demanderPromotion(depart, arrivee) {
  const blanc = etat.humain === "w";
  document.getElementById("promoChoix").innerHTML = ["q", "r", "b", "n"].map(p =>
    `<button data-p="${p}">${piece(blanc ? p.toUpperCase() : p)}</button>`).join("");
  document.getElementById("promo").style.display = "flex";
  enAttente = {depart, arrivee};
}

document.getElementById("promoChoix").addEventListener("click", ev => {
  const b = ev.target.closest("button");
  if (!b || !enAttente) return;
  document.getElementById("promo").style.display = "none";
  jouer(enAttente.depart, enAttente.arrivee, b.dataset.p);
  enAttente = null;
});

document.getElementById("plateau").addEventListener("click", ev => {
  const c = ev.target.closest(".case");
  if (!c || !etat || vue !== null) return;
  const nom = c.dataset.c;
  if (choisie && etat.legaux[choisie] && etat.legaux[choisie].includes(nom)) {
    const p = etat.pieces[choisie];
    if (p && p.toLowerCase() === "p" && (nom[1] === "8" || nom[1] === "1")) demanderPromotion(choisie, nom);
    else jouer(choisie, nom);
    return;
  }
  choisie = etat.legaux[nom] && nom !== choisie ? nom : null;
  dessiner();
});

document.getElementById("nouvelle").addEventListener("click", async () => {
  let couleur = document.getElementById("couleur").value;
  if (couleur === "r") couleur = Math.random() < 0.5 ? "w" : "b";
  const modele = document.getElementById("modele").value;
  await appel("/api/nouvelle", {couleur, modele, niveau: +document.getElementById("niveau").value});
  choisie = null; vue = null; posVue = null; vusPrises = null; modeleAffiche = null;
  await rafraichir();
  dessiner();
});
document.getElementById("annuler").addEventListener("click", async () => { await appel("/api/annuler", {}); choisie = null; vue = null; posVue = null; vusPrises = null; await rafraichir(); dessiner(); });
document.getElementById("niveau").addEventListener("change", e => appel("/api/niveau", {niveau: +e.target.value}));
document.getElementById("modele").addEventListener("change", async e => {
  const modele = e.target.value;
  remplirNiveaux(modele, 1);
  modeleAffiche = modele;
  await appel("/api/modele", {modele});
  await rafraichir();
});

document.getElementById("coups").addEventListener("click", ev => {
  const td = ev.target.closest("td[data-ply]");
  if (td) voir(+td.dataset.ply);
});
document.getElementById("vue").addEventListener("click", ev => { if (ev.target.id === "retour") voir(etat.coups.length); });
document.addEventListener("keydown", ev => {
  if (!etat || ["SELECT", "INPUT", "TEXTAREA"].includes(ev.target.tagName)) return;
  const n = etat.coups.length, courant = vue === null ? n : vue;
  if (ev.key === "ArrowLeft") { ev.preventDefault(); voir(courant - 1); }
  else if (ev.key === "ArrowRight" && vue !== null) { ev.preventDefault(); voir(courant + 1); }
  else if (ev.key === "Escape" && vue !== null) voir(n);
});
document.getElementById("exporter").addEventListener("click", () => { window.location.href = "/api/export"; });
document.getElementById("copier").addEventListener("click", async ev => {
  const texte = await (await fetch("/api/pgn")).text();
  try {
    await navigator.clipboard.writeText(texte);
    ev.target.textContent = "Copié !";
    setTimeout(() => ev.target.textContent = "Copier la partie", 1500);
  } catch (err) {
    window.prompt("Copie la partie (Ctrl+C) :", texte);
  }
});

rafraichir();
setInterval(rafraichir, 600);
</script>
</body>
</html>
"""


def page_html():
    options = "".join(
        f'<option value="{mid}"{" selected" if mid == MODELE_DEFAUT else ""}>{cfg["libelle"]}</option>'
        for mid, cfg in MODELES.items()
    )
    modeles_js = json.dumps({
        mid: {
            "libelle": cfg["libelle"],
            "niveaux": [{"valeur": n, "label": nc["label"]} for n, nc in cfg["niveaux"].items()],
        }
        for mid, cfg in MODELES.items()
    }, ensure_ascii=False)
    return (PAGE.replace("%%PIECES%%", "".join(chess.svg.PIECES.values()))
                .replace("%%MODELES_OPTIONS%%", options)
                .replace("%%MODELES_JSON%%", modeles_js))


class Serveur(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/etat":
            return self._json(partie.etat())
        if url.path == "/api/position":
            try:
                ply = int(parse_qs(url.query).get("ply", ["0"])[0])
            except ValueError:
                ply = 0
            return self._json(partie.position(max(0, ply)))
        if url.path == "/api/export":
            corps = json.dumps(partie.export(), ensure_ascii=False, indent=1).encode()
            nom = "pawn_partie_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{nom}"')
            self.send_header("Content-Length", str(len(corps)))
            self.end_headers()
            self.wfile.write(corps)
            return
        if url.path == "/api/pgn":
            texte = partie.pgn().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(texte)))
            self.end_headers()
            self.wfile.write(texte)
            return
        if url.path in ("/", "/index.html"):
            page = page_html().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
            return
        self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            d = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._json({"ok": False}, 400)
        if self.path == "/api/jouer":
            return self._json({"ok": partie.jouer(d.get("depart", ""), d.get("arrivee", ""), d.get("promo"))})
        if self.path == "/api/nouvelle":
            modele = d.get("modele") if d.get("modele") in MODELES else MODELE_DEFAUT
            partie.nouvelle(chess.BLACK if d.get("couleur") == "b" else chess.WHITE, modele, int(d.get("niveau", NIVEAU_DEFAUT)))
            return self._json({"ok": True})
        if self.path == "/api/annuler":
            partie.annuler()
            return self._json({"ok": True})
        if self.path == "/api/niveau":
            partie.regler_niveau(int(d.get("niveau", NIVEAU_DEFAUT)))
            return self._json({"ok": True})
        if self.path == "/api/modele":
            partie.regler_modele(d.get("modele", MODELE_DEFAUT))
            return self._json({"ok": True})
        self.send_error(404)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    serveur = None
    for port in range(PORT, PORT + 10):
        try:
            serveur = ThreadingHTTPServer(("127.0.0.1", port), Serveur)
            break
        except OSError:
            continue
    if serveur is None:
        sys.exit("Aucun port libre entre 8765 et 8774.")
    url = f"http://127.0.0.1:{serveur.server_address[1]}"
    print(f"P.A.W.N. est prêt : {url}\n(Ctrl+C dans cette fenêtre pour arrêter)")
    webbrowser.open(url)
    try:
        serveur.serve_forever()
    except KeyboardInterrupt:
        print("\nÀ bientôt !")