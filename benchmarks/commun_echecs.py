"""Code commun aux notebooks de benchmark échecs (PAWN, PAWN big, contrôle, SPAWN/PWN).

Tout vient de pawn_app.py : même encodage 69 octets, même recherche « arbre complet »,
même alpha-bêta + quiescence. Seule différence : pas de bruit ni de coups au hasard,
on veut mesurer la force réelle.
"""
import glob
import json
import os
import shutil
import stat
import subprocess
import tarfile
import time
import urllib.request

import chess
import chess.engine
import chess.polyglot
import numpy as np
import onnxruntime as ort

DEPOT = "LiamLitle/ia-de-liam"
BRANCHE = "main"
LFS = f"https://media.githubusercontent.com/media/{DEPOT}/{BRANCHE}/"
DOSSIER_POIDS = os.environ.get("PAWN_POIDS", "/kaggle/working/poids" if os.path.isdir("/kaggle") else "poids")
MATE = 100_000

# nom -> chemin dans le dépôt
RESEAUX = {
    "PAWN": "Pawn/pawn.onnx",
    "PAWN_BIG": "PawnBig-V1/pawn_big.onnx",
    "CONTROLE": "PawnBig-controlleur/pawn_big_controle.onnx",
    "SPAWN": "PWN-soluce/pawn_soluce.onnx",
    "POWN": "Pown/pown.onnx",
}


def poids(chemin_depot):
    """cherche le fichier dans /kaggle/input (dataset ajouté au notebook), sinon le télécharge depuis GitHub LFS"""
    nom = os.path.basename(chemin_depot)
    for racine in ("/kaggle/input", DOSSIER_POIDS):
        trouves = glob.glob(os.path.join(racine, "**", nom), recursive=True)
        trouves = [t for t in trouves if os.path.getsize(t) > 1000]  # pas un pointeur LFS
        if trouves:
            return trouves[0]
    os.makedirs(DOSSIER_POIDS, exist_ok=True)
    dest = os.path.join(DOSSIER_POIDS, nom)
    print(f"téléchargement de {chemin_depot} ...")
    urllib.request.urlretrieve(LFS + chemin_depot, dest)
    if os.path.getsize(dest) < 1000:
        raise RuntimeError(f"{dest} ressemble à un pointeur LFS, pas au vrai fichier")
    return dest


GPU_ID = 0       # carte utilisée par ce processus (Kaggle « GPU T4 x2 » en a deux)
GPU_MEM_MO = 0   # plafond mémoire GPU par réseau (0 = pas de plafond) ; fixé quand plusieurs processus partagent le GPU
_CUDA_OK = None  # None = pas encore essayé, False = échec -> on reste sur CPU sans réessayer


def providers(gpu=True):
    dispo = ort.get_available_providers()
    if gpu and _CUDA_OK is not False and "CUDAExecutionProvider" in dispo:
        try:
            ort.preload_dlls()  # charge CUDA/cuDNN depuis les paquets pip nvidia-* (déjà là avec torch sur Kaggle)
        except Exception:
            pass
        # HEURISTIC : cuDNN choisit ses algos sans essayer ceux qui demandent d'énormes
        # zones de travail (sinon des blocs de 150 Mo dépassent le plafond mémoire)
        opts = {"arena_extend_strategy": "kSameAsRequested", "cudnn_conv_algo_search": "HEURISTIC",
                "cudnn_conv_use_max_workspace": "0", "device_id": GPU_ID}
        if GPU_MEM_MO:
            opts["gpu_mem_limit"] = GPU_MEM_MO * 1024 * 1024
        return [("CUDAExecutionProvider", opts), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


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


class Evaluateur:
    """un réseau ONNX : liste de FEN -> centipions du point de vue du camp au trait"""

    def __init__(self, nom, gpu=True, threads=0):
        self.nom = nom
        self.so = ort.SessionOptions()
        if threads:
            self.so.intra_op_num_threads = threads
            self.so.inter_op_num_threads = 1
        global _CUDA_OK
        prov = providers(gpu)
        self.session = ort.InferenceSession(poids(RESEAUX[nom]), self.so, providers=prov)
        self.sur_gpu = "CUDAExecutionProvider" in self.session.get_providers()
        if prov[0] != "CPUExecutionProvider":
            _CUDA_OK = self.sur_gpu
            if not _CUDA_OK:
                print("⚠️ GPU indisponible pour onnxruntime : on continue sur CPU (plus lent mais résultats identiques)")
                ort.set_default_logger_severity(4)
        self.nb_evals = 0
        self.cache = {}

    def evals(self, fens, lot=4096):
        if not fens:
            return np.zeros(0, dtype=np.float32)
        if self.sur_gpu and GPU_MEM_MO:
            lot = min(lot, 512)  # petits lots quand la mémoire GPU est plafonnée
        res = []
        for i in range(0, len(fens), lot):
            x = np.frombuffer(b"".join(map(enc_fen, fens[i:i + lot])), np.uint8).reshape(-1, 69).copy()
            n = len(x)
            if self.sur_gpu and GPU_MEM_MO:
                # taille de lot arrondie à une puissance de 2 : peu de formes différentes, donc
                # onnxruntime réutilise ses blocs mémoire au lieu d'en allouer de nouveaux à chaque coup
                taille = max(64, 1 << (n - 1).bit_length())
                x = np.concatenate([x, np.zeros((taille - n, 69), np.uint8)]) if taille > n else x
            try:
                res.append(self.session.run(None, {"board": x})[0][:n])
            except Exception as e:
                if not self.sur_gpu:
                    raise
                # GPU plein : ce réseau passe sur CPU pour la suite (mêmes résultats, plus lent)
                print(f"⚠️ {self.nom} : erreur GPU ({str(e)[:80]}...) -> CPU")
                self.session = ort.InferenceSession(poids(RESEAUX[self.nom]), self.so, providers=["CPUExecutionProvider"])
                self.sur_gpu = False
                res.append(self.session.run(None, {"board": x})[0])
        self.nb_evals += len(fens)
        return np.concatenate(res)

    def eval1(self, fen):
        # l'alpha-bêta réévalue souvent les mêmes positions en quiescence : le réseau est
        # déterministe, donc un cache ne change rien au résultat, juste la vitesse
        v = self.cache.get(fen)
        if v is None:
            if len(self.cache) > 2_000_000:
                self.cache.clear()
            v = self.cache[fen] = float(self.evals([fen])[0])
        return v


# -- P.O.W.N. (NNUE HalfKAv2_hm) : autre encodage d'entrée, même interface -------------

def _bucket_pown(roi, miroir):
    f = chess.square_file(roi)
    if miroir:
        f = 7 - f
    return chess.square_rank(roi) * 4 + f


def features_pown(board, couleur):
    """doit rester identique à features() de Pown/commun_pown.py (vérifié par test)"""
    roi = board.king(couleur)
    miroir = chess.square_file(roi) >= 4
    b = _bucket_pown(roi, miroir)
    idx = []
    for case, piece in board.piece_map().items():
        if piece.piece_type == chess.KING and piece.color == couleur:
            continue
        f, r = chess.square_file(case), chess.square_rank(case)
        if miroir:
            f = 7 - f
        nous = piece.color == couleur
        t = (piece.piece_type - 1) if nous else 5 + (piece.piece_type - 1)
        idx.append(b * 704 + (r * 8 + f) * 11 + t)
    return idx


class EvaluateurPOWN(Evaluateur):
    """P.O.W.N. : ONNX à 4 entrées (index + décalages des deux points de vue) -> logit ; cp = 400 x logit"""

    def __init__(self, nom, gpu=False, threads=0):
        self.nom = nom
        self.so = ort.SessionOptions()
        if threads:
            self.so.intra_op_num_threads = threads
            self.so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(poids(RESEAUX[nom]), self.so, providers=["CPUExecutionProvider"])
        self.sur_gpu = False
        self.nb_evals = 0
        self.cache = {}

    def evals(self, fens, lot=4096):
        if not fens:
            return np.zeros(0, dtype=np.float32)
        res = []
        for i in range(0, len(fens), lot):
            idx_n, dec_n, idx_e, dec_e = [], [], [], []
            for fen in fens[i:i + lot]:
                b = chess.Board(fen)
                n, e = features_pown(b, b.turn), features_pown(b, not b.turn)
                dec_n.append(len(idx_n)); idx_n += n
                dec_e.append(len(idx_e)); idx_e += e
            sortie = self.session.run(None, {
                "idx_nous": np.array(idx_n, dtype=np.int64), "dec_nous": np.array(dec_n, dtype=np.int64),
                "idx_eux": np.array(idx_e, dtype=np.int64), "dec_eux": np.array(dec_e, dtype=np.int64)})[0]
            res.append(400.0 * sortie)
        self.nb_evals += len(fens)
        return np.concatenate(res)


# -- recherche « arbre complet » (PAWN, PAWN big, contrôle, SPAWN) -----------------

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


def choisir_arbre(board, ev, depth):
    feuilles = []
    racine = arbre(board, depth, 0, feuilles)
    v = ev.evals(feuilles) if feuilles else []
    notes = [(-valeur(c, v), mv) for mv, c in racine[1]]
    return max(notes, key=lambda t: t[0])[1]


# -- recherche alpha-bêta + quiescence (PWN) --------------------------------------

VAL_PIECE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def trier_coups(board, coups, priorite):
    def cle(mv):
        if mv == priorite:
            return 10_000
        if board.is_capture(mv):
            prise = board.piece_type_at(mv.to_square) or chess.PAWN
            attaquant = board.piece_type_at(mv.from_square)
            return 1000 + VAL_PIECE[prise] * 10 - VAL_PIECE[attaquant]
        if board.gives_check(mv):
            return 500
        return 0
    return sorted(coups, key=cle, reverse=True)


def quiescence(board, ev, alpha, beta, prof_q):
    stand_pat = ev.eval1(board.fen())
    if stand_pat >= beta:
        return beta
    alpha = max(alpha, stand_pat)
    if prof_q <= 0:
        return alpha
    captures = [m for m in board.legal_moves if board.is_capture(m)]
    for mv in trier_coups(board, captures, None):
        board.push(mv)
        score = -quiescence(board, ev, -beta, -alpha, prof_q - 1)
        board.pop()
        if score >= beta:
            return beta
        alpha = max(alpha, score)
    return alpha


def negamax(board, ev, profondeur, alpha, beta, ply, prof_q, tt, coup_tt):
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
        return quiescence(board, ev, alpha, beta, prof_q)
    meilleur, meilleur_coup = -MATE - 1, None
    for mv in trier_coups(board, list(board.legal_moves), coup_tt.get(hash_)):
        board.push(mv)
        score = -negamax(board, ev, profondeur - 1, -beta, -alpha, ply + 1, prof_q, tt, coup_tt)
        board.pop()
        if score > meilleur:
            meilleur, meilleur_coup = score, mv
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    drapeau = "exact" if alpha0 < meilleur < beta else ("min" if meilleur >= beta else "max")
    tt[hash_] = (profondeur, meilleur, drapeau, meilleur_coup)
    if meilleur_coup is not None:
        coup_tt[hash_] = meilleur_coup
    return meilleur


def choisir_alphabeta(board, ev, depth, prof_q=4):
    tt, coup_tt = {}, {}
    meilleur_coup = None
    for d in range(1, depth + 1):
        alpha, beta = -MATE - 1, MATE + 1
        hash_ = chess.polyglot.zobrist_hash(board)
        meilleur_score, meilleur_du_tour = -MATE - 1, None
        for mv in trier_coups(board, list(board.legal_moves), coup_tt.get(hash_)):
            board.push(mv)
            score = -negamax(board, ev, d - 1, -beta, -alpha, 1, prof_q, tt, coup_tt)
            board.pop()
            if score > meilleur_score:
                meilleur_score, meilleur_du_tour = score, mv
            alpha = max(alpha, score)
        meilleur_coup = meilleur_du_tour
        coup_tt[hash_] = meilleur_coup
    return meilleur_coup


# -- recherche « équilibrée » : alpha-bêta améliorée + durée fixée par coup (POWN@t5 = 5 s par coup) ----

class _Temps(Exception):
    pass


class Recherche:
    """Alpha-bêta avec approfondissement itératif jusqu'à épuisement du temps. Par rapport à choisir_alphabeta :
    tri des coups (table, captures MVV-LVA, tueurs, historique), quiescence qui écarte les mauvaises captures,
    extension d'échec, coup nul, réductions (LMR). Le coup rendu est celui de la dernière profondeur terminée."""

    def __init__(self, ev, secondes, prof_max=20, prof_q=4):
        self.ev, self.secondes, self.prof_max, self.prof_q = ev, secondes, prof_max, prof_q
        self.tt, self.tueurs, self.histo, self.noeuds = {}, {}, {}, 0
        self.fin = 0.0
        self.prof_atteinte = 0

    def _horloge(self):
        self.noeuds += 1
        if self.noeuds & 15 == 0 and time.perf_counter() > self.fin:
            raise _Temps()

    def _cle(self, board, mv, coup_tt, ply):
        if mv == coup_tt:
            return 1_000_000
        if board.is_capture(mv) or mv.promotion:
            victime = VAL_PIECE[board.piece_type_at(mv.to_square) or chess.PAWN]
            return 100_000 + victime * 100 - VAL_PIECE[board.piece_type_at(mv.from_square)] + (50 if mv.promotion else 0)
        if mv in self.tueurs.get(ply, ()):
            return 90_000
        return self.histo.get((board.turn, mv.from_square, mv.to_square), 0)

    def quiescence(self, board, alpha, beta, prof):
        self._horloge()
        en_echec = board.is_check()
        if en_echec:
            coups = list(board.legal_moves)
            if not coups:
                return -MATE
            gain = -MATE
        else:
            gain = self.ev.eval1(board.fen())
            if gain >= beta:
                return gain
            alpha = max(alpha, gain)
            if prof <= 0:
                return gain
            coups = []
            for m in board.legal_moves:
                if board.is_capture(m) or m.promotion:
                    victime = VAL_PIECE[board.piece_type_at(m.to_square) or chess.PAWN]
                    # delta : même en gagnant la pièce, on reste loin de alpha
                    if gain + 100 * victime + 200 < alpha and not m.promotion:
                        continue
                    # mauvaise capture : pièce plus chère qui prend une pièce moins chère protégée
                    if (VAL_PIECE[board.piece_type_at(m.from_square)] > victime and not m.promotion
                            and board.is_attacked_by(not board.turn, m.to_square)):
                        continue
                    coups.append(m)
        coups.sort(key=lambda m: self._cle(board, m, None, 0), reverse=True)
        for mv in coups:
            board.push(mv)
            v = -self.quiescence(board, -beta, -alpha, prof - 1)
            board.pop()
            if v > gain:
                gain = v
                if v > alpha:
                    alpha = v
                    if alpha >= beta:
                        break
        return gain

    def negamax(self, board, depth, alpha, beta, ply, nul_ok=True):
        self._horloge()
        coups = list(board.legal_moves)
        if not coups:
            return -(MATE - ply) if board.is_check() else 0
        if ply > 0 and (board.is_insufficient_material() or board.is_repetition(2) or board.halfmove_clock >= 100):
            return 0
        alpha0 = alpha
        h = chess.polyglot.zobrist_hash(board)
        e = self.tt.get(h)
        coup_tt = e[3] if e else None
        if e and e[0] >= depth and e[2] != "none":
            _, v, drapeau, _ = e
            if drapeau == "exact":
                return v
            if drapeau == "min":
                alpha = max(alpha, v)
            else:
                beta = min(beta, v)
            if alpha >= beta:
                return v
        en_echec = board.is_check()
        if en_echec and ply < 40:
            depth += 1                       # extension d'échec
        if depth <= 0:
            return self.quiescence(board, alpha, beta, self.prof_q)
        # coup nul : si même en passant son tour on reste >= beta, on coupe (pas en finale de pions : zugzwang)
        if (nul_ok and depth >= 3 and not en_echec and beta < MATE - 1000
                and any(board.pieces(t, board.turn) for t in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN))):
            board.push(chess.Move.null())
            v = -self.negamax(board, depth - 3, -beta, -beta + 1, ply + 1, False)
            board.pop()
            if v >= beta:
                return beta
        coups.sort(key=lambda m: self._cle(board, m, coup_tt, ply), reverse=True)
        meilleur, meilleur_coup = -MATE - 1, None
        for i, mv in enumerate(coups):
            calme = not board.is_capture(mv) and not mv.promotion
            donne_echec = board.gives_check(mv)
            board.push(mv)
            if i >= 4 and depth >= 3 and calme and not en_echec and not donne_echec:
                v = -self.negamax(board, depth - 2, -alpha - 1, -alpha, ply + 1)      # réduit, fenêtre nulle
                if v > alpha:
                    v = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            else:
                v = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            if v > meilleur:
                meilleur, meilleur_coup = v, mv
            if v > alpha:
                alpha = v
            if alpha >= beta:
                if calme:
                    t = self.tueurs.setdefault(ply, [])
                    if mv not in t:
                        t.insert(0, mv)
                        del t[2:]
                    k = (board.turn, mv.from_square, mv.to_square)
                    self.histo[k] = self.histo.get(k, 0) + depth * depth
                break
        if abs(meilleur) > MATE - 1000:
            drapeau = "none"                 # score de mat dépendant de la distance : on ne le réutilise pas
        else:
            drapeau = "exact" if alpha0 < meilleur < beta else ("min" if meilleur >= beta else "max")
        self.tt[h] = (depth, meilleur, drapeau, meilleur_coup)
        return meilleur

    def choisir(self, board):
        b = board.copy(stack=True)
        coups = list(b.legal_moves)
        if len(coups) == 1:
            return coups[0]
        debut = time.perf_counter()
        self.fin = debut + self.secondes
        meilleur_coup, ordre = coups[0], coups
        for d in range(1, self.prof_max + 1):
            try:
                alpha, beta, meilleur_score, du_tour = -MATE - 1, MATE + 1, -MATE - 1, None
                for mv in ordre:
                    b.push(mv)
                    v = -self.negamax(b, d - 1, -beta, -alpha, 1)
                    b.pop()
                    if v > meilleur_score:
                        meilleur_score, du_tour = v, mv
                    alpha = max(alpha, v)
            except _Temps:
                break
            meilleur_coup, self.prof_atteinte = du_tour, d
            ordre = [du_tour] + [m for m in ordre if m != du_tour]    # le meilleur d'abord à la profondeur suivante
            if abs(meilleur_score) > MATE - 1000:
                break                                                # mat trouvé : inutile d'aller plus loin
            if time.perf_counter() - debut > 0.4 * self.secondes:
                break                                                # la profondeur suivante coûterait plus que ce qui reste
        return meilleur_coup


DERNIERE = {}   # profondeur atteinte et nombre de noeuds de la dernière recherche (pour les statistiques)


def choisir_temps(board, ev, secondes):
    r = Recherche(ev, secondes)
    mv = r.choisir(board)
    DERNIERE.update(profondeur=r.prof_atteinte, noeuds=r.noeuds)
    return mv


# -- joueurs ------------------------------------------------------------------------
# un « joueur » = un réseau + un algorithme de recherche + une profondeur.
# ex : "PAWN_BIG@arbre2", "SPAWN@arbre1", "PWN@ab3" (PWN = poids SPAWN + alpha-bêta).

def decoder(joueur):
    nom, rech = joueur.split("@")
    if nom == "PWN":
        nom = "SPAWN"
    if rech.startswith("arbre"):
        return nom, "arbre", int(rech[5:])
    if rech.startswith("ab"):
        return nom, "ab", int(rech[2:])
    if rech.startswith("t"):          # ex. "POWN@t5" : recherche équilibrée, 5 secondes par coup
        return nom, "temps", float(rech[1:])
    raise ValueError(joueur)


_EVALS = {}
_SF = {}
GPU = True
THREADS = 0
SF_CHEMIN = None
SF_TEMPS = 0.1


def evaluateur(nom, gpu=None):
    gpu = GPU if gpu is None else gpu
    if (nom, gpu) not in _EVALS:
        classe = EvaluateurPOWN if nom == "POWN" else Evaluateur
        _EVALS[(nom, gpu)] = classe(nom, gpu=gpu, threads=THREADS)
    return _EVALS[(nom, gpu)]


def configurer(gpu=True, threads=0, sf_chemin=None, sf_temps=0.1):
    """à appeler dans chaque processus fils avant de jouer"""
    global GPU, THREADS, SF_CHEMIN, SF_TEMPS
    GPU, THREADS, SF_CHEMIN, SF_TEMPS = gpu, threads, sf_chemin, sf_temps
    _EVALS.clear()
    _SF.clear()


def stockfish(cfg):
    """cfg = "elo1500" (UCI_LimitStrength) ou "skill0" (Skill Level)"""
    if cfg not in _SF:
        e = chess.engine.SimpleEngine.popen_uci(SF_CHEMIN)
        opts = {"Threads": 1, "Hash": 16}
        if cfg.startswith("elo"):
            opts.update({"UCI_LimitStrength": True, "UCI_Elo": int(cfg[3:])})
        elif cfg.startswith("skill"):
            opts["Skill Level"] = int(cfg[5:])
        e.configure(opts)
        _SF[cfg] = e
    return _SF[cfg]


def fermer_stockfish():
    for e in _SF.values():
        e.quit()
    _SF.clear()


def coup(joueur, board):
    if joueur.startswith("SF@"):
        return stockfish(joueur[3:]).play(board, chess.engine.Limit(time=SF_TEMPS)).move
    nom, rech, depth = decoder(joueur)
    b = board.copy(stack=True)
    if rech == "arbre":
        return choisir_arbre(b, evaluateur(nom), depth)
    if rech == "temps":
        return choisir_temps(b, evaluateur(nom, gpu=False), depth)
    # l'alpha-bêta évalue une position à la fois : sur Kaggle la latence CPU (~6 ms)
    # est deux fois plus basse que celle du GPU (~14 ms), donc on reste sur CPU
    return choisir_alphabeta(b, evaluateur(nom, gpu=False), depth)


# -- Stockfish -------------------------------------------------------------------------

URLS_STOCKFISH = [
    "https://github.com/official-stockfish/Stockfish/releases/download/sf_17.1/stockfish-ubuntu-x86-64-avx2.tar",
    "https://github.com/official-stockfish/Stockfish/releases/download/sf_17/stockfish-ubuntu-x86-64-avx2.tar",
    "https://github.com/official-stockfish/Stockfish/releases/download/sf_16.1/stockfish-ubuntu-x86-64-avx2.tar",
]


def rendre_executable(binaire):
    """/kaggle/working interdit d'exécuter des programmes : on copie Stockfish ailleurs
    et on garde le premier emplacement où il démarre vraiment"""
    for dest in ("/usr/local/bin/stockfish-pawn", "/tmp/stockfish-pawn", binaire):
        try:
            if dest != binaire:
                shutil.copy(binaire, dest)
            os.chmod(dest, 0o755)
            subprocess.run([dest, "quit"], check=True, timeout=10, capture_output=True)
            return dest
        except Exception as e:
            print("Stockfish ne démarre pas depuis", dest, ":", e)
    raise RuntimeError("Stockfish ne peut être exécuté nulle part")


def installer_stockfish(dossier=None):
    dossier = dossier or os.path.join(os.path.dirname(os.path.abspath(DOSSIER_POIDS)), "stockfish")
    deja = glob.glob(os.path.join(dossier, "**", "stockfish-ubuntu-*"), recursive=True)
    deja = [d for d in deja if os.path.isfile(d) and not d.endswith(".tar")]
    if deja:
        return rendre_executable(deja[0])
    os.makedirs(dossier, exist_ok=True)
    for url in URLS_STOCKFISH:
        try:
            tar = os.path.join(dossier, "sf.tar")
            urllib.request.urlretrieve(url, tar)
            with tarfile.open(tar) as t:
                t.extractall(dossier)
            binaire = [d for d in glob.glob(os.path.join(dossier, "**", "stockfish-ubuntu-*"), recursive=True)
                       if os.path.isfile(d) and not d.endswith(".tar")][0]
            return rendre_executable(binaire)
        except Exception as e:
            print("échec", url, e)
    if shutil.which("stockfish") is None:
        subprocess.run("apt-get -qq install -y stockfish", shell=True)
    for p in (shutil.which("stockfish"), "/usr/games/stockfish"):
        if p and os.path.exists(p):
            return p
    raise RuntimeError("impossible d'installer Stockfish")


# -- puzzles Lichess ---------------------------------------------------------------

def resoudre_puzzle(joueur, fen, moves):
    """format Lichess : FEN avant le coup adverse, moves[0] = coup adverse, puis solution.
    Comme sur Lichess, un mat est toujours accepté même si ce n'est pas le coup attendu.
    renvoie (résolu, premier_coup_bon, nb_coups_joués, secondes)"""
    b = chess.Board(fen)
    moves = moves.split()
    b.push_uci(moves[0])
    premier, joues, t0 = None, 0, time.perf_counter()
    for i in range(1, len(moves), 2):
        attendu = chess.Move.from_uci(moves[i])
        mv = coup(joueur, b)
        joues += 1
        b.push(mv)
        mat = b.is_checkmate()
        b.pop()
        ok = mv == attendu or mat
        if premier is None:
            premier = ok
        if not ok:
            return False, premier, joues, time.perf_counter() - t0
        if mat:
            break
        b.push(attendu)
        if i + 1 < len(moves):
            b.push_uci(moves[i + 1])
    return True, premier, joues, time.perf_counter() - t0


# -- parties -------------------------------------------------------------------------

def ecrire_suivi(suivi, blancs, noirs, b, ouverture, debut):
    # position en cours dans un petit fichier JSON, lu par le notebook pour l'afficher en direct
    dernier = b.peek() if b.move_stack else None
    san = ""
    if dernier:
        prec = b.copy()
        prec.pop()
        san = prec.san(dernier)
    tmp = suivi + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"blancs": blancs, "noirs": noirs, "fen": b.fen(), "dernier": dernier.uci() if dernier else "",
                   "san": san, "ply": b.ply(), "ouverture": ouverture, "debut": debut}, f)
    os.replace(tmp, suivi)


def jouer_partie(blancs, noirs, ouverture, max_plies=300, suivi=None, nom_ouverture=""):
    """ouverture = liste de coups UCI joués d'office ; renvoie un dict (résultat, pgn, temps)"""
    import chess.pgn
    b = chess.Board()
    positions = []  # chaque coup : qui, coup, position obtenue, temps de réflexion
    for u in ouverture:
        mv = chess.Move.from_uci(u)
        positions.append({"ply": b.ply() + 1, "joueur": "ouverture", "san": b.san(mv), "uci": u, "secondes": 0.0})
        b.push(mv)
        positions[-1]["fen"] = b.fen()
    temps = {blancs: 0.0, noirs: 0.0}
    nb = {blancs: 0, noirs: 0}
    debut = time.time()
    if suivi:
        ecrire_suivi(suivi, blancs, noirs, b, nom_ouverture, debut)
    while not b.is_game_over(claim_draw=True) and b.ply() < max_plies:
        j = blancs if b.turn == chess.WHITE else noirs
        t0 = time.perf_counter()
        mv = coup(j, b)
        dt = time.perf_counter() - t0
        temps[j] += dt
        nb[j] += 1
        positions.append({"ply": b.ply() + 1, "joueur": j, "san": b.san(mv), "uci": mv.uci(), "secondes": round(dt, 3)})
        b.push(mv)
        positions[-1]["fen"] = b.fen()
        if suivi:
            ecrire_suivi(suivi, blancs, noirs, b, nom_ouverture, debut)
    res = b.result(claim_draw=True) if b.is_game_over(claim_draw=True) else "1/2-1/2"
    fin = b.outcome(claim_draw=True)
    jeu = chess.pgn.Game.from_board(b)
    jeu.headers.update(Event="Benchmark P.A.W.N.", White=blancs, Black=noirs, Result=res)
    return {
        "blancs": blancs, "noirs": noirs, "resultat": res,
        "fin": fin.termination.name if fin else "MAX_PLIES",
        "plies": b.ply(), "pgn": str(jeu),
        "s_par_coup_blancs": temps[blancs] / max(1, nb[blancs]),
        "s_par_coup_noirs": temps[noirs] / max(1, nb[noirs]),
        "positions": positions,
    }


# -- tâches pour ProcessPoolExecutor (doivent être importables depuis ce module) -------

def nb_gpu():
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        return max(1, sum(l.startswith("GPU") for l in r.stdout.splitlines()))
    except Exception:
        return 1


def init_worker(gpu, threads, sf_chemin=None, sf_temps=0.1, gpu_mem_mo=None):
    # plusieurs processus x plusieurs réseaux sur un GPU : sans plafond, onnxruntime réserve
    # la mémoire trop largement et le T4 sature (BFCArena / CUBLAS_STATUS_ALLOC_FAILED).
    # Les processus sont répartis sur les GPU disponibles, avec un plafond par réseau.
    global GPU_MEM_MO, GPU_ID
    n = nb_gpu()
    GPU_ID = os.getpid() % n
    GPU_MEM_MO = gpu_mem_mo or (700 if n == 1 else 1200)
    configurer(gpu=gpu, threads=threads, sf_chemin=sf_chemin, sf_temps=sf_temps)


def tache_puzzle(args):
    joueur, pid, fen, moves = args
    ok, premier, joues, sec = resoudre_puzzle(joueur, fen, moves)
    return {"joueur": joueur, "PuzzleId": pid, "resolu": ok, "premier_coup": premier,
            "coups_joues": joues, "secondes": sec}


def tache_partie(args):
    blancs, noirs, ouverture, max_plies, id_ouv = args[:5]
    suivi = args[5] if len(args) > 5 else None   # dossier de suivi en direct (facultatif)
    fichier = None
    if suivi:
        os.makedirs(suivi, exist_ok=True)
        fichier = os.path.join(suivi, f"{blancs}_{noirs}_{id_ouv}.json".replace(" ", "-").replace("@", "-"))
    try:
        r = jouer_partie(blancs, noirs, ouverture, max_plies, fichier, id_ouv)
    finally:
        # un Stockfish ouvert garde un thread vivant qui empêche le processus fils de se terminer
        fermer_stockfish()
        if fichier and os.path.exists(fichier):
            os.remove(fichier)
    r["ouverture"] = id_ouv
    return r
