"""Code commun de P.O.W.N. (Pawn Optimized With NNUE).

Encodage HalfKAv2_hm (comme le NNUE de Stockfish, avec l'astuce du miroir gauche-droite),
réseau, flux de données (dump officiel Lichess) et boucle d'entraînement.
"""
import io
import json
import math
import os
import time

import chess
import torch
from torch import nn
from torch.utils.data import IterableDataset

# -- encodage HalfKAv2_hm -------------------------------------------------------------
# Pour une perspective donnée : le roi (a-d après miroir) choisit un "bucket" (32 possibles),
# puis chaque autre pièce (sauf notre roi, implicite dans le bucket) ajoute une combinaison
# (case, type de pièce). 11 types : 5 des nôtres (sans roi) + 6 de l'adversaire (avec son roi).
NUM_BUCKETS = 32
NUM_TYPES = 11
F = NUM_BUCKETS * 64 * NUM_TYPES  # 22 528


def _bucket(roi, miroir):
    f = chess.square_file(roi)
    if miroir:
        f = 7 - f
    return chess.square_rank(roi) * 4 + f


def features(board, couleur):
    """Liste des index actifs pour le camp `couleur`, vus depuis son propre roi."""
    roi = board.king(couleur)
    miroir = chess.square_file(roi) >= 4
    b = _bucket(roi, miroir)
    idx = []
    for case, piece in board.piece_map().items():
        if piece.piece_type == chess.KING and piece.color == couleur:
            continue  # notre roi : déjà dans le bucket, pas la peine de le redire
        f, r = chess.square_file(case), chess.square_rank(case)
        if miroir:
            f = 7 - f
        case_vue = r * 8 + f
        nous = piece.color == couleur
        t = (piece.piece_type - 1) if nous else 5 + (piece.piece_type - 1)
        idx.append(b * (64 * NUM_TYPES) + case_vue * NUM_TYPES + t)
    return idx


def exemple(fen, cp):
    """(idx camp au trait, idx camp adverse, cible sigmoid(cp/400)) — cp du point de vue du trait."""
    if len(fen.split(" ")) < 6:
        fen += " 0 1"  # le dump Lichess tronque le FEN aux 4 premiers champs
    board = chess.Board(fen)
    stm = board.turn
    cible = 1.0 / (1.0 + math.exp(-max(-3000, min(3000, cp)) / 400))
    return features(board, stm), features(board, not stm), cible


# -- réseau ----------------------------------------------------------------------------

class ReseauPOWN(nn.Module):
    def __init__(self, f=F, h=256):
        super().__init__()
        self.transformateur = nn.EmbeddingBag(f, h, mode="sum")
        self.c1 = nn.Linear(2 * h, 32)
        self.c2 = nn.Linear(32, 32)
        self.sortie = nn.Linear(32, 1)

    def forward(self, idx_nous, decal_nous, idx_eux, decal_eux):
        a = self.transformateur(idx_nous, decal_nous)
        b = self.transformateur(idx_eux, decal_eux)
        x = torch.clamp(torch.cat([a, b], dim=1), 0, 1)  # "clipped relu" façon NNUE
        x = torch.clamp(self.c1(x), 0, 1)
        x = torch.clamp(self.c2(x), 0, 1)
        return self.sortie(x).squeeze(-1)

    def eval_cp(self, fen):
        """Évalue une position (cp, point de vue du joueur au trait) — pour les tests à la main."""
        board = chess.Board(fen)
        n, e = features(board, board.turn), features(board, not board.turn)
        with torch.no_grad():
            logit = self.forward(torch.tensor(n), torch.tensor([0]), torch.tensor(e), torch.tensor([0]))
        return 400 * math.log(1 / (1 - torch.sigmoid(logit).item() + 1e-9) - 1) * -1


def rassembler(lot):
    """collate_fn : construit les tenseurs plats + décalages attendus par EmbeddingBag."""
    idx_nous, decal_nous, idx_eux, decal_eux, cibles = [], [0], [], [0], []
    for n, e, c in lot:
        idx_nous += n
        decal_nous.append(decal_nous[-1] + len(n))
        idx_eux += e
        decal_eux.append(decal_eux[-1] + len(e))
        cibles.append(c)
    return (torch.tensor(idx_nous, dtype=torch.long), torch.tensor(decal_nous[:-1], dtype=torch.long),
            torch.tensor(idx_eux, dtype=torch.long), torch.tensor(decal_eux[:-1], dtype=torch.long),
            torch.tensor(cibles, dtype=torch.float32))


# -- flux de données : dump officiel Lichess (FEN + évaluations Stockfish) -------------

URL_LICHESS_EVAL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"


def lignes_lichess(url=URL_LICHESS_EVAL):
    """Générateur de lignes JSON décompressées à la volée (aucun fichier entier téléchargé d'un coup)."""
    import requests
    import zstandard
    with requests.get(url, stream=True, timeout=30) as reponse:
        reponse.raise_for_status()
        flux = zstandard.ZstdDecompressor().stream_reader(reponse.raw)
        for ligne in io.TextIOWrapper(flux, encoding="utf-8"):
            yield ligne


def parser_ligne(ligne, profondeur_min=20):
    """Une ligne du dump -> (fen, cp) ou None si filtrée (mauvaise qualité / format inattendu)."""
    try:
        d = json.loads(ligne)
        meilleur = max(d["evals"], key=lambda e: e.get("depth", 0))
        if meilleur.get("depth", 0) < profondeur_min:
            return None
        pv = meilleur["pvs"][0]
        if "mate" in pv:
            cp = 3000 if pv["mate"] > 0 else -3000
        else:
            cp = pv["cp"]
        return d["fen"], cp
    except Exception:
        return None


class FluxPositions(IterableDataset):
    """Position par position, encodées à la volée, sans tout charger en mémoire."""

    def __init__(self, source_lignes, n_max, profondeur_min=20):
        self.source_lignes = source_lignes  # fonction sans argument -> générateur de lignes
        self.n_max = n_max
        self.profondeur_min = profondeur_min

    def __iter__(self):
        n = 0
        for ligne in self.source_lignes():
            if n >= self.n_max:
                return
            r = parser_ligne(ligne, self.profondeur_min)
            if r is None:
                continue
            fen, cp = r
            try:
                yield exemple(fen, cp)
            except Exception:
                continue  # FEN illisible ou position invalide : on ignore, on n'arrête pas tout
            n += 1


# -- sauvegarde / reprise ---------------------------------------------------------------

def sauvegarder(chemin, modele, opt, n_vues, secondes):
    tmp = chemin + ".tmp"
    torch.save({"modele": modele.state_dict(), "opt": opt.state_dict(),
                "n_vues": n_vues, "secondes": secondes}, tmp)
    os.replace(tmp, chemin)


def charger(chemin, modele, opt=None, device="cpu"):
    pt = torch.load(chemin, map_location=device)
    modele.load_state_dict(pt["modele"])
    if opt is not None and "opt" in pt:
        opt.load_state_dict(pt["opt"])
    return pt.get("n_vues", 0), pt.get("secondes", 0.0)
