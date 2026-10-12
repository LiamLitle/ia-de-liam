"""Génère le notebook Colab d'entraînement de P.O.W.N.

    python construire_pown.py

commun_pown.py est embarqué (via %%writefile) pour que le notebook soit autonome.
"""
import json
import os

ICI = os.path.dirname(os.path.abspath(__file__))


def notebook(cellules):
    cells = []
    for genre, texte in cellules:
        texte = texte.strip("\n")
        lignes = [l + "\n" for l in texte.split("\n")]
        lignes[-1] = lignes[-1].rstrip("\n")
        if genre == "md":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": lignes})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lignes})
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def ecrire(nom, cellules):
    with open(os.path.join(ICI, nom), "w", encoding="utf-8") as f:
        json.dump(notebook(cellules), f, ensure_ascii=False, indent=1)
    print("écrit", nom)


with open(os.path.join(ICI, "commun_pown.py"), encoding="utf-8") as f:
    COMMUN = f.read()

NB = [
    ("md", r"""
# 🧠♟️ P.O.W.N. — Pawn Optimized With NNUE  (version Kaggle)

Réseau d'évaluation façon NNUE : **HalfKAv2_hm** (roi × case × pièce, astuce du miroir),
22 528 → 256 (×2 perspectives) → 32 → 32 → 1, soit **≈ 5,78 M de paramètres**.
Données : dump officiel Lichess (FEN + évaluations Stockfish), en streaming.

## ✅ Avant de lancer sur Kaggle
1. **Accélérateur : GPU T4** (un seul suffit, le réseau est petit).
2. **Internet : ON** (panneau de droite → *Session options*).
3. Pour que ça tourne **sans garder l'onglet ouvert** : bouton **Save Version → Save & Run All (Commit)**.
   Le modèle sera dans l'onglet *Output* (`pown.pt`).
4. **Reprendre plus tard** : *Add Input* → ton ancien notebook (son Output) ; le code retrouve
   tout seul `pown.pt` et continue (optimiseur compris).

## ⚡ Ce qui est optimisé par rapport à la version Colab
- **Un seul téléchargement**, mais le JSON + l'encodage sont répartis sur **tous les cœurs CPU**,
  en avance sur le GPU (avant : tout se faisait l'un après l'autre sur un seul cœur).
- **Taux d'apprentissage décroissant** (cosinus) selon le temps imparti : le modèle finit bien « affiné ».
- Points de contrôle réguliers, arrêt propre avant la limite.
"""),
    ("code", r"""
!pip install -q chess zstandard
"""),
    ("code", r"""
import os, glob
SORTIE = "/kaggle/working"
os.makedirs(SORTIE, exist_ok=True)
print("CPU :", os.cpu_count(), "cœurs")
import requests
try:
    r = requests.head("https://database.lichess.org/lichess_db_eval.jsonl.zst", timeout=20)
    print("accès à Lichess :", r.status_code, "| taille :", round(int(r.headers.get("content-length", 0)) / 1e9, 1), "Go")
except Exception as e:
    print("❌ pas d'accès Internet — active-le dans Session options :", e)
"""),
    ("code", "%%writefile commun_pown.py\n" + COMMUN),
    ("md", r"""
## Réglages
"""),
    ("code", r"""
import sys, time, math
sys.path.insert(0, os.getcwd())
import torch
import commun_pown as pown

# Reprise : si un ancien pown.pt est ajouté en Input, on le retrouve tout seul.
anciens = sorted(glob.glob("/kaggle/input/**/pown.pt", recursive=True))
DOSSIER_DEPART = anciens[-1] if anciens else None
print("reprise depuis :", DOSSIER_DEPART)

NOM_SORTIE = "pown"
TIME_MIN = 290              # 4h50 — s'arrête proprement avant, avec sauvegarde
N_POS = 400_000_000         # plafond de positions (c'est le temps qui coupe en pratique)
PROFONDEUR_MIN = 20
TAILLE_LOT = 16384
NB_PROC = max(1, (os.cpu_count() or 2) - 1)   # processus d'encodage
LR = 1e-3 if DOSSIER_DEPART is None else 4e-4  # en reprise : on repart plus bas
LR_FIN = 0.03               # le LR descend jusqu'à LR * LR_FIN
SAUVEGARDE_SEC = 600
SEED = 42
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device :", DEVICE, "| processus d'encodage :", NB_PROC)
if DEVICE == "cpu":
    print("⚠️ pas de GPU : active un accélérateur (T4)")
"""),
    ("md", r"""
## Modèle
"""),
    ("code", r"""
modele = pown.ReseauPOWN().to(DEVICE)
opt = torch.optim.AdamW(modele.parameters(), lr=LR)
n_vues, secondes_avant = 0, 0.0
if DOSSIER_DEPART:
    n_vues, secondes_avant = pown.charger(DOSSIER_DEPART, modele, opt, DEVICE)
    for g in opt.param_groups:
        g["lr"] = LR
    print(f"reprise : {n_vues:,} positions déjà vues, {secondes_avant/60:.0f} min déjà passées")
print("paramètres :", sum(t.numel() for t in modele.parameters()))
"""),
    ("md", r"""
## Entraînement
Les lignes de log donnent : positions vues, perte, vitesse, minutes écoulées, taux d'apprentissage.
Si la vitesse reste basse, c'est la bande passante vers Lichess qui limite (pas le calcul).
"""),
    ("code", r"""
CHEMIN = f"{SORTIE}/{NOM_SORTIE}.pt"
perte_fn = torch.nn.BCEWithLogitsLoss()
budget = TIME_MIN * 60
t0 = time.time()
derniere_sauvegarde = t0
pertes, n_session = [], 0

for i, (idx_n, dec_n, idx_e, dec_e, cibles) in enumerate(
        pown.lots_paralleles(pown.lignes_lichess, N_POS, PROFONDEUR_MIN, TAILLE_LOT, NB_PROC)):
    ecoule = time.time() - t0
    if ecoule > budget:
        print("budget temps atteint, on s'arrête proprement")
        break
    lr = LR * (LR_FIN + (1 - LR_FIN) * 0.5 * (1 + math.cos(math.pi * ecoule / budget)))
    for g in opt.param_groups:
        g["lr"] = lr

    idx_n, dec_n = idx_n.to(DEVICE, non_blocking=True), dec_n.to(DEVICE, non_blocking=True)
    idx_e, dec_e = idx_e.to(DEVICE, non_blocking=True), dec_e.to(DEVICE, non_blocking=True)
    cibles = cibles.to(DEVICE, non_blocking=True)

    opt.zero_grad()
    perte = perte_fn(modele(idx_n, dec_n, idx_e, dec_e), cibles)
    perte.backward()
    opt.step()

    n_vues += cibles.shape[0]
    n_session += cibles.shape[0]
    pertes.append(perte.item())

    if i % 50 == 0:
        print(f"{n_vues:>12,} positions | perte {sum(pertes[-50:]) / len(pertes[-50:]):.4f} "
              f"| {n_session / (ecoule + 1e-9):,.0f} pos/s | {ecoule / 60:.0f} min | lr {lr:.1e}")

    if time.time() - derniere_sauvegarde > SAUVEGARDE_SEC:
        pown.sauvegarder(CHEMIN, modele, opt, n_vues, secondes_avant + time.time() - t0)
        derniere_sauvegarde = time.time()

pown.sauvegarder(CHEMIN, modele, opt, n_vues, secondes_avant + time.time() - t0)
print(f"terminé : {n_vues:,} positions vues au total, modèle sauvegardé dans {CHEMIN}")
"""),
    ("md", r"""
## Test de cohérence
"""),
    ("code", r"""
import chess

modele.eval()
print("position de départ :", round(modele.eval_cp(chess.STARTING_FEN), 1), "cp (attendu : proche de 0)")
print("blancs avec une Dame de plus :",
      round(modele.eval_cp("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"), 1), "cp (attendu : très positif)")
print("blancs sans leur Dame :",
      round(modele.eval_cp("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"), 1), "cp (attendu : très négatif)")
"""),
    ("md", r"""
### Et après ?
- **Continuer** : *Add Input* → l'Output de ce notebook, relance (reprise automatique, LR plus bas).
- **Le comparer aux autres** : il faudra l'export des poids pour le moteur C++ / un chargeur dans `commun_echecs.py`.
"""),
]

if __name__ == "__main__":
    ecrire("pown_train.ipynb", NB)
