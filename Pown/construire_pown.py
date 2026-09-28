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
    ("md", r'''
# 🧠♟️ P.O.W.N. — Pawn Optimized With NNUE

Un nouveau réseau d'évaluation, dans l'esprit du NNUE de Stockfish : **HalfKAv2_hm**
(encodage roi + case + pièce, avec l'astuce du miroir gauche-droite), pensé pour être
**beaucoup plus rapide** que PAWN/SPAWN/PWN une fois porté en C++ (mise à jour incrémentale
à chaque coup, au lieu de tout recalculer).

| | |
|---|---|
| Encodage | HalfKAv2_hm — 22 528 combinaisons possibles (roi × case × pièce) |
| Réseau | 22 528 → 256 (×2 perspectives) → 32 → 32 → 1 |
| Paramètres | **≈ 5,78 millions** |
| Données | dump officiel Lichess (FEN + évaluations Stockfish), en streaming |
| Objectif | ≈ 100 millions de positions, 2-3 passages |

⚠️ Cette v1 est volontairement simple (encodage "maison", pas un port exact du NNUE de
Stockfish) : l'important est que l'entraînement et l'inférence utilisent la même formule,
pas de coller au bit près à l'implémentation officielle.

## ⚙️ Spécifique à Colab (différent de Kaggle)

- **Monte ton Drive** (cellule ci-dessous) : contrairement à Kaggle, Colab ne garde rien
  automatiquement. Sans ça, tout est perdu si la session se déconnecte.
- **Limite de temps** (`TIME_MIN`) : l'entraînement s'arrête proprement à l'heure fixée et
  sauvegarde un point de contrôle. Pour continuer un autre jour, remets le chemin de ce
  point de contrôle dans `DOSSIER_DEPART`.
- Le clé Hugging Face (`HF_TOKEN`, dans les **Secrets** de Colab, icône 🔑) n'est **pas
  nécessaire** ici : on télécharge directement le fichier public de Lichess, sans compte.
  Le code la lit quand même si elle existe, pour un futur notebook qui en aurait besoin.
'''),
    ("code", r'''
!pip install -q chess zstandard
'''),
    ("code", r'''
from google.colab import drive
drive.mount("/content/drive")
import os
SORTIE = "/content/drive/MyDrive/pown"
os.makedirs(SORTIE, exist_ok=True)
print("les points de contrôle seront sauvegardés dans :", SORTIE)
'''),
    ("code", r'''
import os
try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    print("HF_TOKEN chargé (pas indispensable pour ce notebook)")
except Exception:
    print("pas de HF_TOKEN configuré — sans importance, on ne télécharge rien depuis Hugging Face ici")
'''),
    ("code", "%%writefile commun_pown.py\n" + COMMUN),
    ("md", r'''
## Réglages
'''),
    ("code", r'''
import sys, time
sys.path.insert(0, os.getcwd())
import torch
import commun_pown as pown

DOSSIER_DEPART = None       # ex : f"{SORTIE}/pown.pt" pour reprendre un entraînement précédent
NOM_SORTIE = "pown"
TIME_MIN = 330              # 5h30 ; s'arrête proprement avant, sauvegarde faite régulièrement
N_POS = 100_000_000         # positions visées par passage (le temps peut couper avant)
EPOQUES = 3                 # nombre de passages sur le flux (si le temps le permet)
PROFONDEUR_MIN = 20         # comme PAWN : on ignore les évaluations Stockfish trop peu profondes
TAILLE_LOT = 16384
LR = 1e-3
SAUVEGARDE_SEC = 600        # point de contrôle toutes les 10 min
SEED = 42
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device :", DEVICE)
if DEVICE == "cpu":
    print("⚠️ pas de GPU : Runtime → Modifier le type d'exécution → GPU (T4)")
'''),
    ("md", r'''
## Modèle (reprise si `DOSSIER_DEPART` est renseigné)
'''),
    ("code", r'''
modele = pown.ReseauPOWN().to(DEVICE)
opt = torch.optim.AdamW(modele.parameters(), lr=LR)
n_vues, secondes_avant = 0, 0.0
if DOSSIER_DEPART and os.path.exists(DOSSIER_DEPART):
    n_vues, secondes_avant = pown.charger(DOSSIER_DEPART, modele, opt, DEVICE)
    print(f"reprise : {n_vues:,} positions déjà vues, {secondes_avant/60:.0f} min déjà passées")
print("paramètres :", sum(t.numel() for t in modele.parameters()))
'''),
    ("md", r'''
## Entraînement (flux Lichess, coupure au temps imparti)
'''),
    ("code", r'''
from torch.utils.data import DataLoader

CHEMIN = f"{SORTIE}/{NOM_SORTIE}.pt"
perte_fn = torch.nn.BCEWithLogitsLoss()
t0 = time.time()
derniere_sauvegarde = t0
pertes = []

for epoque in range(EPOQUES):
    if time.time() - t0 > TIME_MIN * 60:
        break
    print(f"--- passage {epoque + 1}/{EPOQUES} ---")
    flux = pown.FluxPositions(pown.lignes_lichess, n_max=N_POS, profondeur_min=PROFONDEUR_MIN)
    dl = DataLoader(flux, batch_size=TAILLE_LOT, collate_fn=pown.rassembler)
    for i, (idx_n, dec_n, idx_e, dec_e, cibles) in enumerate(dl):
        idx_n, dec_n = idx_n.to(DEVICE), dec_n.to(DEVICE)
        idx_e, dec_e = idx_e.to(DEVICE), dec_e.to(DEVICE)
        cibles = cibles.to(DEVICE)

        opt.zero_grad()
        sortie = modele(idx_n, dec_n, idx_e, dec_e)
        perte = perte_fn(sortie, cibles)
        perte.backward()
        opt.step()

        n_vues += cibles.shape[0]
        pertes.append(perte.item())

        if i % 200 == 0:
            vitesse = n_vues / (time.time() - t0 + 1e-9)
            print(f"{n_vues:>12,} positions | perte {sum(pertes[-200:]) / len(pertes[-200:]):.4f} "
                  f"| {vitesse:,.0f} pos/s | {(time.time() - t0) / 60:.0f} min")

        if time.time() - derniere_sauvegarde > SAUVEGARDE_SEC:
            pown.sauvegarder(CHEMIN, modele, opt, n_vues, secondes_avant + time.time() - t0)
            derniere_sauvegarde = time.time()

        if time.time() - t0 > TIME_MIN * 60:
            print("budget temps atteint, on s'arrête proprement")
            break

pown.sauvegarder(CHEMIN, modele, opt, n_vues, secondes_avant + time.time() - t0)
print(f"terminé : {n_vues:,} positions vues au total, modèle sauvegardé dans {CHEMIN}")
'''),
    ("md", r'''
## Petit test de cohérence
'''),
    ("code", r'''
import chess

modele.eval()
print("position de départ :", round(modele.eval_cp(chess.STARTING_FEN), 1), "cp (attendu : proche de 0)")

# net avantage matériel pour les blancs (une Dame de plus)
print("blancs avec une Dame de plus :",
      round(modele.eval_cp("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"), 1), "cp")
b_sans_dame_noire = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
print("(sans la dame noire, ce devrait être largement positif)")
modele.train()
'''),
    ("md", r'''
### Et après ?
- **Pour continuer demain** : remets `DOSSIER_DEPART = f"{SORTIE}/pown.pt"` (le fichier est
  dans ton Drive, il survit à la session) et relance.
- **Pour le comparer aux autres modèles** : il faudra un petit pont ONNX ou un chargeur
  direct dans `commun_echecs.py`, plus une recherche alpha-bêta qui l'utilise — prochaine étape.
- Cette v1 mesure juste que **l'entraînement tourne et que la perte baisse**. Le vrai jugement
  viendra des benchmarks (comme pour PAWN, SPAWN, PWN) une fois qu'il pourra jouer.
'''),
]

if __name__ == "__main__":
    ecrire("pown_train.ipynb", NB)
