# P.O.W.N. — Pawn Optimized With NNUE

Un nouveau réseau d'évaluation, dans l'esprit du NNUE de Stockfish : encodage **HalfKAv2_hm**
(roi + case + pièce, avec l'astuce du miroir gauche-droite), pensé pour être **très rapide**
une fois porté en C++ (mise à jour incrémentale à chaque coup, au lieu de tout recalculer
comme PAWN/SPAWN/PWN).

| | |
|---|---|
| Encodage | HalfKAv2_hm — 22 528 combinaisons (roi × case × pièce) |
| Réseau | 22 528 → 256 (×2 perspectives) → 32 → 32 → 1 |
| Paramètres | **5 784 673** |
| Données | dump officiel Lichess (FEN + évaluations Stockfish), en streaming |
| Objectif | ≈ 100 millions de positions, 2-3 passages |

## Lancer l'entraînement (Google Colab)

1. Télécharge `pown_train.ipynb` et importe-le sur [colab.research.google.com](https://colab.research.google.com).
2. **Exécution → Modifier le type d'exécution → GPU (T4)**.
3. (Optionnel) Dans les **Secrets** de Colab (icône 🔑), ajoute `HF_TOKEN` — pas indispensable
   pour ce notebook, gardé pour un futur besoin.
4. **Exécuter tout**. La première cellule te demande d'autoriser l'accès à ton Google Drive :
   c'est là que les points de contrôle sont sauvegardés (Colab ne garde rien tout seul,
   contrairement à l'Output de Kaggle).
5. Réglage `TIME_MIN` dans la cellule des réglages : l'entraînement s'arrête proprement à
   l'heure fixée, avec sauvegarde régulière.

## Reprendre un entraînement

Remets `DOSSIER_DEPART = f"{SORTIE}/pown.pt"` dans la cellule des réglages avant de relancer :
le modèle repart exactement du point de contrôle, sans revoir ce qu'il avait déjà appris.

## Modifier le notebook

Le code commun est dans `commun_pown.py`, recopié dans le notebook via `%%writefile` pour
qu'il soit autonome. Après une modification :

```bash
python Pown/construire_pown.py
```
