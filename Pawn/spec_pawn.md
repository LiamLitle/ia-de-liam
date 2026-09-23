# pawn — modèle historique

**Acronyme :** *Poorly Automated Weak Noob*
**Rôle :** le tout premier modèle du projet, gardé tel quel comme référence/historique. Sélectionnable dans l'app sous « Ancien P.A.W.N. ».

## Architecture

- Évaluateur de position, CNN pur (pas de Soluce)
- 8 blocs résiduels × 96 filtres
- Entrée : 17 plans (12 plans de pièces + 4 droits de roque + 1 prise en passant), point de vue du joueur au trait
- Sortie : un logit, converti en centipions via `cp = 400 × logit`

## Nombre de paramètres

**1 414 906 paramètres** (compté directement sur les poids du checkpoint `pawn.pt`)

## Données et entraînement

- Source : dataset Lichess/Stockfish (`lichess/chess-evaluations`) — positions annotées par Stockfish
- **30 000 000** positions chargées (`N_POS`)
- Évaluations Stockfish peu profondes ignorées (`MIN_DEPTH = 20`)
- Entraînement limité en temps : **60 minutes** sur GPU Kaggle T4 (`TIME_MIN`)
- Batch size 2048, optimiseur AdamW (lr = 2e-3, weight decay = 1e-4)
- Perte : BCE-with-logits sur la cible `sigmoid(cp / 400)`
- 100 000 positions de validation
- Notebook : `pawn_train.ipynb`

## Recherche (au moment de jouer)

- Recherche arborescente classique (full-width, pas d'élagage), profondeur réglable selon le niveau (1 à 5)

## Performance (benchmark 300 puzzles)

- Profondeur 1 : **63,7 %** de réussite
- Profondeur 2 : **69,0 %** de réussite

C'est le modèle le plus faible de la famille — sert de point de comparaison pour mesurer les progrès des versions suivantes.
