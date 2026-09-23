# PAWN — modèle « big »

**Acronyme :** *Precisely Adjusted, Weighted Network*
**Rôle :** version agrandie du modèle d'origine, même architecture de base (pas de Soluce), juste plus de blocs/filtres et plus de données. Sélectionnable dans l'app sous « P.A.W.N. big ».

## Architecture

- Évaluateur de position, CNN pur (pas de Soluce)
- 12 blocs résiduels × 128 filtres (contre 8×96 pour l'ancien pawn)
- Entrée : 17 plans (12 plans de pièces + 4 droits de roque + 1 prise en passant), point de vue du joueur au trait
- Sortie : un logit, converti en centipions via `cp = 400 × logit`

## Nombre de paramètres

**3 638 178 paramètres** (compté directement sur les poids du checkpoint `pawn_big.pt`) — environ 2,6× plus que l'ancien pawn

## Données et entraînement

- Même source : dataset Lichess/Stockfish (`lichess/chess-evaluations`)
- **60 000 000** positions chargées (`N_POS`), soit le double de l'ancien pawn
- Évaluations Stockfish peu profondes ignorées (`MIN_DEPTH = 20`)
- Entraînement limité en temps : **150 minutes** sur GPU Kaggle T4 (`TIME_MIN`)
- Batch size 2048, optimiseur AdamW (lr = 2e-3, weight decay = 1e-4)
- Perte : BCE-with-logits sur la cible `sigmoid(cp / 400)`
- 100 000 positions de validation
- Notebook : `pawn_train_big.ipynb`

## Recherche (au moment de jouer)

- Recherche arborescente classique (full-width, pas d'élagage), profondeur réglable selon le niveau (1 à 5)

## Performance (benchmark 300 puzzles)

- Profondeur 1 : **70,7 %** de réussite
- Profondeur 2 : **75,0 %** de réussite

C'est aussi le modèle de base sur lequel SPAWN a été branché (fine-tuning), et le point de comparaison direct pour mesurer l'apport réel du module Soluce.
