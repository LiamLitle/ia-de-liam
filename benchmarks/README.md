# Benchmarks Kaggle — P.A.W.N. et MIND

4 notebooks prêts à lancer sur Kaggle, chacun autonome (il télécharge tout seul les poids depuis ce dépôt GitHub).

| Notebook | Ce qu'il mesure | Durée estimée (GPU T4) |
|---|---|---|
| `01_echecs_precision_evaluation.ipynb` | Précision des 4 réseaux **sans recherche** vs Stockfish (profondeur ≥ 20) sur des positions jamais vues à l'entraînement : BCE, erreur de win-prob, Spearman, accord avec le meilleur coup de Stockfish, symétrie, vitesse | ~20 min |
| `02_echecs_puzzles_lichess.ipynb` | **Puzzles Lichess** stratifiés de 400 à 2800 : taux de résolution, **Elo puzzle**, résultats par thème, tests de McNemar | ~1–2 h |
| `03_echecs_tournoi_elo.ipynb` | **Tournoi** entre les modèles + matchs contre **Stockfish 17** bridé → **classement Elo** avec intervalles de confiance, PGN de toutes les parties | ~2–4 h |
| `04_mind_embeddings_mteb_fr.ipynb` | **MIND** vs 5 modèles de référence : STSb fr / en / fr↔en, vitesse, et le benchmark officiel **MTEB français** (25 tâches) | ~2–4 h (`LEGER=True` : ~30 min) |

Les modèles d'échecs testés :

| Nom dans les notebooks | Fichier | Recherche |
|---|---|---|
| `PAWN` | `Pawn/pawn.onnx` | arbre complet |
| `PAWN_BIG` | `PawnBig-V1/pawn_big.onnx` | arbre complet |
| `CONTROLE` | `PawnBig-controlleur/pawn_big_controle.onnx` | arbre complet |
| `SPAWN` | `PWN-soluce/pawn_soluce.onnx` | arbre complet |
| `PWN` | `PWN-soluce/pawn_soluce.onnx` (mêmes poids que SPAWN) | alpha-bêta + quiescence |

Un *joueur* s'écrit `NOM@arbreN` (arbre complet, profondeur N) ou `PWN@abN` (alpha-bêta, profondeur N).
La recherche est **exactement celle de `pawn_app.py`**, sans le bruit ni les coups au hasard des niveaux faciles.

## Lancer sur Kaggle

1. Sur kaggle.com : **Create → New Notebook**, puis **File → Import Notebook** et choisir le `.ipynb` (téléchargé depuis GitHub).
2. Dans le panneau de droite (**Settings**) :
   - **Internet : On** (il faut un compte Kaggle vérifié par téléphone) ;
   - **Accelerator : GPU T4 x1**.
3. Pour un essai rapide : **Run All**. Pour le vrai benchmark : **Save Version → Save & Run All (Commit)** — ça tourne
   en arrière-plan jusqu'à 12 h, même navigateur fermé, et les résultats (CSV, PNG, PGN) sont dans l'onglet **Output**.

Les notebooks 2, 3 et 4 **sauvegardent au fur et à mesure** : si la session est coupée, relancer reprend où ça s'était arrêté
(en mode interactif ; une nouvelle version « Commit » repart de zéro, sauf si tu ajoutes l'output précédent comme input).

### Poids sans Internet (facultatif)

Si tu préfères ne pas dépendre de GitHub : crée un dataset Kaggle avec les fichiers `.onnx` et le dossier `mind-v2-final/`,
ajoute-le au notebook (**Add Input**). Les notebooks cherchent d'abord dans `/kaggle/input/` avant de télécharger.

## Régler la durée

Tout est dans la cellule « réglages » de chaque notebook :
- notebook 2 : `PAR_TRANCHE` (puzzles par tranche de 200 points), `JOUEURS` ;
- notebook 3 : `N_OUVERTURES` (parties par paire = 2 × `N_OUVERTURES`), `MODELES`, `BUDGET_H` ;
- notebook 4 : `LEGER`, `BASELINES`.

## Modifier les notebooks

Les `.ipynb` sont générés par `construire_notebooks.py` (le code échecs commun est dans `commun_echecs.py` et
recopié dans chaque notebook pour qu'il soit autonome). Après une modification :

```bash
python benchmarks/construire_notebooks.py
```
