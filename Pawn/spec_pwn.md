# PWN — Soluce + alpha-bêta/quiescence

**Acronyme :** *Pruned With Neurons* (avec un clin d'œil au sens gaming/hacker de « pwn » : écraser l'adversaire)
**Rôle :** la version la plus forte actuelle. Sélectionnable dans l'app sous « Soluce + alpha-bêta/quiescence ».

## Architecture — réseau

**Exactement le même réseau que SPAWN, avec les mêmes poids** (`pawn_soluce.onnx`) : 12 blocs × 128 filtres, module Soluce, 27 plans d'entrée, 3 649 698 paramètres. PWN n'est pas un modèle réentraîné séparément — c'est SPAWN combiné à un algorithme de recherche différent au moment de jouer.

## Nombre de paramètres

**3 649 698 paramètres** (identiques à SPAWN — voir `spec_spawn.md` pour le détail de l'entraînement)

## Données et entraînement

Identiques à SPAWN (voir `spec_spawn.md`). Aucun entraînement supplémentaire n'est nécessaire pour PWN : c'est un changement de moteur de recherche, pas de réseau.

## Recherche (au moment de jouer) — ce qui change vraiment

- **Negamax avec élagage alpha-bêta**, au lieu d'une recherche arborescente complète (full-width)
- **Recherche de quiescence** : à chaque feuille, on prolonge la recherche sur les captures uniquement, pour éviter les erreurs d'horizon (rater un coup juste après la profondeur limite)
- **Ordre des coups** optimisé : MVV-LVA (la pièce la plus précieuse prise par la moins précieuse), coup de la table de transposition en premier, coups d'échec priorisés
- **Table de transposition** indexée par hash de Zobrist (positions déjà évaluées, réutilisées)
- **Approfondissement itératif** (iterative deepening) : réutilise le meilleur coup de la profondeur précédente pour mieux trier les coups à la profondeur suivante
- Correction mathématiquement vérifiée : à profondeur de quiescence nulle, PWN retrouve exactement les mêmes scores que l'ancienne recherche full-width (testé sur 8 positions aléatoires)
- Profondeur et profondeur de quiescence réglables selon le niveau (1 à 6, le niveau 6 étant très lent, ~40s/coup)

## Performance (benchmark 300 puzzles)

- Profondeur 2 : **85,3 %** de réussite (1,28 s/coup) — plus rapide *et* plus fort que SPAWN en arbre complet à la même profondeur (77,7 %, 3,11 s/coup)
- Profondeur 3 : **89,7 %** de réussite (6,37 s/coup) — là où l'ancienne recherche full-width en profondeur 3 était trop lente pour être jouable (40 à 100 s/coup)

C'est le plus grand saut de performance du projet jusqu'ici : le test de McNemar confirme que l'écart avec SPAWN en arbre complet est statistiquement significatif (44 puzzles gagnés contre 8 perdus).
