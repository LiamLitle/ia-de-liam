# SPAWN — modèle avec module Soluce

**Acronyme :** *Strategic Positional Analysis, Weighted Network*
**Rôle :** PAWN big + le module Soluce (les « cartes tactiques »). Sélectionnable dans l'app sous « Soluce (arbre complet) ».

## Architecture

- Même corps que PAWN big : 12 blocs résiduels × 128 filtres
- Ajout du module **Soluce** : calcule, à l'intérieur même du réseau, 10 plans tactiques supplémentaires à partir des 17 plans de base — cases attaquées par soi/l'adversaire, nombre d'attaquants (plafonné à 3), pièces en prise, pièces clouées, pression sur la sécurité du roi (soi/adversaire)
- Entrée totale : **27 plans** (17 de base + 10 Soluce)
- Le module Soluce lui-même n'a aucun poids appris — c'est un calcul géométrique fixe (décalages, tracé de rayons, détection de clouages), vérifié à la main contre une implémentation de référence en python-chess (0 erreur sur 3000 positions testées)
- Sortie : un logit, converti en centipions via `cp = 400 × logit`

## Nombre de paramètres

**3 649 698 paramètres** — soit PAWN big (3 638 178) + **11 520 nouveaux poids** dans la première couche de convolution, pour lire les 10 plans Soluce ajoutés. Le module Soluce en lui-même n'ajoute aucun paramètre entraîné.

## Données et entraînement

- Fine-tuning à partir des poids de PAWN big : les 11 520 nouveaux poids sont initialisés à zéro, donc SPAWN démarre **identique** à PAWN big avant l'entraînement (vérifié : écart max 0,0005 cp)
- **30 000 000 nouvelles positions**, jamais vues par PAWN big (les 60 000 000 positions d'entraînement de PAWN big sont explicitement sautées — `SKIP = 60 000 000`)
- **15 000 pas** d'entraînement (`STEPS`), toujours exécutés jusqu'au bout — pas d'arrêt anticipé
- Le meilleur point de contrôle (au sens de la perte de validation, vérifiée tous les 2500 pas) est conservé et restauré à la fin, même s'il n'est pas au dernier pas
- Batch size 2048, optimiseur AdamW (lr = 5e-4, weight decay = 1e-4)
- Entraîné **en parallèle d'un modèle « contrôle »** (PAWN big continué sans Soluce, mêmes données, même ordre) pour isoler ce que le module Soluce apporte réellement par rapport à plus d'entraînement tout court — SPAWN a montré une amélioration de perte de validation environ 2× plus grande que le contrôle
- Notebook : `pawn_soluce_finetune.ipynb` (Kaggle)

## Recherche (au moment de jouer)

- Recherche arborescente classique (full-width, pas d'élagage), profondeur réglable selon le niveau (1 à 5)

## Performance (benchmark 300 puzzles)

- Profondeur 1 : **73,3 %** de réussite
- Profondeur 2 : **77,7 %** de réussite

Meilleur que PAWN big aux deux profondeurs, avec un gain plus marqué sur les thèmes sacrifice, mat en 2, finale de pions et attaque à l'aile roi.
