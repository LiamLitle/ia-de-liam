# Solution (S) — moteur d'échecs C++ avec le réseau P.O.W.N.

**S** est le moteur complet autour de P.O.W.N. : règles, recherche, gestion du temps et protocole UCI.

| Pièce | Fichier | Remarque |
|---|---|---|
| Règles (bitboards, coups légaux, roque, en passant, promotions) | `src/position.*`, `src/movegen.*`, `src/bitboard.*` | validé par *perft* (6 positions de référence) et contre `python-chess` (1 562 positions aléatoires, 0 différence) |
| Réseau P.O.W.N. (NNUE HalfKAv2_hm) | `src/nnue.*` | mise à jour incrémentale ; même valeur que PyTorch/onnxruntime (écart < 1 cp) |
| Recherche | `src/search.*`, `src/tt.h` | PVS, approfondissement itératif, fenêtres d'aspiration, table de transposition, coup nul, réductions (LMR), futilité, SEE, quiescence, extension d'échec, tueurs + historique |
| Temps, UCI | `src/main.cpp` | `go movetime/wtime/btime/depth/nodes/infinite`, options `Hash`, `EvalFile`, `Move Overhead`, `Material` |

## Compiler
```
g++ -O3 -march=native -std=c++17 -pthread -o solution src/*.cpp      # (sans nnue_test.cpp / main_test.cpp)
```
Sous Windows : MSYS2/MinGW ou WSL (même commande). `-march=native` accélère beaucoup le calcul du réseau.

## Poids
Le moteur lit `pown.bin`, créé depuis `pown.onnx` (paquet `onnx`) ou `pown.pt` (torch) :
```
python tools/exporter_poids.py pown.onnx pown.bin
```

## Utiliser
```
./solution pown.bin                 # puis commandes UCI (uci, isready, position, go movetime 1000, ...)
./solution pown.bin perft 5         # test des règles : 4865609
./solution pown.bin bench           # vitesse de recherche
```
Dans les benchmarks Python : joueur **`POWN@S1`** (S, 1 s par coup) via `SOLUTION_BIN` et `SOLUTION_POIDS` (voir `benchmarks/10_solution_cpp_vs_stockfish.ipynb`).
Compatible avec tout programme UCI (Arena, Cute Chess, Lichess-bot…).

## Pas encore fait (idées, à mesurer avant de les ajouter)
- poids quantifiés en entiers (int16/int8) + instructions SIMD écrites à la main ;
- plusieurs processeurs (Lazy SMP) ; livre d'ouvertures ; tables de finales (Syzygy) ;
- réseau qui propose aussi les coups à essayer en premier ; cartes tactiques « Soluce » en entrée.
