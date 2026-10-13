// Réseau P.O.W.N. (NNUE HalfKAv2_hm) en C++ : même formule que Pown/commun_pown.py.
//   22528 -> 256 (par point de vue, poids partagés) -> [2x256] -> 32 -> 32 -> 1 ; activations « clipped » dans [0,1]
//   sortie = logit ; centipions = 400 x logit (du point de vue du camp au trait)
#pragma once
#include <string>
#include <vector>
#include "position.h"

constexpr int NNUE_F = 22528, NNUE_H = 256, NNUE_H1 = 32, NNUE_H2 = 32;

struct Accumulator {
    alignas(64) float v[2][NNUE_H];   // un par point de vue (blanc, noir)
};

class NNUE {
public:
    bool loaded = false;
    bool load(const std::string& path);                          // fichier pown.bin (voir tools/exporter_poids.py)
    void load_random(unsigned seed);                             // poids aléatoires (tests)
    static int feature(int persp, int ksq, int piece, int sq);   // index de la caractéristique, -1 pour son propre roi
    void refresh(const Position& pos, Accumulator& a, int persp) const;
    void refresh_all(const Position& pos, Accumulator& a) const { refresh(pos, a, WHITE); refresh(pos, a, BLACK); }
    // met à jour l'accumulateur `to` à partir de `from` après le coup dont les changements sont dans st
    void update(const Position& pos, const Accumulator& from, Accumulator& to, const StateInfo& st) const;
    int evaluate(const Accumulator& a, int stm) const;           // centipions, point de vue du camp au trait
private:
    std::vector<float> ft;                                       // NNUE_F x NNUE_H
    alignas(64) float fb[NNUE_H], w1t[2 * NNUE_H * NNUE_H1], b1[NNUE_H1], w2t[NNUE_H1 * NNUE_H2], b2[NNUE_H2], w3[NNUE_H2], b3 = 0;
};
