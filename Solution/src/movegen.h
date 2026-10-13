// Génération des coups pseudo-légaux (la légalité — roi pas en échec — est vérifiée par Position::do_move).
#pragma once
#include "position.h"

enum GenType { GEN_ALL, GEN_CAPTURES };   // CAPTURES : prises + promotions en dame (pour la quiescence)

struct MoveList {
    Move m[256];
    int n = 0;
    void add(Move mv) { m[n++] = mv; }
};

void generate(const Position& pos, MoveList& list, GenType type);
Move parse_uci_move(Position& pos, const std::string& s);   // MOVE_NONE si illégal
uint64_t perft(Position& pos, int depth);
