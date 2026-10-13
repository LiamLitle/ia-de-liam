// Table de transposition : mémorise le résultat de positions déjà recherchées.
#pragma once
#include <cstdint>
#include <vector>
#include "types.h"

enum TTFlag : uint8_t { TT_NONE = 0, TT_LOWER = 1, TT_UPPER = 2, TT_EXACT = 3 };

struct TTEntry {
    uint64_t key;
    int16_t score;
    Move move;
    int8_t depth;
    uint8_t flag;
};

class TT {
public:
    explicit TT(size_t mb = 64) { resize(mb); }
    void resize(size_t mb) {
        size_t n = 1;
        while (n * 2 * sizeof(TTEntry) <= mb * 1024 * 1024) n *= 2;
        table.assign(n, TTEntry{0, 0, 0, 0, 0});
        mask = n - 1;
    }
    void clear() { std::fill(table.begin(), table.end(), TTEntry{0, 0, 0, 0, 0}); }
    TTEntry* probe(uint64_t key) {
        TTEntry* e = &table[key & mask];
        return (e->key == key && e->flag != TT_NONE) ? e : nullptr;
    }
    void store(uint64_t key, int score, Move move, int depth, uint8_t flag) {
        TTEntry* e = &table[key & mask];
        // on garde l'entrée plus profonde de la même position ; sinon on remplace
        if (e->key == key && e->depth > depth + 2 && flag != TT_EXACT) {
            if (move != MOVE_NONE && e->move == MOVE_NONE) e->move = move;
            return;
        }
        if (move == MOVE_NONE && e->key == key) move = e->move;
        *e = TTEntry{key, (int16_t)score, move, (int8_t)depth, flag};
    }
private:
    std::vector<TTEntry> table;
    size_t mask = 0;
};
