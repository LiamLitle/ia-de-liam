// Recherche alpha-bêta (PVS) avec approfondissement itératif, gestion du temps, quiescence et élagages.
#pragma once
#include <atomic>
#include <chrono>
#include "movegen.h"
#include "nnue.h"
#include "tt.h"

constexpr int MAX_PLY = 128;
constexpr int INF_SCORE = 32001;
constexpr int MATE_SCORE = 32000;
constexpr int MATE_BOUND = MATE_SCORE - MAX_PLY;

struct Limits {
    int depth = 0;
    int64_t nodes = 0;
    int movetime = 0, wtime = 0, btime = 0, winc = 0, binc = 0, movestogo = 0;
    bool infinite = false;
};

struct SearchResult {
    Move best = MOVE_NONE;
    int score = 0, depth = 0;
    int64_t nodes = 0;
};

class Search {
public:
    Search(NNUE& n, TT& t) : net(n), tt(t) { init_tables(); }
    bool use_material = false;       // évaluation « matériel seul » (tests de la recherche)
    int move_overhead = 30;          // millisecondes réservées par coup (communication, latence)
    bool quiet = false;              // pas d'affichage « info »
    SearchResult go(const Position& root, const Limits& lim, std::atomic<bool>& stop_flag);
private:
    NNUE& net;
    TT& tt;
    Position pos;
    Accumulator acc[MAX_PLY + 8];
    Move killers[MAX_PLY + 4][2];
    int history[2][64][64];
    Move pv[MAX_PLY + 2][MAX_PLY + 2];
    int pv_len[MAX_PLY + 2];
    int lmr[64][64];
    int64_t nodes = 0, node_limit = 0;
    std::chrono::steady_clock::time_point t0;
    int64_t soft_ms = 0, hard_ms = 0;
    std::atomic<bool>* stop_ptr = nullptr;
    bool stopped = false;
    bool ignore_time = false;        // la profondeur 1 est toujours terminée (un coup jouable à tout prix)

    void init_tables();
    int64_t elapsed_ms() const;
    void check_time();
    int eval(int ply);
    bool make(Move m, int ply);
    int negamax(int depth, int alpha, int beta, int ply, bool do_null);
    int qsearch(int alpha, int beta, int ply, int qply = 0);
    void set_time(const Limits& lim);
    void print_info(int depth, int score, const SearchResult& r);
};

std::string score_to_uci(int score);
