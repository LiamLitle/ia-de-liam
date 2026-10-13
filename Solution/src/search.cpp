#include "search.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

static const int MAT_VAL[7] = {0, 100, 320, 330, 500, 900, 0};

static inline int to_tt(int s, int ply) { return s >= MATE_BOUND ? s + ply : (s <= -MATE_BOUND ? s - ply : s); }
static inline int from_tt(int s, int ply) { return s >= MATE_BOUND ? s - ply : (s <= -MATE_BOUND ? s + ply : s); }

std::string score_to_uci(int s) {
    if (s >= MATE_BOUND) return "mate " + std::to_string((MATE_SCORE - s + 1) / 2);
    if (s <= -MATE_BOUND) return "mate " + std::to_string(-(MATE_SCORE + s + 1) / 2);
    return "cp " + std::to_string(s);
}

void Search::init_tables() {
    for (int d = 1; d < 64; d++)
        for (int m = 1; m < 64; m++) lmr[d][m] = (int)(0.75 + std::log((double)d) * std::log((double)m) / 2.25);
    for (auto& a : lmr[0]) a = 0;
}

int64_t Search::elapsed_ms() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
}

void Search::check_time() {
    if (stop_ptr->load(std::memory_order_relaxed)) { stopped = true; return; }
    if (node_limit && nodes >= node_limit) { stopped = true; return; }
    if (!ignore_time && hard_ms && elapsed_ms() >= hard_ms) stopped = true;
}

int Search::eval(int ply) {
    if (use_material || !net.loaded) {
        int v = 0;
        for (int t = PAWN; t <= QUEEN; t++)
            v += MAT_VAL[t] * (popcount(pos.pieces(pos.stm, t)) - popcount(pos.pieces(pos.stm ^ 1, t)));
        return v;
    }
    return net.evaluate(acc[ply], pos.stm);
}

bool Search::make(Move m, int ply) {
    if (!pos.do_move(m)) return false;
    if (!use_material && net.loaded) net.update(pos, acc[ply], acc[ply + 1], pos.last());
    return true;
}

int Search::qsearch(int alpha, int beta, int ply, int qply) {
    nodes++;
    if ((nodes & 1023) == 0) check_time();
    if (stopped) return 0;
    if (pos.is_repetition(ply) || pos.halfmove >= 100 || pos.insufficient_material()) return 0;
    if (ply >= MAX_PLY - 1) return eval(ply);
    bool in_check = pos.in_check();
    int best, stand = -INF_SCORE;
    const int orig_alpha = alpha;
    if (TTEntry* e = tt.probe(pos.key)) {
        int s = from_tt(e->score, ply);
        if (e->flag == TT_EXACT || (e->flag == TT_LOWER && s >= beta) || (e->flag == TT_UPPER && s <= alpha)) return s;
    }
    MoveList ml;
    if (in_check) {
        best = -MATE_SCORE + ply;
        generate(pos, ml, GEN_ALL);
    } else {
        stand = best = eval(ply);
        if (best >= beta) return best;
        alpha = std::max(alpha, best);
        if (qply >= 8) return best;                                  // la quiescence ne s'éternise pas
        generate(pos, ml, GEN_CAPTURES);
    }
    int score[256];
    for (int i = 0; i < ml.n; i++) {
        Move m = ml.m[i];
        int victim = pos.board[to_sq(m)] ? type_of(pos.board[to_sq(m)]) : PAWN;
        int attacker = type_of(pos.board[from_sq(m)]);
        score[i] = pos.is_capture(m) ? 10000 + victim * 16 - attacker : (promo_of(m) ? 9000 : 0);
    }
    for (int i = 0; i < ml.n; i++) {
        int bi = i;
        for (int j = i + 1; j < ml.n; j++) if (score[j] > score[bi]) bi = j;
        std::swap(ml.m[i], ml.m[bi]); std::swap(score[i], score[bi]);
        Move m = ml.m[i];
        if (!in_check) {
            int victim = pos.board[to_sq(m)] ? MAT_VAL[type_of(pos.board[to_sq(m)])] : 100;
            if (!promo_of(m) && stand + victim + 200 < alpha) continue;       // même en gagnant la pièce, loin de alpha
            if (!promo_of(m) && pos.see(m) < 0) continue;                      // mauvaise capture
        }
        if (!make(m, ply)) continue;
        int v = -qsearch(-beta, -alpha, ply + 1, in_check ? qply : qply + 1);
        pos.undo_move();
        if (stopped) return 0;
        if (v > best) {
            best = v;
            if (v > alpha) { alpha = v; if (alpha >= beta) break; }
        }
    }
    tt.store(pos.key, to_tt(best, ply), MOVE_NONE, 0, best >= beta ? TT_LOWER : (best > orig_alpha ? TT_EXACT : TT_UPPER));
    return best;
}

int Search::negamax(int depth, int alpha, int beta, int ply, bool do_null) {
    pv_len[ply] = ply;
    if (stopped) return 0;
    const bool pv_node = beta - alpha > 1, root = ply == 0;
    if (!root) {
        if (pos.is_repetition(ply) || pos.halfmove >= 100 || pos.insufficient_material()) return 0;
        if (ply >= MAX_PLY - 1) return eval(ply);
        alpha = std::max(alpha, -MATE_SCORE + ply);
        beta = std::min(beta, MATE_SCORE - ply - 1);
        if (alpha >= beta) return alpha;
    }
    const bool in_check = pos.in_check();
    if (in_check) depth++;
    if (depth <= 0) return qsearch(alpha, beta, ply);
    nodes++;
    if ((nodes & 1023) == 0) check_time();
    if (stopped) return 0;

    Move tt_move = MOVE_NONE;
    if (TTEntry* e = tt.probe(pos.key)) {
        tt_move = e->move;
        if (!pv_node && e->depth >= depth) {
            int s = from_tt(e->score, ply);
            if (e->flag == TT_EXACT || (e->flag == TT_LOWER && s >= beta) || (e->flag == TT_UPPER && s <= alpha)) return s;
        }
    }
    const int us = pos.stm;
    int static_eval = in_check ? -INF_SCORE : eval(ply);

    if (!pv_node && !in_check && std::abs(beta) < MATE_BOUND) {
        // élagage « inverse » : trop bien pour être inquiété à faible profondeur
        if (depth <= 6 && static_eval - 90 * depth >= beta) return static_eval;
        // coup nul : même en passant son tour on reste au-dessus de beta
        if (do_null && depth >= 3 && static_eval >= beta && pos.has_non_pawn_material(us)) {
            int R = 3 + depth / 4 + std::min((static_eval - beta) / 200, 2);
            pos.do_null();
            acc[ply + 1] = acc[ply];
            int v = -negamax(depth - 1 - R, -beta, -beta + 1, ply + 1, false);
            pos.undo_null();
            if (stopped) return 0;
            if (v >= beta) return v >= MATE_BOUND ? beta : v;
        }
    }

    MoveList ml;
    generate(pos, ml, GEN_ALL);
    int score[256];
    for (int i = 0; i < ml.n; i++) {
        Move m = ml.m[i];
        if (m == tt_move) { score[i] = 1 << 30; continue; }
        if (pos.is_capture(m)) {
            int victim = pos.board[to_sq(m)] ? type_of(pos.board[to_sq(m)]) : PAWN;
            int attacker = type_of(pos.board[from_sq(m)]);
            int mvv = victim * 16 - attacker;
            score[i] = (attacker <= victim || pos.see(m) >= 0) ? 500000 + mvv : -100000 + mvv;
        } else if (promo_of(m)) {
            score[i] = promo_of(m) == QUEEN ? 600000 : -200000;
        } else if (m == killers[ply][0]) score[i] = 400000;
        else if (m == killers[ply][1]) score[i] = 390000;
        else score[i] = history[us][from_sq(m)][to_sq(m)];
    }

    int best = -INF_SCORE, move_count = 0, nq = 0;
    Move best_move = MOVE_NONE, quiets[64];
    const int orig_alpha = alpha;
    for (int i = 0; i < ml.n; i++) {
        int bi = i;
        for (int j = i + 1; j < ml.n; j++) if (score[j] > score[bi]) bi = j;
        std::swap(ml.m[i], ml.m[bi]); std::swap(score[i], score[bi]);
        Move m = ml.m[i];
        const bool capture = pos.is_capture(m), is_quiet = !capture && !promo_of(m);
        if (!pv_node && !in_check && best > -MATE_BOUND && move_count > 0) {
            if (is_quiet && depth <= 4 && move_count >= 6 + depth * depth) continue;           // trop de coups calmes déjà essayés
            if (capture && depth <= 3 && pos.see(m) < -90 * depth) continue;                    // perd trop de matériel
        }
        if (!make(m, ply)) continue;
        move_count++;
        const bool gives_check = pos.in_check();
        if (!pv_node && !in_check && is_quiet && !gives_check && best > -MATE_BOUND && depth <= 3
            && move_count > 1 && static_eval + 100 + 120 * depth <= alpha) {                 // futilité : sans espoir de remonter
            pos.undo_move();
            continue;
        }
        int new_depth = depth - 1, v;
        int r = 0;
        if (depth >= 3 && move_count > 3 && is_quiet && !in_check) {
            r = lmr[std::min(depth, 63)][std::min(move_count, 63)];
            if (!pv_node) r++;
            if (gives_check) r--;
            if (m == killers[ply][0] || m == killers[ply][1]) r--;
            r = std::max(0, std::min(r, new_depth - 1));
        }
        if (move_count == 1) {
            v = -negamax(new_depth, -beta, -alpha, ply + 1, true);
        } else {
            v = -negamax(new_depth - r, -alpha - 1, -alpha, ply + 1, true);
            if (v > alpha && r > 0) v = -negamax(new_depth, -alpha - 1, -alpha, ply + 1, true);
            if (v > alpha && v < beta) v = -negamax(new_depth, -beta, -alpha, ply + 1, true);
        }
        pos.undo_move();
        if (stopped) return 0;
        if (v > best) {
            best = v;
            best_move = m;
            if (v > alpha) {
                alpha = v;
                pv[ply][ply] = m;
                for (int k = ply + 1; k < pv_len[ply + 1]; k++) pv[ply][k] = pv[ply + 1][k];
                pv_len[ply] = pv_len[ply + 1];
                if (alpha >= beta) {
                    if (is_quiet) {
                        if (killers[ply][0] != m) { killers[ply][1] = killers[ply][0]; killers[ply][0] = m; }
                        int bonus = std::min(depth * depth, 400);
                        auto upd = [&](Move q, int b) {
                            int& h = history[us][from_sq(q)][to_sq(q)];
                            h += b - h * std::abs(b) / 16384;
                        };
                        upd(m, bonus);
                        for (int k = 0; k < nq; k++) upd(quiets[k], -bonus);
                    }
                    break;
                }
            }
        }
        if (is_quiet && nq < 64) quiets[nq++] = m;
    }
    if (move_count == 0) return in_check ? -MATE_SCORE + ply : 0;
    uint8_t flag = best >= beta ? TT_LOWER : (best > orig_alpha ? TT_EXACT : TT_UPPER);
    tt.store(pos.key, to_tt(best, ply), best_move, depth, flag);
    return best;
}

void Search::set_time(const Limits& lim) {
    soft_ms = hard_ms = 0;
    int ov = move_overhead;
    if (lim.movetime) {
        soft_ms = hard_ms = std::max(5, lim.movetime - ov);
    } else if (lim.wtime || lim.btime) {
        int64_t t = pos.stm == WHITE ? lim.wtime : lim.btime, inc = pos.stm == WHITE ? lim.winc : lim.binc;
        int mtg = lim.movestogo ? std::min(lim.movestogo, 40) : 30;
        int64_t base = t / mtg + inc * 3 / 4;
        soft_ms = std::max<int64_t>(5, std::min<int64_t>(base, t * 8 / 10) - ov);
        hard_ms = std::max<int64_t>(soft_ms, std::min<int64_t>(base * 3, t * 4 / 10) - ov);
    }
}

void Search::print_info(int depth, int score, const SearchResult& r) {
    if (quiet) return;
    int64_t ms = elapsed_ms();
    std::cout << "info depth " << depth << " score " << score_to_uci(score) << " nodes " << nodes
              << " nps " << (ms ? nodes * 1000 / ms : nodes) << " time " << ms << " pv";
    for (int i = 0; i < pv_len[0]; i++) std::cout << " " << move_to_uci(pv[0][i]);
    std::cout << std::endl;
    (void)r;
}

SearchResult Search::go(const Position& root, const Limits& lim, std::atomic<bool>& stop_flag) {
    pos = root;
    stop_ptr = &stop_flag;
    stopped = false;
    nodes = 0;
    node_limit = lim.nodes;
    t0 = std::chrono::steady_clock::now();
    set_time(lim);
    std::memset(killers, 0, sizeof(killers));
    std::memset(history, 0, sizeof(history));
    if (!use_material && net.loaded) net.refresh_all(pos, acc[0]);

    SearchResult res;
    MoveList ml;
    generate(pos, ml, GEN_ALL);
    for (int i = 0; i < ml.n; i++)
        if (pos.do_move(ml.m[i])) { pos.undo_move(); if (res.best == MOVE_NONE) res.best = ml.m[i]; }
    if (res.best == MOVE_NONE) return res;           // aucun coup légal

    const int max_depth = lim.depth ? std::min(lim.depth, MAX_PLY - 2) : MAX_PLY - 2;
    int score = 0;
    for (int d = 1; d <= max_depth; d++) {
        ignore_time = (d == 1);
        int alpha = -INF_SCORE, beta = INF_SCORE, delta = 25;
        if (d >= 5) { alpha = std::max(score - delta, -INF_SCORE); beta = std::min(score + delta, INF_SCORE); }
        int v;
        while (true) {
            v = negamax(d, alpha, beta, 0, false);
            if (stopped) break;
            if (v <= alpha && alpha > -INF_SCORE) { alpha = std::max(alpha - delta * 2, -INF_SCORE); delta *= 2; }
            else if (v >= beta && beta < INF_SCORE) { beta = std::min(beta + delta * 2, INF_SCORE); delta *= 2; }
            else break;
        }
        if (stopped) break;
        score = v;
        if (pv_len[0] > 0 && pv[0][0] != MOVE_NONE) res.best = pv[0][0];
        res.score = score; res.depth = d; res.nodes = nodes;
        print_info(d, score, res);
        if (lim.infinite) continue;
        if (soft_ms && elapsed_ms() * 2 > soft_ms) break;          // la profondeur suivante dépasserait le temps prévu
        if (std::abs(score) >= MATE_BOUND && d >= 6 && !lim.depth) break;
    }
    res.nodes = nodes;
    return res;
}
