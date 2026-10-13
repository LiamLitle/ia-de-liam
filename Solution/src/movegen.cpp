#include "movegen.h"

static void add_pawn_move(MoveList& l, int from, int to, bool promo_rank, GenType type) {
    if (promo_rank) {
        l.add(make_move(from, to, QUEEN));
        if (type == GEN_ALL) {
            l.add(make_move(from, to, ROOK));
            l.add(make_move(from, to, BISHOP));
            l.add(make_move(from, to, KNIGHT));
        }
    } else {
        l.add(make_move(from, to));
    }
}

void generate(const Position& pos, MoveList& list, GenType type) {
    const int us = pos.stm, them = us ^ 1;
    const Bitboard occ = pos.pieces(), mine = pos.byColor[us], theirs = pos.byColor[them];
    const Bitboard target = type == GEN_ALL ? ~mine : theirs;

    // pions
    Bitboard pawns = pos.pieces(us, PAWN);
    const int up = us == WHITE ? 8 : -8, start_rank = us == WHITE ? 1 : 6, last_rank = us == WHITE ? 6 : 1;
    while (pawns) {
        int from = pop_lsb(pawns);
        bool promo = rank_of(from) == last_rank;
        int one = from + up;
        if (!(occ & bb(one))) {
            if (type == GEN_ALL || promo) add_pawn_move(list, from, one, promo, type);
            if (type == GEN_ALL && rank_of(from) == start_rank && !(occ & bb(one + up)))
                list.add(make_move(from, one + up));
        }
        Bitboard caps = PAWN_ATT[us][from] & theirs;
        while (caps) add_pawn_move(list, from, pop_lsb(caps), promo, type);
        if (pos.ep >= 0 && (PAWN_ATT[us][from] & bb(pos.ep))) list.add(make_move(from, pos.ep));
    }
    // cavaliers, fous, tours, dames, roi
    Bitboard b = pos.pieces(us, KNIGHT);
    while (b) {
        int from = pop_lsb(b);
        Bitboard t = KNIGHT_ATT[from] & target;
        while (t) list.add(make_move(from, pop_lsb(t)));
    }
    b = pos.pieces(us, BISHOP);
    while (b) {
        int from = pop_lsb(b);
        Bitboard t = bishop_attacks(from, occ) & target;
        while (t) list.add(make_move(from, pop_lsb(t)));
    }
    b = pos.pieces(us, ROOK);
    while (b) {
        int from = pop_lsb(b);
        Bitboard t = rook_attacks(from, occ) & target;
        while (t) list.add(make_move(from, pop_lsb(t)));
    }
    b = pos.pieces(us, QUEEN);
    while (b) {
        int from = pop_lsb(b);
        Bitboard t = queen_attacks(from, occ) & target;
        while (t) list.add(make_move(from, pop_lsb(t)));
    }
    int ksq = pos.king_sq(us);
    Bitboard t = KING_ATT[ksq] & target;
    while (t) list.add(make_move(ksq, pop_lsb(t)));

    // roques (cases libres, roi pas en échec, case traversée non attaquée ; l'arrivée est vérifiée par do_move)
    if (type == GEN_ALL) {
        if (us == WHITE && ksq == 4) {
            if ((pos.castling & CASTLE_WK) && !(occ & (bb(5) | bb(6))) && pos.board[7] == make_piece(WHITE, ROOK)
                && !pos.attacked(4, BLACK) && !pos.attacked(5, BLACK) && !pos.attacked(6, BLACK))
                list.add(make_move(4, 6));
            if ((pos.castling & CASTLE_WQ) && !(occ & (bb(1) | bb(2) | bb(3))) && pos.board[0] == make_piece(WHITE, ROOK)
                && !pos.attacked(4, BLACK) && !pos.attacked(3, BLACK) && !pos.attacked(2, BLACK))
                list.add(make_move(4, 2));
        } else if (us == BLACK && ksq == 60) {
            if ((pos.castling & CASTLE_BK) && !(occ & (bb(61) | bb(62))) && pos.board[63] == make_piece(BLACK, ROOK)
                && !pos.attacked(60, WHITE) && !pos.attacked(61, WHITE) && !pos.attacked(62, WHITE))
                list.add(make_move(60, 62));
            if ((pos.castling & CASTLE_BQ) && !(occ & (bb(57) | bb(58) | bb(59))) && pos.board[56] == make_piece(BLACK, ROOK)
                && !pos.attacked(60, WHITE) && !pos.attacked(59, WHITE) && !pos.attacked(58, WHITE))
                list.add(make_move(60, 58));
        }
    }
}

Move parse_uci_move(Position& pos, const std::string& s) {
    MoveList l;
    generate(pos, l, GEN_ALL);
    for (int i = 0; i < l.n; i++)
        if (move_to_uci(l.m[i]) == s) {
            if (pos.do_move(l.m[i])) { pos.undo_move(); return l.m[i]; }
            return MOVE_NONE;
        }
    return MOVE_NONE;
}

uint64_t perft(Position& pos, int depth) {
    MoveList l;
    generate(pos, l, GEN_ALL);
    uint64_t n = 0;
    for (int i = 0; i < l.n; i++) {
        if (!pos.do_move(l.m[i])) continue;
        n += depth <= 1 ? 1 : perft(pos, depth - 1);
        pos.undo_move();
    }
    return n;
}
