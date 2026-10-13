#include "position.h"
#include <sstream>
#include <cstdlib>

static uint64_t Z_PIECE[16][64], Z_CASTLE[16], Z_EP[8], Z_SIDE;
static int CASTLE_MASK[64];
static const int SEE_VAL[7] = {0, 100, 320, 330, 500, 900, 20000};

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static uint64_t rnd() {
    rng_state ^= rng_state >> 12; rng_state ^= rng_state << 25; rng_state ^= rng_state >> 27;
    return rng_state * 2685821657736338717ULL;
}

void Position::init() {
    init_bitboards();
    for (auto& a : Z_PIECE) for (auto& k : a) k = rnd();
    for (auto& k : Z_CASTLE) k = rnd();
    for (auto& k : Z_EP) k = rnd();
    Z_SIDE = rnd();
    for (int s = 0; s < 64; s++) CASTLE_MASK[s] = 15;
    CASTLE_MASK[4] = 15 & ~(CASTLE_WK | CASTLE_WQ);
    CASTLE_MASK[0] = 15 & ~CASTLE_WQ;
    CASTLE_MASK[7] = 15 & ~CASTLE_WK;
    CASTLE_MASK[60] = 15 & ~(CASTLE_BK | CASTLE_BQ);
    CASTLE_MASK[56] = 15 & ~CASTLE_BQ;
    CASTLE_MASK[63] = 15 & ~CASTLE_BK;
}

uint64_t Position::ep_key() const {
    if (ep < 0) return 0;
    if (PAWN_ATT[stm ^ 1][ep] & pieces(stm, PAWN)) return Z_EP[file_of(ep)];
    return 0;
}

void Position::put(int p, int s) {
    board[s] = p;
    byType[type_of(p)] |= bb(s);
    byColor[color_of(p)] |= bb(s);
    key ^= Z_PIECE[p][s];
}
void Position::remove(int s) {
    int p = board[s];
    board[s] = NO_PIECE;
    byType[type_of(p)] &= ~bb(s);
    byColor[color_of(p)] &= ~bb(s);
    key ^= Z_PIECE[p][s];
}
void Position::move_piece(int a, int b) {
    int p = board[a];
    remove(a);
    put(p, b);
}

bool Position::set_fen(const std::string& fen) {
    *this = Position();
    std::istringstream ss(fen);
    std::string placement, side, cast, epstr;
    int hm = 0, fm = 1;
    if (!(ss >> placement >> side)) return false;
    if (!(ss >> cast)) cast = "-";
    if (!(ss >> epstr)) epstr = "-";
    ss >> hm >> fm;
    int r = 7, f = 0;
    for (char ch : placement) {
        if (ch == '/') { r--; f = 0; }
        else if (ch >= '1' && ch <= '8') f += ch - '0';
        else {
            const char* names = "pnbrqk";
            const char* p = nullptr;
            for (const char* q = names; *q; q++) if (*q == (ch | 32)) p = q;
            if (!p || r < 0 || f > 7) return false;
            int color = (ch >= 'a') ? BLACK : WHITE;
            put(make_piece(color, int(p - names) + 1), r * 8 + f);
            f++;
        }
    }
    if (!(byType[KING] & byColor[WHITE]) || !(byType[KING] & byColor[BLACK])) return false;
    stm = side == "b" ? BLACK : WHITE;
    for (char ch : cast) {
        if (ch == 'K') castling |= CASTLE_WK;
        if (ch == 'Q') castling |= CASTLE_WQ;
        if (ch == 'k') castling |= CASTLE_BK;
        if (ch == 'q') castling |= CASTLE_BQ;
    }
    ep = (epstr.size() == 2) ? (epstr[1] - '1') * 8 + (epstr[0] - 'a') : -1;
    halfmove = hm;
    fullmove = fm;
    if (stm == BLACK) key ^= Z_SIDE;
    key ^= Z_CASTLE[castling] ^ ep_key();
    return true;
}

std::string Position::fen() const {
    std::string s;
    for (int r = 7; r >= 0; r--) {
        int empty = 0;
        for (int f = 0; f < 8; f++) {
            int p = board[r * 8 + f];
            if (!p) { empty++; continue; }
            if (empty) { s += char('0' + empty); empty = 0; }
            char c = " pnbrqk"[type_of(p)];
            s += color_of(p) == WHITE ? char(c - 32) : c;
        }
        if (empty) s += char('0' + empty);
        if (r) s += '/';
    }
    s += stm == WHITE ? " w " : " b ";
    std::string c;
    if (castling & CASTLE_WK) c += 'K';
    if (castling & CASTLE_WQ) c += 'Q';
    if (castling & CASTLE_BK) c += 'k';
    if (castling & CASTLE_BQ) c += 'q';
    s += (c.empty() ? "-" : c) + " " + (ep >= 0 ? square_name(ep) : "-") + " " + std::to_string(halfmove) + " " + std::to_string(fullmove);
    return s;
}

Bitboard Position::attackers_to(int sq, Bitboard occ) const {
    return (PAWN_ATT[BLACK][sq] & pieces(WHITE, PAWN)) | (PAWN_ATT[WHITE][sq] & pieces(BLACK, PAWN))
         | (KNIGHT_ATT[sq] & byType[KNIGHT]) | (KING_ATT[sq] & byType[KING])
         | (bishop_attacks(sq, occ) & (byType[BISHOP] | byType[QUEEN]))
         | (rook_attacks(sq, occ) & (byType[ROOK] | byType[QUEEN]));
}

bool Position::attacked(int sq, int by) const {
    Bitboard occ = pieces();
    return (PAWN_ATT[by ^ 1][sq] & pieces(by, PAWN)) || (KNIGHT_ATT[sq] & pieces(by, KNIGHT))
        || (KING_ATT[sq] & pieces(by, KING))
        || (bishop_attacks(sq, occ) & (pieces(by, BISHOP) | pieces(by, QUEEN)))
        || (rook_attacks(sq, occ) & (pieces(by, ROOK) | pieces(by, QUEEN)));
}

bool Position::is_capture(Move m) const {
    int from = from_sq(m), to = to_sq(m);
    if (board[to]) return true;
    return type_of(board[from]) == PAWN && to == ep && file_of(from) != file_of(to);
}

bool Position::do_move(Move m) {
    int us = stm, them = us ^ 1, from = from_sq(m), to = to_sq(m), promo = promo_of(m);
    int pc = board[from], pt = type_of(pc), cap = board[to];
    StateInfo st;
    st.key = key; st.castling = castling; st.ep = ep; st.halfmove = halfmove; st.captured = NO_PIECE; st.cap_sq = -1;
    st.move = m; st.dn = 0;
    key ^= ep_key() ^ Z_CASTLE[castling];
    int cap_sq = to;
    if (pt == PAWN && to == ep && cap == NO_PIECE && file_of(from) != file_of(to)) {
        cap_sq = to + (us == WHITE ? -8 : 8);
        cap = board[cap_sq];
    }
    if (cap) {
        st.captured = cap; st.cap_sq = cap_sq;
        remove(cap_sq);
        st.d[st.dn++] = {cap, cap_sq, -1};
    }
    if (promo) {
        remove(from);
        int np = make_piece(us, promo);
        put(np, to);
        st.d[st.dn++] = {pc, from, -1};
        st.d[st.dn++] = {np, -1, to};
    } else {
        move_piece(from, to);
        st.d[st.dn++] = {pc, from, to};
    }
    if (pt == KING && (to - from == 2 || from - to == 2)) {
        int rf = to > from ? from + 3 : from - 4, rt = to > from ? from + 1 : from - 1;
        int rook = board[rf];
        move_piece(rf, rt);
        st.d[st.dn++] = {rook, rf, rt};
    }
    castling &= CASTLE_MASK[from] & CASTLE_MASK[to];
    ep = (pt == PAWN && (to - from == 16 || from - to == 16)) ? (from + to) / 2 : -1;
    halfmove = (pt == PAWN || st.captured) ? 0 : halfmove + 1;
    if (us == BLACK) fullmove++;
    stm = them;
    key ^= Z_SIDE ^ Z_CASTLE[castling] ^ ep_key();
    hist.push_back(st);
    if (attacked(king_sq(us), them)) {
        undo_move();
        return false;
    }
    return true;
}

void Position::undo_move() {
    StateInfo st = hist.back();
    hist.pop_back();
    stm ^= 1;
    int us = stm, from = from_sq(st.move), to = to_sq(st.move), promo = promo_of(st.move);
    if (promo) {
        remove(to);
        put(make_piece(us, PAWN), from);
    } else {
        move_piece(to, from);
    }
    int pt = type_of(board[from]);
    if (pt == KING && (to - from == 2 || from - to == 2)) {
        int rf = to > from ? from + 3 : from - 4, rt = to > from ? from + 1 : from - 1;
        move_piece(rt, rf);
    }
    if (st.captured) put(st.captured, st.cap_sq);
    castling = st.castling; ep = st.ep; halfmove = st.halfmove; key = st.key;
    if (us == BLACK) fullmove--;
}

void Position::do_null() {
    StateInfo st;
    st.key = key; st.castling = castling; st.ep = ep; st.halfmove = halfmove; st.captured = NO_PIECE; st.cap_sq = -1;
    st.move = MOVE_NONE; st.dn = 0;
    key ^= ep_key();
    ep = -1;
    stm ^= 1;
    key ^= Z_SIDE;
    halfmove++;
    hist.push_back(st);
}

void Position::undo_null() {
    StateInfo st = hist.back();
    hist.pop_back();
    stm ^= 1;
    ep = st.ep; halfmove = st.halfmove; key = st.key;
}

bool Position::has_non_pawn_material(int c) const {
    return (byColor[c] & (byType[KNIGHT] | byType[BISHOP] | byType[ROOK] | byType[QUEEN])) != 0;
}

bool Position::insufficient_material() const {
    if (byType[PAWN] | byType[ROOK] | byType[QUEEN]) return false;
    return popcount(byType[KNIGHT] | byType[BISHOP]) <= 1;
}

bool Position::is_repetition(int ply_from_root) const {
    int n = (int)hist.size();
    int lim = std::min(halfmove, n);
    int count = 0;
    for (int i = 2; i <= lim; i += 2) {
        if (hist[n - i].key == key) {
            if (i <= ply_from_root) return true;
            if (++count >= 2) return true;
        }
    }
    return false;
}

int Position::see(Move m) const {
    int from = from_sq(m), to = to_sq(m);
    int gain[32], d = 0;
    int target = board[to] ? type_of(board[to]) : PAWN;   // en passant : un pion
    int attacker = type_of(board[from]);
    gain[0] = SEE_VAL[target];
    if (promo_of(m)) gain[0] += SEE_VAL[promo_of(m)] - SEE_VAL[PAWN];
    Bitboard occ = pieces() ^ bb(from);
    int side = stm ^ 1;
    int cur_val = promo_of(m) ? SEE_VAL[promo_of(m)] : SEE_VAL[attacker];
    while (true) {
        Bitboard att = attackers_to(to, occ) & occ & byColor[side];
        if (!att) break;
        int best = -1;
        for (int t = PAWN; t <= KING; t++) {
            Bitboard b = att & byType[t];
            if (b) { best = lsb(b); break; }
        }
        d++;
        gain[d] = cur_val - gain[d - 1];
        if (std::max(-gain[d - 1], gain[d]) < 0) break;
        cur_val = SEE_VAL[type_of(board[best])];
        occ ^= bb(best);
        side ^= 1;
        if (d >= 30) break;
    }
    for (; d > 0; d--) gain[d - 1] = -std::max(-gain[d - 1], gain[d]);
    return gain[0];
}
