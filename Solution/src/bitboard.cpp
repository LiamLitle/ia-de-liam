#include "bitboard.h"

Bitboard KNIGHT_ATT[64], KING_ATT[64], PAWN_ATT[2][64], RAY[8][64], BETWEEN[64][64], LINE[64][64];

std::string square_name(int s) { return std::string(1, char('a' + file_of(s))) + char('1' + rank_of(s)); }

std::string move_to_uci(Move m) {
    if (m == MOVE_NONE) return "0000";
    std::string s = square_name(from_sq(m)) + square_name(to_sq(m));
    static const char pc[] = " pnbrqk";
    if (promo_of(m)) s += pc[promo_of(m)];
    return s;
}

void init_bitboards() {
    static const int DR[8] = {1, 1, 0, 1, -1, -1, 0, -1};
    static const int DF[8] = {0, 1, 1, -1, 0, -1, -1, 1};
    for (int s = 0; s < 64; s++) {
        int r = rank_of(s), f = file_of(s);
        for (int d = 0; d < 8; d++) {
            Bitboard b = 0;
            for (int rr = r + DR[d], ff = f + DF[d]; rr >= 0 && rr < 8 && ff >= 0 && ff < 8; rr += DR[d], ff += DF[d])
                b |= bb(rr * 8 + ff);
            RAY[d][s] = b;
        }
        KNIGHT_ATT[s] = KING_ATT[s] = PAWN_ATT[0][s] = PAWN_ATT[1][s] = 0;
        static const int KN[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
        for (auto& k : KN) {
            int rr = r + k[0], ff = f + k[1];
            if (rr >= 0 && rr < 8 && ff >= 0 && ff < 8) KNIGHT_ATT[s] |= bb(rr * 8 + ff);
        }
        for (int d = 0; d < 8; d++) {
            int rr = r + DR[d], ff = f + DF[d];
            if (rr >= 0 && rr < 8 && ff >= 0 && ff < 8) KING_ATT[s] |= bb(rr * 8 + ff);
        }
        for (int df : {-1, 1}) {
            if (f + df >= 0 && f + df < 8) {
                if (r + 1 < 8) PAWN_ATT[WHITE][s] |= bb((r + 1) * 8 + f + df);
                if (r - 1 >= 0) PAWN_ATT[BLACK][s] |= bb((r - 1) * 8 + f + df);
            }
        }
    }
    for (int a = 0; a < 64; a++)
        for (int b = 0; b < 64; b++) {
            BETWEEN[a][b] = LINE[a][b] = 0;
            for (int d = 0; d < 8; d++)
                if (RAY[d][a] & bb(b)) {
                    BETWEEN[a][b] = RAY[d][a] & ~RAY[d][b] & ~bb(b);
                    LINE[a][b] = RAY[d][a] | RAY[(d + 4) % 8][a] | bb(a);
                }
        }
}
