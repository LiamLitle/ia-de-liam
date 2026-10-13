// Tables d'attaques précalculées et attaques des pièces glissantes (par rayons).
#pragma once
#include "types.h"

extern Bitboard KNIGHT_ATT[64], KING_ATT[64], PAWN_ATT[2][64];
extern Bitboard RAY[8][64];          // 8 directions : N NE E NW | S SW W SE (les 4 premières vont vers les grands indices)
extern Bitboard BETWEEN[64][64];     // cases strictement entre deux cases alignées (sinon 0)
extern Bitboard LINE[64][64];        // toute la ligne passant par deux cases alignées (sinon 0)

void init_bitboards();

inline Bitboard ray_attack(int dir, int sq, Bitboard occ) {
    Bitboard a = RAY[dir][sq];
    Bitboard blockers = a & occ;
    if (blockers) {
        int s = dir < 4 ? lsb(blockers) : msb(blockers);
        a ^= RAY[dir][s];
    }
    return a;
}
inline Bitboard rook_attacks(int sq, Bitboard occ) {
    return ray_attack(0, sq, occ) | ray_attack(2, sq, occ) | ray_attack(4, sq, occ) | ray_attack(6, sq, occ);
}
inline Bitboard bishop_attacks(int sq, Bitboard occ) {
    return ray_attack(1, sq, occ) | ray_attack(3, sq, occ) | ray_attack(5, sq, occ) | ray_attack(7, sq, occ);
}
inline Bitboard queen_attacks(int sq, Bitboard occ) { return rook_attacks(sq, occ) | bishop_attacks(sq, occ); }
