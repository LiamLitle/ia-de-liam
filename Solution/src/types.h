// Types de base du moteur Solution (S) : cases, pièces, coups, opérations sur bitboards.
#pragma once
#include <cstdint>
#include <string>

using Bitboard = uint64_t;
using Move = uint16_t;

enum Color { WHITE = 0, BLACK = 1 };
enum PieceType { NO_PIECE_TYPE = 0, PAWN = 1, KNIGHT = 2, BISHOP = 3, ROOK = 4, QUEEN = 5, KING = 6 };

// pièce = couleur * 8 + type  (blancs 1..6, noirs 9..14), 0 = case vide
constexpr int NO_PIECE = 0;
constexpr int make_piece(int c, int t) { return c * 8 + t; }
constexpr int color_of(int p) { return p >> 3; }
constexpr int type_of(int p) { return p & 7; }

// cases : a1 = 0, b1 = 1, ..., h8 = 63 (comme python-chess)
constexpr int file_of(int s) { return s & 7; }
constexpr int rank_of(int s) { return s >> 3; }
constexpr Bitboard bb(int s) { return 1ULL << s; }

// coup : de (6 bits) | vers (6 bits) | promotion (type de pièce, 3 bits)
constexpr Move MOVE_NONE = 0;
constexpr Move make_move(int from, int to, int promo = 0) { return Move(from | (to << 6) | (promo << 12)); }
constexpr int from_sq(Move m) { return m & 63; }
constexpr int to_sq(Move m) { return (m >> 6) & 63; }
constexpr int promo_of(Move m) { return (m >> 12) & 7; }

#if defined(_MSC_VER)
#include <intrin.h>
inline int lsb(Bitboard b) { unsigned long i; _BitScanForward64(&i, b); return (int)i; }
inline int msb(Bitboard b) { unsigned long i; _BitScanReverse64(&i, b); return (int)i; }
inline int popcount(Bitboard b) { return (int)__popcnt64(b); }
#else
inline int lsb(Bitboard b) { return __builtin_ctzll(b); }
inline int msb(Bitboard b) { return 63 - __builtin_clzll(b); }
inline int popcount(Bitboard b) { return __builtin_popcountll(b); }
#endif
inline int pop_lsb(Bitboard& b) { int s = lsb(b); b &= b - 1; return s; }

std::string square_name(int s);
std::string move_to_uci(Move m);
