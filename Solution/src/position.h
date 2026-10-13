// Position d'échecs : bitboards + tableau de cases, coups jouables/annulables, hachage de Zobrist.
#pragma once
#include <string>
#include <vector>
#include "bitboard.h"

constexpr int CASTLE_WK = 1, CASTLE_WQ = 2, CASTLE_BK = 4, CASTLE_BQ = 8;

// changement de pièce provoqué par un coup (sert à mettre à jour le réseau NNUE sans tout recalculer)
struct PieceDelta { int piece, from, to; };   // from = -1 : pièce ajoutée ; to = -1 : pièce retirée

struct StateInfo {
    uint64_t key;
    int castling, ep, halfmove, captured, cap_sq;
    Move move;
    int dn;
    PieceDelta d[3];
};

class Position {
public:
    Bitboard byType[7] = {}, byColor[2] = {};
    int board[64] = {};
    int stm = WHITE, castling = 0, ep = -1, halfmove = 0, fullmove = 1;
    uint64_t key = 0;
    std::vector<StateInfo> hist;

    static void init();                  // tables de Zobrist
    bool set_fen(const std::string& fen);
    std::string fen() const;

    Bitboard pieces() const { return byColor[WHITE] | byColor[BLACK]; }
    Bitboard pieces(int c, int t) const { return byColor[c] & byType[t]; }
    int king_sq(int c) const { return lsb(byType[KING] & byColor[c]); }

    Bitboard attackers_to(int sq, Bitboard occ) const;
    bool attacked(int sq, int by) const;
    bool in_check() const { return attacked(king_sq(stm), stm ^ 1); }

    bool do_move(Move m);                // rend false (et annule) si le coup laisse le roi en échec
    void undo_move();
    void do_null();
    void undo_null();

    bool is_capture(Move m) const;
    bool has_non_pawn_material(int c) const;
    bool insufficient_material() const;
    bool is_repetition(int ply_from_root) const;
    int see(Move m) const;               // gain matériel estimé de l'échange sur la case d'arrivée (en centipions)
    const StateInfo& last() const { return hist.back(); }

private:
    uint64_t ep_key() const;
    void put(int p, int s);
    void remove(int s);
    void move_piece(int a, int b);
};
