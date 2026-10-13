#include "nnue.h"
#include <cstdio>
#include <cstring>
#include <random>
#include <algorithm>

int NNUE::feature(int persp, int ksq, int piece, int sq) {
    int pc = color_of(piece), pt = type_of(piece);
    if (pc == persp && pt == KING) return -1;
    int kf = file_of(ksq);
    bool mirror = kf >= 4;
    int bucket = rank_of(ksq) * 4 + (mirror ? 7 - kf : kf);
    int f = file_of(sq);
    if (mirror) f = 7 - f;
    int t = (pc == persp) ? pt - 1 : 5 + pt - 1;
    return bucket * 704 + (rank_of(sq) * 8 + f) * 11 + t;
}

bool NNUE::load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[4];
    int32_t version = 0, dims[4] = {0, 0, 0, 0};
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "POWN", 4) != 0 || fread(&version, 4, 1, f) != 1 || version != 1
        || fread(dims, 4, 4, f) != 4 || dims[0] != NNUE_F || dims[1] != NNUE_H || dims[2] != NNUE_H1 || dims[3] != NNUE_H2) {
        fclose(f);
        return false;
    }
    ft.resize((size_t)NNUE_F * NNUE_H);
    std::vector<float> w1((size_t)NNUE_H1 * 2 * NNUE_H), w2((size_t)NNUE_H2 * NNUE_H1);
    bool ok = fread(ft.data(), 4, ft.size(), f) == ft.size() && fread(fb, 4, NNUE_H, f) == NNUE_H
        && fread(w1.data(), 4, w1.size(), f) == w1.size() && fread(b1, 4, NNUE_H1, f) == NNUE_H1
        && fread(w2.data(), 4, w2.size(), f) == w2.size() && fread(b2, 4, NNUE_H2, f) == NNUE_H2
        && fread(w3, 4, NNUE_H2, f) == NNUE_H2 && fread(&b3, 4, 1, f) == 1;
    fclose(f);
    if (!ok) return false;
    // torch stocke (sortie, entrée) ; on transpose pour parcourir les entrées en dehors, les sorties en dedans
    for (int o = 0; o < NNUE_H1; o++)
        for (int i = 0; i < 2 * NNUE_H; i++) w1t[i * NNUE_H1 + o] = w1[(size_t)o * 2 * NNUE_H + i];
    for (int o = 0; o < NNUE_H2; o++)
        for (int i = 0; i < NNUE_H1; i++) w2t[i * NNUE_H2 + o] = w2[(size_t)o * NNUE_H1 + i];
    loaded = true;
    return true;
}

void NNUE::load_random(unsigned seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> n01(0.f, 1.f);
    ft.resize((size_t)NNUE_F * NNUE_H);
    for (auto& x : ft) x = n01(g) * 0.05f;
    for (auto& x : fb) x = 0.5f;
    for (auto& x : w1t) x = n01(g) * 0.04f;
    for (auto& x : b1) x = n01(g) * 0.04f;
    for (auto& x : w2t) x = n01(g) * 0.17f;
    for (auto& x : b2) x = n01(g) * 0.17f;
    for (auto& x : w3) x = n01(g) * 0.17f;
    b3 = 0.1f;
    loaded = true;
}

void NNUE::refresh(const Position& pos, Accumulator& a, int persp) const {
    float* v = a.v[persp];
    std::memcpy(v, fb, sizeof(float) * NNUE_H);
    int ksq = pos.king_sq(persp);
    Bitboard occ = pos.pieces();
    while (occ) {
        int s = pop_lsb(occ);
        int idx = feature(persp, ksq, pos.board[s], s);
        if (idx < 0) continue;
        const float* row = &ft[(size_t)idx * NNUE_H];
        for (int j = 0; j < NNUE_H; j++) v[j] += row[j];
    }
}

void NNUE::update(const Position& pos, const Accumulator& from, Accumulator& to, const StateInfo& st) const {
    for (int persp = 0; persp < 2; persp++) {
        // si le roi de ce point de vue a bougé, la case de référence change : on recalcule tout
        bool king_moved = false;
        for (int i = 0; i < st.dn; i++)
            if (st.d[i].piece == make_piece(persp, KING) && st.d[i].from >= 0 && st.d[i].to >= 0) king_moved = true;
        if (king_moved) { refresh(pos, to, persp); continue; }
        float* v = to.v[persp];
        std::memcpy(v, from.v[persp], sizeof(float) * NNUE_H);
        int ksq = pos.king_sq(persp);
        for (int i = 0; i < st.dn; i++) {
            const PieceDelta& d = st.d[i];
            if (d.from >= 0) {
                int idx = feature(persp, ksq, d.piece, d.from);
                if (idx >= 0) { const float* row = &ft[(size_t)idx * NNUE_H]; for (int j = 0; j < NNUE_H; j++) v[j] -= row[j]; }
            }
            if (d.to >= 0) {
                int idx = feature(persp, ksq, d.piece, d.to);
                if (idx >= 0) { const float* row = &ft[(size_t)idx * NNUE_H]; for (int j = 0; j < NNUE_H; j++) v[j] += row[j]; }
            }
        }
    }
}

static inline float clip01(float x) { return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); }

int NNUE::evaluate(const Accumulator& a, int stm) const {
    alignas(64) float x[2 * NNUE_H], h1[NNUE_H1], h2[NNUE_H2];
    const float* us = a.v[stm];
    const float* them = a.v[stm ^ 1];
    for (int i = 0; i < NNUE_H; i++) { x[i] = clip01(us[i]); x[NNUE_H + i] = clip01(them[i]); }
    std::memcpy(h1, b1, sizeof(h1));
    for (int i = 0; i < 2 * NNUE_H; i++) {
        float xi = x[i];
        if (xi == 0.f) continue;                      // beaucoup de zéros après l'écrêtage : on les saute
        const float* w = &w1t[i * NNUE_H1];
        for (int o = 0; o < NNUE_H1; o++) h1[o] += w[o] * xi;
    }
    for (int o = 0; o < NNUE_H1; o++) h1[o] = clip01(h1[o]);
    std::memcpy(h2, b2, sizeof(h2));
    for (int i = 0; i < NNUE_H1; i++) {
        float xi = h1[i];
        if (xi == 0.f) continue;
        const float* w = &w2t[i * NNUE_H2];
        for (int o = 0; o < NNUE_H2; o++) h2[o] += w[o] * xi;
    }
    float out = b3;
    for (int o = 0; o < NNUE_H2; o++) out += clip01(h2[o]) * w3[o];
    float cp = 400.f * out;
    return (int)std::max(-3000.f, std::min(3000.f, cp));
}
