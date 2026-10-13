// Test : pour chaque ligne "fen;coup coup ...", joue les coups en mettant l'accumulateur à jour
// de façon incrémentale et affiche : évaluation incrémentale, évaluation recalculée à zéro.
#include <iostream>
#include <sstream>
#include <vector>
#include "movegen.h"
#include "nnue.h"
int main(int argc, char** argv) {
    Position::init();
    NNUE net;
    if (argc < 2 || !net.load(argv[1])) { std::cerr << "poids introuvables\n"; return 1; }
    std::string line;
    while (std::getline(std::cin, line)) {
        auto sc = line.find(';');
        Position pos;
        if (!pos.set_fen(line.substr(0, sc))) { std::cout << "ERR\n"; continue; }
        std::vector<Accumulator> acc(300);
        net.refresh_all(pos, acc[0]);
        std::istringstream ms(line.substr(sc + 1));
        std::string mv;
        int ply = 0;
        while (ms >> mv) {
            Move m = parse_uci_move(pos, mv);
            if (m == MOVE_NONE || !pos.do_move(m)) { std::cout << "ERRMOVE " << mv << "\n"; break; }
            net.update(pos, acc[ply], acc[ply + 1], pos.last());
            ply++;
        }
        Accumulator full;
        net.refresh_all(pos, full);
        std::cout << net.evaluate(acc[ply], pos.stm) << " " << net.evaluate(full, pos.stm) << " " << pos.fen() << "\n";
    }
}
