// Solution (S) : moteur d'échecs UCI utilisant le réseau P.O.W.N.
//   ./solution [pown.bin]          puis commandes UCI sur l'entrée standard
//   ./solution perft 5             nombre de positions (test des règles)
//   ./solution bench               vitesse de recherche
#include <iostream>
#include <sstream>
#include <thread>
#include <memory>
#include <fstream>
#include "search.h"

static const char* NAME = "Solution S 1.0 (P.O.W.N.)";

struct Engine {
    NNUE net;
    TT tt{64};
    Search search{net, tt};
    Position pos;
    std::atomic<bool> stop_flag{false};
    std::thread worker;

    Engine() { pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"); }
    void wait() { if (worker.joinable()) worker.join(); }
    ~Engine() { stop_flag = true; wait(); }

    void set_position(std::istringstream& in) {
        std::string tok, fen;
        in >> tok;
        if (tok == "startpos") {
            pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
            in >> tok;
        } else if (tok == "fen") {
            while (in >> tok && tok != "moves") fen += tok + " ";
            pos.set_fen(fen);
        }
        if (tok == "moves") {
            while (in >> tok) {
                Move m = parse_uci_move(pos, tok);
                if (m == MOVE_NONE || !pos.do_move(m)) { std::cout << "info string coup illégal : " << tok << std::endl; break; }
            }
        }
    }

    void go(std::istringstream& in) {
        wait();
        Limits lim;
        std::string tok;
        while (in >> tok) {
            if (tok == "depth") in >> lim.depth;
            else if (tok == "nodes") in >> lim.nodes;
            else if (tok == "movetime") in >> lim.movetime;
            else if (tok == "wtime") in >> lim.wtime;
            else if (tok == "btime") in >> lim.btime;
            else if (tok == "winc") in >> lim.winc;
            else if (tok == "binc") in >> lim.binc;
            else if (tok == "movestogo") in >> lim.movestogo;
            else if (tok == "infinite") lim.infinite = true;
            else if (tok == "perft") { int d; in >> d; std::cout << "nodes " << perft(pos, d) << std::endl; return; }
        }
        if (!net.loaded && !search.use_material) {
            std::cout << "info string ERREUR : poids P.O.W.N. non chargés (option EvalFile ou argument pown.bin)" << std::endl;
        }
        stop_flag = false;
        Position root = pos;
        worker = std::thread([this, root, lim]() {
            SearchResult r = search.go(root, lim, stop_flag);
            std::cout << "bestmove " << move_to_uci(r.best) << std::endl;
        });
    }
};

static void bench(Engine& e) {
    const char* fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "2r2rk1/1p2qppp/p1n1pn2/3p4/3P4/P1NBPN2/1P3PPP/2RQ1RK1 w - - 0 1",
        "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 0 8",
    };
    int64_t total = 0;
    auto t0 = std::chrono::steady_clock::now();
    e.search.quiet = true;
    for (auto f : fens) {
        Position p; p.set_fen(f);
        e.tt.clear();
        Limits l; l.depth = 14; l.nodes = 300000;   // borné : durée prévisible
        std::atomic<bool> st{false};
        SearchResult r = e.search.go(p, l, st);
        total += r.nodes;
        std::cout << f << " -> " << move_to_uci(r.best) << " score " << r.score << " noeuds " << r.nodes << std::endl;
    }
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "total " << total << " noeuds, " << s << " s, " << (int64_t)(total / s) << " noeuds/s" << std::endl;
}

int main(int argc, char** argv) {
    Position::init();
    Engine e;
    std::string weights, action;
    int action_arg = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--random") e.net.load_random(1);
        else if (a == "--material") e.search.use_material = true;
        else if (a == "perft" && i + 1 < argc) { action = a; action_arg = std::atoi(argv[++i]); }
        else if (a == "bench") action = a;
        else weights = a;
    }
    if (!weights.empty() && !e.net.load(weights)) std::cout << "info string impossible de lire " << weights << std::endl;
    if (action == "perft") {
        Position p; p.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        std::cout << perft(p, action_arg) << std::endl;
        return 0;
    }
    if (action == "bench") {
        if (!e.net.loaded && !e.search.use_material) e.net.load_random(1);
        bench(e);
        return 0;
    }

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string cmd;
        in >> cmd;
        if (cmd == "uci") {
            std::cout << "id name " << NAME << "\nid author Liam\n"
                      << "option name Hash type spin default 64 min 1 max 4096\n"
                      << "option name EvalFile type string default pown.bin\n"
                      << "option name Threads type spin default 1 min 1 max 1\n"
                      << "option name Move Overhead type spin default 30 min 0 max 5000\n"
                      << "option name Material type check default false\n"
                      << "uciok" << std::endl;
        } else if (cmd == "isready") {
            std::cout << "readyok" << std::endl;
        } else if (cmd == "ucinewgame") {
            e.wait(); e.tt.clear();
        } else if (cmd == "setoption") {
            std::string tok, name, value;
            in >> tok;
            while (in >> tok && tok != "value") name += (name.empty() ? "" : " ") + tok;
            std::getline(in, value);
            if (!value.empty() && value[0] == ' ') value.erase(0, 1);
            if (name == "Hash") { e.wait(); e.tt.resize(std::max(1, std::atoi(value.c_str()))); }
            else if (name == "EvalFile") { if (!e.net.load(value)) std::cout << "info string impossible de lire " << value << std::endl; }
            else if (name == "Move Overhead") e.search.move_overhead = std::atoi(value.c_str());
            else if (name == "Material") e.search.use_material = (value == "true");
        } else if (cmd == "position") {
            e.wait(); e.set_position(in);
        } else if (cmd == "go") {
            e.go(in);
        } else if (cmd == "stop") {
            e.stop_flag = true; e.wait();
        } else if (cmd == "d") {
            std::cout << e.pos.fen() << std::endl;
        } else if (cmd == "eval") {
            Accumulator a; e.net.refresh_all(e.pos, a);
            std::cout << "eval " << e.net.evaluate(a, e.pos.stm) << " cp (point de vue du camp au trait)" << std::endl;
        } else if (cmd == "legalmoves") {
            MoveList l; generate(e.pos, l, GEN_ALL);
            for (int i = 0; i < l.n; i++) if (e.pos.do_move(l.m[i])) { e.pos.undo_move(); std::cout << move_to_uci(l.m[i]) << " "; }
            std::cout << std::endl;
        } else if (cmd == "bench") {
            e.wait(); if (!e.net.loaded && !e.search.use_material) e.net.load_random(1); bench(e);
        } else if (cmd == "quit") {
            break;
        }
    }
    e.stop_flag = true; e.wait();
    return 0;
}
