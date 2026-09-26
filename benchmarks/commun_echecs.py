"""Code commun aux notebooks de benchmark échecs (PAWN, PAWN big, contrôle, SPAWN/PWN).

Tout vient de pawn_app.py : même encodage 69 octets, même recherche « arbre complet »,
même alpha-bêta + quiescence. Seule différence : pas de bruit ni de coups au hasard,
on veut mesurer la force réelle.
"""
import glob
import os
import shutil
import stat
import subprocess
import tarfile
import time
import urllib.request

import chess
import chess.engine
import chess.polyglot
import numpy as np
import onnxruntime as ort

DEPOT = "LiamLitle/ia-de-liam"
BRANCHE = "main"
LFS = f"https://media.githubusercontent.com/media/{DEPOT}/{BRANCHE}/"
DOSSIER_POIDS = os.environ.get("PAWN_POIDS", "/kaggle/working/poids" if os.path.isdir("/kaggle") else "poids")
MATE = 100_000

# nom -> chemin dans le dépôt
RESEAUX = {
    "PAWN": "Pawn/pawn.onnx",
    "PAWN_BIG": "PawnBig-V1/pawn_big.onnx",
    "CONTROLE": "PawnBig-controlleur/pawn_big_controle.onnx",
    "SPAWN": "PWN-soluce/pawn_soluce.onnx",
}


def poids(chemin_depot):
    """cherche le fichier dans /kaggle/input (dataset ajouté au notebook), sinon le télécharge depuis GitHub LFS"""
    nom = os.path.basename(chemin_depot)
    for racine in ("/kaggle/input", DOSSIER_POIDS):
        trouves = glob.glob(os.path.join(racine, "**", nom), recursive=True)
        trouves = [t for t in trouves if os.path.getsize(t) > 1000]  # pas un pointeur LFS
        if trouves:
            return trouves[0]
    os.makedirs(DOSSIER_POIDS, exist_ok=True)
    dest = os.path.join(DOSSIER_POIDS, nom)
    print(f"téléchargement de {chemin_depot} ...")
    urllib.request.urlretrieve(LFS + chemin_depot, dest)
    if os.path.getsize(dest) < 1000:
        raise RuntimeError(f"{dest} ressemble à un pointeur LFS, pas au vrai fichier")
    return dest


_CUDA_OK = None  # None = pas encore essayé, False = échec -> on reste sur CPU sans réessayer


def providers(gpu=True):
    dispo = ort.get_available_providers()
    if gpu and _CUDA_OK is not False and "CUDAExecutionProvider" in dispo:
        try:
            ort.preload_dlls()  # charge CUDA/cuDNN depuis les paquets pip nvidia-* (déjà là avec torch sur Kaggle)
        except Exception:
            pass
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


PIECES_ID = {c: i + 1 for i, c in enumerate("PNBRQK")}


def enc_fen(fen):
    f = fen.split(" ")
    noir = f[1] == "b"
    out = bytearray(69)
    r = c = 0
    for ch in f[0]:
        if ch == "/":
            r += 1
            c = 0
        elif ch.isdigit():
            c += int(ch)
        else:
            nous = ch.isupper() != noir
            out[(r if noir else 7 - r) * 8 + c] = PIECES_ID[ch.upper()] + (0 if nous else 6)
            c += 1
    ours, theirs = ("kq", "KQ") if noir else ("KQ", "kq")
    out[64] = ours[0] in f[2]
    out[65] = ours[1] in f[2]
    out[66] = theirs[0] in f[2]
    out[67] = theirs[1] in f[2]
    if f[3] != "-":
        out[68] = ord(f[3][0]) - 96
    return bytes(out)


class Evaluateur:
    """un réseau ONNX : liste de FEN -> centipions du point de vue du camp au trait"""

    def __init__(self, nom, gpu=True, threads=0):
        self.nom = nom
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
        global _CUDA_OK
        prov = providers(gpu)
        self.session = ort.InferenceSession(poids(RESEAUX[nom]), so, providers=prov)
        if "CUDAExecutionProvider" in prov:
            _CUDA_OK = "CUDAExecutionProvider" in self.session.get_providers()
            if not _CUDA_OK:
                print("⚠️ GPU indisponible pour onnxruntime : on continue sur CPU (plus lent mais résultats identiques)")
                ort.set_default_logger_severity(4)
        self.nb_evals = 0
        self.cache = {}

    def evals(self, fens, lot=4096):
        if not fens:
            return np.zeros(0, dtype=np.float32)
        res = []
        for i in range(0, len(fens), lot):
            x = np.frombuffer(b"".join(map(enc_fen, fens[i:i + lot])), np.uint8).reshape(-1, 69).copy()
            res.append(self.session.run(None, {"board": x})[0])
        self.nb_evals += len(fens)
        return np.concatenate(res)

    def eval1(self, fen):
        # l'alpha-bêta réévalue souvent les mêmes positions en quiescence : le réseau est
        # déterministe, donc un cache ne change rien au résultat, juste la vitesse
        v = self.cache.get(fen)
        if v is None:
            if len(self.cache) > 2_000_000:
                self.cache.clear()
            v = self.cache[fen] = float(self.evals([fen])[0])
        return v


# -- recherche « arbre complet » (PAWN, PAWN big, contrôle, SPAWN) -----------------

def arbre(board, depth, ply, feuilles):
    if not any(board.legal_moves):
        return ("t", -(MATE - ply) if board.is_check() else 0)
    if ply > 0 and (board.is_insufficient_material() or board.is_repetition(2) or board.halfmove_clock >= 100):
        return ("t", 0)
    if depth == 0:
        feuilles.append(board.fen())
        return ("f", len(feuilles) - 1)
    fils = []
    for mv in list(board.legal_moves):
        board.push(mv)
        fils.append((mv, arbre(board, depth - 1, ply + 1, feuilles)))
        board.pop()
    return ("n", fils)


def valeur(noeud, v):
    genre, x = noeud
    if genre == "t":
        return x
    if genre == "f":
        return v[x]
    return max(-valeur(c, v) for _, c in x)


def choisir_arbre(board, ev, depth):
    feuilles = []
    racine = arbre(board, depth, 0, feuilles)
    v = ev.evals(feuilles) if feuilles else []
    notes = [(-valeur(c, v), mv) for mv, c in racine[1]]
    return max(notes, key=lambda t: t[0])[1]


# -- recherche alpha-bêta + quiescence (PWN) --------------------------------------

VAL_PIECE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def trier_coups(board, coups, priorite):
    def cle(mv):
        if mv == priorite:
            return 10_000
        if board.is_capture(mv):
            prise = board.piece_type_at(mv.to_square) or chess.PAWN
            attaquant = board.piece_type_at(mv.from_square)
            return 1000 + VAL_PIECE[prise] * 10 - VAL_PIECE[attaquant]
        if board.gives_check(mv):
            return 500
        return 0
    return sorted(coups, key=cle, reverse=True)


def quiescence(board, ev, alpha, beta, prof_q):
    stand_pat = ev.eval1(board.fen())
    if stand_pat >= beta:
        return beta
    alpha = max(alpha, stand_pat)
    if prof_q <= 0:
        return alpha
    captures = [m for m in board.legal_moves if board.is_capture(m)]
    for mv in trier_coups(board, captures, None):
        board.push(mv)
        score = -quiescence(board, ev, -beta, -alpha, prof_q - 1)
        board.pop()
        if score >= beta:
            return beta
        alpha = max(alpha, score)
    return alpha


def negamax(board, ev, profondeur, alpha, beta, ply, prof_q, tt, coup_tt):
    alpha0 = alpha
    if not any(board.legal_moves):
        return -(MATE - ply) if board.is_check() else 0
    if ply > 0 and (board.is_insufficient_material() or board.is_repetition(2) or board.halfmove_clock >= 100):
        return 0
    hash_ = chess.polyglot.zobrist_hash(board)
    entree = tt.get(hash_)
    if entree and entree[0] >= profondeur:
        d, v, drapeau, _ = entree
        if drapeau == "exact":
            return v
        if drapeau == "min":
            alpha = max(alpha, v)
        elif drapeau == "max":
            beta = min(beta, v)
        if alpha >= beta:
            return v
    if profondeur <= 0:
        return quiescence(board, ev, alpha, beta, prof_q)
    meilleur, meilleur_coup = -MATE - 1, None
    for mv in trier_coups(board, list(board.legal_moves), coup_tt.get(hash_)):
        board.push(mv)
        score = -negamax(board, ev, profondeur - 1, -beta, -alpha, ply + 1, prof_q, tt, coup_tt)
        board.pop()
        if score > meilleur:
            meilleur, meilleur_coup = score, mv
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    drapeau = "exact" if alpha0 < meilleur < beta else ("min" if meilleur >= beta else "max")
    tt[hash_] = (profondeur, meilleur, drapeau, meilleur_coup)
    if meilleur_coup is not None:
        coup_tt[hash_] = meilleur_coup
    return meilleur


def choisir_alphabeta(board, ev, depth, prof_q=4):
    tt, coup_tt = {}, {}
    meilleur_coup = None
    for d in range(1, depth + 1):
        alpha, beta = -MATE - 1, MATE + 1
        hash_ = chess.polyglot.zobrist_hash(board)
        meilleur_score, meilleur_du_tour = -MATE - 1, None
        for mv in trier_coups(board, list(board.legal_moves), coup_tt.get(hash_)):
            board.push(mv)
            score = -negamax(board, ev, d - 1, -beta, -alpha, 1, prof_q, tt, coup_tt)
            board.pop()
            if score > meilleur_score:
                meilleur_score, meilleur_du_tour = score, mv
            alpha = max(alpha, score)
        meilleur_coup = meilleur_du_tour
        coup_tt[hash_] = meilleur_coup
    return meilleur_coup


# -- joueurs ------------------------------------------------------------------------
# un « joueur » = un réseau + un algorithme de recherche + une profondeur.
# ex : "PAWN_BIG@arbre2", "SPAWN@arbre1", "PWN@ab3" (PWN = poids SPAWN + alpha-bêta).

def decoder(joueur):
    nom, rech = joueur.split("@")
    if nom == "PWN":
        nom = "SPAWN"
    if rech.startswith("arbre"):
        return nom, "arbre", int(rech[5:])
    if rech.startswith("ab"):
        return nom, "ab", int(rech[2:])
    raise ValueError(joueur)


_EVALS = {}
_SF = {}
GPU = True
THREADS = 0
SF_CHEMIN = None
SF_TEMPS = 0.1


def evaluateur(nom, gpu=None):
    gpu = GPU if gpu is None else gpu
    if (nom, gpu) not in _EVALS:
        _EVALS[(nom, gpu)] = Evaluateur(nom, gpu=gpu, threads=THREADS)
    return _EVALS[(nom, gpu)]


def configurer(gpu=True, threads=0, sf_chemin=None, sf_temps=0.1):
    """à appeler dans chaque processus fils avant de jouer"""
    global GPU, THREADS, SF_CHEMIN, SF_TEMPS
    GPU, THREADS, SF_CHEMIN, SF_TEMPS = gpu, threads, sf_chemin, sf_temps
    _EVALS.clear()
    _SF.clear()


def stockfish(cfg):
    """cfg = "elo1500" (UCI_LimitStrength) ou "skill0" (Skill Level)"""
    if cfg not in _SF:
        e = chess.engine.SimpleEngine.popen_uci(SF_CHEMIN)
        opts = {"Threads": 1, "Hash": 16}
        if cfg.startswith("elo"):
            opts.update({"UCI_LimitStrength": True, "UCI_Elo": int(cfg[3:])})
        elif cfg.startswith("skill"):
            opts["Skill Level"] = int(cfg[5:])
        e.configure(opts)
        _SF[cfg] = e
    return _SF[cfg]


def fermer_stockfish():
    for e in _SF.values():
        e.quit()
    _SF.clear()


def coup(joueur, board):
    if joueur.startswith("SF@"):
        return stockfish(joueur[3:]).play(board, chess.engine.Limit(time=SF_TEMPS)).move
    nom, rech, depth = decoder(joueur)
    b = board.copy(stack=True)
    if rech == "arbre":
        return choisir_arbre(b, evaluateur(nom), depth)
    # l'alpha-bêta évalue une position à la fois : sur Kaggle la latence CPU (~6 ms)
    # est deux fois plus basse que celle du GPU (~14 ms), donc on reste sur CPU
    return choisir_alphabeta(b, evaluateur(nom, gpu=False), depth)


# -- Stockfish -------------------------------------------------------------------------

URLS_STOCKFISH = [
    "https://github.com/official-stockfish/Stockfish/releases/download/sf_17.1/stockfish-ubuntu-x86-64-avx2.tar",
    "https://github.com/official-stockfish/Stockfish/releases/download/sf_17/stockfish-ubuntu-x86-64-avx2.tar",
    "https://github.com/official-stockfish/Stockfish/releases/download/sf_16.1/stockfish-ubuntu-x86-64-avx2.tar",
]


def installer_stockfish(dossier=None):
    dossier = dossier or os.path.join(os.path.dirname(os.path.abspath(DOSSIER_POIDS)), "stockfish")
    deja = glob.glob(os.path.join(dossier, "**", "stockfish-ubuntu-*"), recursive=True)
    deja = [d for d in deja if os.path.isfile(d) and not d.endswith(".tar")]
    if deja:
        return deja[0]
    os.makedirs(dossier, exist_ok=True)
    for url in URLS_STOCKFISH:
        try:
            tar = os.path.join(dossier, "sf.tar")
            urllib.request.urlretrieve(url, tar)
            with tarfile.open(tar) as t:
                t.extractall(dossier)
            binaire = [d for d in glob.glob(os.path.join(dossier, "**", "stockfish-ubuntu-*"), recursive=True)
                       if os.path.isfile(d) and not d.endswith(".tar")][0]
            os.chmod(binaire, os.stat(binaire).st_mode | stat.S_IEXEC)
            subprocess.run([binaire, "quit"], check=True, timeout=10)
            return binaire
        except Exception as e:
            print("échec", url, e)
    if shutil.which("stockfish") is None:
        subprocess.run("apt-get -qq install -y stockfish", shell=True)
    for p in (shutil.which("stockfish"), "/usr/games/stockfish"):
        if p and os.path.exists(p):
            return p
    raise RuntimeError("impossible d'installer Stockfish")


# -- puzzles Lichess ---------------------------------------------------------------

def resoudre_puzzle(joueur, fen, moves):
    """format Lichess : FEN avant le coup adverse, moves[0] = coup adverse, puis solution.
    Comme sur Lichess, un mat est toujours accepté même si ce n'est pas le coup attendu.
    renvoie (résolu, premier_coup_bon, nb_coups_joués, secondes)"""
    b = chess.Board(fen)
    moves = moves.split()
    b.push_uci(moves[0])
    premier, joues, t0 = None, 0, time.perf_counter()
    for i in range(1, len(moves), 2):
        attendu = chess.Move.from_uci(moves[i])
        mv = coup(joueur, b)
        joues += 1
        b.push(mv)
        mat = b.is_checkmate()
        b.pop()
        ok = mv == attendu or mat
        if premier is None:
            premier = ok
        if not ok:
            return False, premier, joues, time.perf_counter() - t0
        if mat:
            break
        b.push(attendu)
        if i + 1 < len(moves):
            b.push_uci(moves[i + 1])
    return True, premier, joues, time.perf_counter() - t0


# -- parties -------------------------------------------------------------------------

def jouer_partie(blancs, noirs, ouverture, max_plies=300):
    """ouverture = liste de coups UCI joués d'office ; renvoie un dict (résultat, pgn, temps)"""
    import chess.pgn
    b = chess.Board()
    for u in ouverture:
        b.push_uci(u)
    temps = {blancs: 0.0, noirs: 0.0}
    nb = {blancs: 0, noirs: 0}
    while not b.is_game_over(claim_draw=True) and b.ply() < max_plies:
        j = blancs if b.turn == chess.WHITE else noirs
        t0 = time.perf_counter()
        mv = coup(j, b)
        temps[j] += time.perf_counter() - t0
        nb[j] += 1
        b.push(mv)
    res = b.result(claim_draw=True) if b.is_game_over(claim_draw=True) else "1/2-1/2"
    fin = b.outcome(claim_draw=True)
    jeu = chess.pgn.Game.from_board(b)
    jeu.headers.update(Event="Benchmark P.A.W.N.", White=blancs, Black=noirs, Result=res)
    return {
        "blancs": blancs, "noirs": noirs, "resultat": res,
        "fin": fin.termination.name if fin else "MAX_PLIES",
        "plies": b.ply(), "pgn": str(jeu),
        "s_par_coup_blancs": temps[blancs] / max(1, nb[blancs]),
        "s_par_coup_noirs": temps[noirs] / max(1, nb[noirs]),
    }


# -- tâches pour ProcessPoolExecutor (doivent être importables depuis ce module) -------

def init_worker(gpu, threads, sf_chemin=None, sf_temps=0.1):
    configurer(gpu=gpu, threads=threads, sf_chemin=sf_chemin, sf_temps=sf_temps)


def tache_puzzle(args):
    joueur, pid, fen, moves = args
    ok, premier, joues, sec = resoudre_puzzle(joueur, fen, moves)
    return {"joueur": joueur, "PuzzleId": pid, "resolu": ok, "premier_coup": premier,
            "coups_joues": joues, "secondes": sec}


def tache_partie(args):
    blancs, noirs, ouverture, max_plies, id_ouv = args
    try:
        r = jouer_partie(blancs, noirs, ouverture, max_plies)
    finally:
        # un Stockfish ouvert garde un thread vivant qui empêche le processus fils de se terminer
        fermer_stockfish()
    r["ouverture"] = id_ouv
    return r
