"""Génère les notebooks Kaggle de benchmark à partir de ce fichier.

    python construire_notebooks.py

Les notebooks échecs embarquent commun_echecs.py (via %%writefile) pour être autonomes sur Kaggle.
"""
import json
import os

ICI = os.path.dirname(os.path.abspath(__file__))


def notebook(cellules, gpu=True):
    cells = []
    for genre, texte in cellules:
        texte = texte.strip("\n")
        lignes = [l + "\n" for l in texte.split("\n")]
        lignes[-1] = lignes[-1].rstrip("\n")
        if genre == "md":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": lignes})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lignes})
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4" if gpu else "none", "isInternetEnabled": True,
                       "isGpuEnabled": gpu, "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def ecrire(nom, cellules, gpu=True):
    with open(os.path.join(ICI, nom), "w", encoding="utf-8") as f:
        json.dump(notebook(cellules, gpu), f, ensure_ascii=False, indent=1)
    print("écrit", nom)


with open(os.path.join(ICI, "commun_echecs.py"), encoding="utf-8") as f:
    COMMUN = f.read()

INSTALL_ECHECS = r'''
# onnxruntime-gpu remplace onnxruntime (les deux ne cohabitent pas) ; marche aussi sans GPU.
# Version fixée : les plus récentes demandent CUDA 13, Kaggle a CUDA 12. [cuda,cudnn] installe les bibliothèques CUDA 12.
!pip uninstall -y -q onnxruntime onnxruntime-gpu > /dev/null 2>&1
!pip install -q chess zstandard "onnxruntime-gpu[cuda,cudnn]==1.24.1"
'''

MODULE_ECHECS = "%%writefile commun_echecs.py\n" + COMMUN

IMPORT_ECHECS = r'''
import os, sys, time, json, random
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import chess
sys.path.insert(0, os.getcwd())  # pour que les processus fils trouvent commun_echecs.py
import onnxruntime as ort
import commun_echecs as ce

print("onnxruntime", ort.__version__, "| providers dispo :", ort.get_available_providers())
ev_test = ce.Evaluateur("PAWN")
print("session PAWN sur :", ev_test.session.get_providers())
print("éval position initiale :", ev_test.eval1(chess.STARTING_FEN), "cp")
'''

# =====================================================================================
# 01 — précision de l'évaluation
# =====================================================================================

NB1 = [
    ("md", r'''
# ♟️ Benchmark 1 — Précision de l'évaluation des réseaux P.A.W.N.

On compare les **4 réseaux** d'évaluation (sans recherche) à **Stockfish** (profondeur ≥ 20) sur des positions
qu'**aucun modèle n'a vues à l'entraînement** :

| Réseau | Dossier | Taille | Données d'entraînement |
|---|---|---|---|
| PAWN | `Pawn/` | 8×96, 1,41 M | 30 M premières positions |
| PAWN_BIG | `PawnBig-V1/` | 12×128, 3,64 M | 60 M premières positions |
| CONTROLE | `PawnBig-controlleur/` | 12×128, 3,64 M | PAWN big + 30 M suivantes (sans Soluce) |
| SPAWN (= réseau de PWN) | `PWN-soluce/` | 12×128 + Soluce, 3,65 M | PAWN big + 30 M suivantes (avec Soluce) |

Les modèles ont été entraînés sur les **90 M premières lignes** de `lichess/chess-evaluations` : on prend donc les positions
de test dans les **derniers fichiers parquet** du dataset (le notebook vérifie qu'ils commencent après la ligne 90 M).

**Métriques**
- **BCE** : la même perte que pendant l'entraînement (cible `sigmoid(cp/400)`) → comparable aux pertes de validation
- **MAE win-prob** : erreur moyenne sur la probabilité de gain (en points de %)
- **MAE cp** : erreur en centipions (évaluations bornées à ±1000)
- **Spearman** : est-ce que le modèle classe les positions dans le même ordre que Stockfish ?
- **Bon camp** : le modèle dit-il le bon camp gagnant quand |éval Stockfish| ≥ 100 cp ?
- **Accord coup** : à profondeur 1, le coup choisi par le réseau est-il le meilleur coup de Stockfish ? (top-1 / top-3)
- **Symétrie** : une position et son miroir gauche↔droite (sans roque) devraient avoir la même éval
- **Vitesse** : latence (1 position) et débit (positions/s) CPU et GPU

⚙️ Réglages Kaggle : **Internet ON**, accélérateur **GPU T4** (facultatif mais plus rapide).
'''),
    ("code", INSTALL_ECHECS),
    ("code", MODULE_ECHECS),
    ("code", IMPORT_ECHECS),
    ("code", r'''
# ---- réglages ----
N_POSITIONS = 200_000   # positions de test pour les métriques d'éval
N_COUPS = 5_000         # positions pour l'accord de coup avec Stockfish
N_SYMETRIE = 20_000     # positions pour le test de symétrie
MIN_DEPTH = 20          # comme à l'entraînement
DEJA_VUS = 90_000_000   # PAWN : lignes 0-30 M, PAWN big : 0-60 M, SPAWN/contrôle : 60-90 M (cf. specs)
SEED = 0
GPU = True
SORTIE = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
rng = np.random.default_rng(SEED)
'''),
    ("md", r'''
## 1. Positions de test (jamais vues à l'entraînement)
'''),
    ("code", r'''
from huggingface_hub import HfApi, HfFileSystem
import pyarrow.parquet as pq

for REPO in ["Lichess/chess-position-evaluations", "lichess/chess-evaluations"]:
    try:
        fichiers = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet"))
        if fichiers:
            break
    except Exception as e:
        print(REPO, "->", e)
print(REPO, ":", len(fichiers), "fichiers parquet")

fs = HfFileSystem()
nb_lignes = []
for f in fichiers:  # on ne lit que le pied de page de chaque parquet (quelques Ko)
    with fs.open(f"datasets/{REPO}/{f}") as h:
        nb_lignes.append(pq.ParquetFile(h).metadata.num_rows)
debuts = np.cumsum([0] + nb_lignes[:-1])
print(f"total : {sum(nb_lignes):,} lignes")

# on part du dernier fichier et on remonte tant qu'il faut, sans jamais descendre sous DEJA_VUS
choisis = []
for i in range(len(fichiers) - 1, -1, -1):
    if debuts[i] < DEJA_VUS:
        break
    choisis.append(i)
    if sum(nb_lignes[j] for j in choisis) > 6 * N_POSITIONS:
        break
assert choisis, "aucun fichier entièrement après les 90 M premières lignes !"
print("fichiers de test :", [fichiers[i] for i in choisis], "| première ligne :", f"{min(debuts[i] for i in choisis):,}")

from huggingface_hub import hf_hub_download
morceaux = []
for i in choisis:
    chemin = hf_hub_download(REPO, fichiers[i], repo_type="dataset")
    cols = [c for c in ["fen", "line", "depth", "cp", "mate"] if c in pq.ParquetFile(chemin).schema.names]
    morceaux.append(pd.read_parquet(chemin, columns=cols))
brut = pd.concat(morceaux, ignore_index=True)
print(brut.shape); brut.head()
'''),
    ("code", r'''
MATE_CP = 10_000
brut = brut[brut["depth"] >= MIN_DEPTH].copy()
# éval Lichess = point de vue des blancs ; mat -> ±(10000 - 10·n)
cp_blancs = brut["cp"].astype("float64")
m = brut["mate"].notna()
cp_blancs[m] = np.sign(brut.loc[m, "mate"]) * (MATE_CP - 10 * brut.loc[m, "mate"].abs())
brut["cp_blancs"] = cp_blancs
brut["trait_blanc"] = brut["fen"].str.split(" ").str[1] == "w"
brut["cp_trait"] = np.where(brut["trait_blanc"], brut["cp_blancs"], -brut["cp_blancs"])

# plusieurs lignes par position (multi-PV, plusieurs profondeurs) : la plus profonde, puis la meilleure pour le camp au trait
fens_uniques = brut["fen"].unique()
tirage = set(rng.choice(fens_uniques, size=min(N_POSITIONS, len(fens_uniques)), replace=False))
df = brut[brut["fen"].isin(tirage)].sort_values(["fen", "depth", "cp_trait"], ascending=[True, False, False])
df = df.groupby("fen", as_index=False).first()
del brut
# les FEN du dataset n'ont pas toujours les compteurs de coups
df["fen"] = df["fen"].apply(lambda f: f if len(f.split()) == 6 else f + " 0 1")
print(len(df), "positions de test"); df.head()
'''),
    ("code", r'''
VALEUR = {chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
def phase(fen):
    b = chess.Board(fen)
    mat = sum(VALEUR.get(p.piece_type, 0) for p in b.piece_map().values())
    return "ouverture" if mat >= 50 else ("milieu" if mat >= 20 else "finale")
df["phase"] = df["fen"].map(phase)
df["zone"] = pd.cut(df["cp_trait"].abs(), [-1, 50, 150, 400, 1000, 1e9],
                    labels=["égal <50", "léger 50-150", "net 150-400", "gagnant 400-1000", "décisif/mat"])
df[["phase", "zone"]].value_counts().unstack()
'''),
    ("md", r'''
## 2. Évaluation des 4 réseaux
'''),
    ("code", r'''
def wp(cp):
    return 1 / (1 + np.exp(-np.asarray(cp, dtype=np.float64) / 400))

cible = df["cp_trait"].to_numpy()
p_cible = wp(cible)
fens = df["fen"].tolist()
preds = {}
for nom in ce.RESEAUX:
    ev = ce.Evaluateur(nom, gpu=GPU)
    t0 = time.perf_counter()
    preds[nom] = ev.evals(fens, lot=8192).astype(np.float64)
    print(f"{nom:9s} {len(fens) / (time.perf_counter() - t0):>10,.0f} positions/s  ({ev.session.get_providers()[0]})")
    df["pred_" + nom] = preds[nom]
'''),
    ("code", r'''
from scipy.stats import spearmanr, pearsonr

def metriques(cible, pred):
    p, q = wp(cible), np.clip(wp(pred), 1e-7, 1 - 1e-7)
    c1, c2 = np.clip(cible, -1000, 1000), np.clip(pred, -1000, 1000)
    net = np.abs(cible) >= 100
    return {
        "BCE": float(np.mean(-(p * np.log(q) + (1 - p) * np.log(1 - q)))),
        "MAE win-prob (pts %)": float(100 * np.mean(np.abs(p - q))),
        "MAE cp (±1000)": float(np.mean(np.abs(c1 - c2))),
        "Pearson cp": float(pearsonr(c1, c2)[0]),
        "Spearman": float(spearmanr(cible, pred)[0]),
        "bon camp (%)": float(100 * np.mean(np.sign(cible[net]) == np.sign(pred[net]))),
    }

tab = pd.DataFrame({nom: metriques(cible, preds[nom]) for nom in preds}).T
# référence : BCE minimale atteignable (entropie de la cible) et un modèle « toujours 0 cp »
h = float(np.mean(-(p_cible * np.log(np.clip(p_cible, 1e-12, 1)) + (1 - p_cible) * np.log(np.clip(1 - p_cible, 1e-12, 1)))))
tab.loc["(réf.) toujours 0 cp"] = metriques(cible, np.zeros_like(cible))
print(f"entropie de la cible (BCE parfaite) = {h:.4f}")
tab.style.format("{:.4f}").background_gradient(axis=0, cmap="RdYlGn_r", subset=["BCE", "MAE win-prob (pts %)", "MAE cp (±1000)"])
'''),
    ("code", r'''
lignes = []
for groupe in ["phase", "zone"]:
    for val, sous in df.groupby(groupe, observed=True):
        for nom in preds:
            m = metriques(sous["cp_trait"].to_numpy(), sous["pred_" + nom].to_numpy())
            lignes.append({"groupe": groupe, "valeur": val, "modèle": nom, "n": len(sous), **m})
detail = pd.DataFrame(lignes)
detail.pivot_table(index=["groupe", "valeur", "n"], columns="modèle", values="MAE win-prob (pts %)").round(2)
'''),
    ("code", r'''
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
bins = np.linspace(0, 1, 21)
centre = (bins[:-1] + bins[1:]) / 2
for nom in preds:
    q = wp(preds[nom]); idx = np.digitize(q, bins) - 1
    moy = [p_cible[idx == k].mean() if (idx == k).sum() > 50 else np.nan for k in range(20)]
    axes[0].plot(centre, moy, marker="o", label=nom)
axes[0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0].set(xlabel="win-prob prédite", ylabel="win-prob Stockfish (moyenne)", title="Calibration")
axes[0].legend()
pv = detail[detail.groupe == "phase"].pivot(index="valeur", columns="modèle", values="MAE win-prob (pts %)")
pv.plot.bar(ax=axes[1], rot=0, title="MAE win-prob par phase (plus bas = mieux)")
plt.tight_layout(); plt.savefig(f"{SORTIE}/eval_calibration.png", dpi=120); plt.show()
'''),
    ("md", r'''
## 3. Accord avec le meilleur coup de Stockfish (profondeur 1, sans recherche)
Pour chaque position on évalue tous les coups légaux avec le réseau et on garde le meilleur ;
on compare avec le premier coup de la ligne principale de Stockfish.
'''),
    ("code", r'''
sous = df[df["line"].notna() & (df["line"].str.len() > 0)].sample(min(N_COUPS, len(df)), random_state=SEED)
enfants, index, terminaux, coups_sf, legaux = [], [], {}, [], []
for k, (fen, ligne) in enumerate(zip(sous["fen"], sous["line"])):
    b = chess.Board(fen)
    coups = list(b.legal_moves)
    legaux.append(coups)
    coups_sf.append(chess.Move.from_uci(ligne.split()[0]))
    for j, mv in enumerate(coups):
        b.push(mv)
        if b.is_checkmate():
            terminaux[(k, j)] = ce.MATE
        elif b.is_game_over():
            terminaux[(k, j)] = 0
        else:
            index.append((k, j)); enfants.append(b.fen())
        b.pop()
print(len(sous), "positions,", len(enfants), "enfants à évaluer")

accord = {}
for nom in ce.RESEAUX:
    v = ce.Evaluateur(nom, gpu=GPU).evals(enfants, lot=8192)
    scores = [np.full(len(c), -np.inf) for c in legaux]
    for (k, j), x in zip(index, v):
        scores[k][j] = -x
    for (k, j), x in terminaux.items():
        scores[k][j] = x
    top1 = top3 = 0
    for k, s in enumerate(scores):
        ordre = [legaux[k][j] for j in np.argsort(-s)]
        top1 += ordre[0] == coups_sf[k]
        top3 += coups_sf[k] in ordre[:3]
    accord[nom] = {"top-1 (%)": 100 * top1 / len(scores), "top-3 (%)": 100 * top3 / len(scores)}
accord = pd.DataFrame(accord).T
accord
'''),
    ("md", r'''
## 4. Symétrie gauche ↔ droite
Sans droits de roque, une position et son miroir (colonne a ↔ colonne h) ont exactement la même valeur.
Un bon évaluateur doit donner presque la même note aux deux.
'''),
    ("code", r'''
sans_roque = df[df["fen"].str.split(" ").str[2] == "-"].head(N_SYMETRIE)
miroirs = [chess.Board(f).transform(chess.flip_horizontal).fen() for f in sans_roque["fen"]]
sym = {}
for nom in ce.RESEAUX:
    a = sans_roque["pred_" + nom].to_numpy()
    b = ce.Evaluateur(nom, gpu=GPU).evals(miroirs, lot=8192)
    d = np.abs(a - b)
    sym[nom] = {"écart moyen (cp)": d.mean(), "écart médian (cp)": np.median(d), "% écart > 50 cp": 100 * (d > 50).mean()}
sym = pd.DataFrame(sym).T
print(len(sans_roque), "positions"); sym
'''),
    ("md", r'''
## 5. Vitesse et taille
'''),
    ("code", r'''
PARAMS_SPEC = {"PAWN": 1_414_906, "PAWN_BIG": 3_638_178, "CONTROLE": 3_638_178, "SPAWN": 3_649_698}
lot = fens[:4096]
vitesse = {}
for nom in ce.RESEAUX:
    ligne = {"paramètres": PARAMS_SPEC[nom], "fichier (Mo)": os.path.getsize(ce.poids(ce.RESEAUX[nom])) / 1e6}
    for dev in (["GPU", "CPU"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPU"]):
        ev = ce.Evaluateur(nom, gpu=dev == "GPU")
        ev.evals(lot[:64])  # chauffe
        t = []
        for f in lot[:200]:
            t0 = time.perf_counter(); ev.evals([f]); t.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); ev.evals(lot, lot=4096); debit = len(lot) / (time.perf_counter() - t0)
        ligne[f"latence {dev} (ms)"] = 1000 * np.median(t)
        ligne[f"débit {dev} (pos/s)"] = debit
    vitesse[nom] = ligne
vitesse = pd.DataFrame(vitesse).T
vitesse.round(2)
'''),
    ("md", r'''
## 6. Synthèse
'''),
    ("code", r'''
synthese = tab.drop(index="(réf.) toujours 0 cp").join(accord).join(sym).join(vitesse)
synthese.to_csv(f"{SORTIE}/bench1_evaluation.csv")
detail.to_csv(f"{SORTIE}/bench1_detail_phase_zone.csv", index=False)
cols = ["BCE", "MAE win-prob (pts %)", "Spearman", "bon camp (%)", "top-1 (%)", "top-3 (%)", "écart moyen (cp)"]
fig, axes = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 3.5))
for ax, c in zip(axes, cols):
    synthese[c].plot.bar(ax=ax, color=["#9aa", "#58a", "#a85", "#5a5"]); ax.set_title(c, fontsize=9); ax.tick_params(labelsize=8)
plt.tight_layout(); plt.savefig(f"{SORTIE}/bench1_synthese.png", dpi=120); plt.show()
synthese[cols].round(3)
'''),
    ("md", r'''
### Comment lire les résultats
- **CONTROLE vs SPAWN** est la comparaison clé : même base, mêmes données, même ordre → la différence vient **uniquement du module Soluce**.
- **PAWN_BIG vs CONTROLE** mesure l'effet de « juste plus d'entraînement ».
- Si **top-1** reste bas (< 40 %) alors que la BCE est bonne, c'est normal : l'éval statique ne voit pas les tactiques, c'est le rôle de la recherche (→ notebooks 2 et 3).
'''),
]

# =====================================================================================
# 02 — puzzles Lichess
# =====================================================================================

NB2 = [
    ("md", r'''
# ♟️ Benchmark 2 — Puzzles Lichess (réseau + recherche)

Le benchmark « 300 puzzles » des specs, en plus grand et **stratifié par difficulté** :
on tire des puzzles de la base officielle Lichess (~5 M puzzles notés) dans chaque tranche de classement,
et on mesure pour chaque *joueur* (réseau + recherche + profondeur) :

- **taux de résolution** (toute la séquence, un mat alternatif est accepté comme sur Lichess)
- **premier coup juste**
- **Elo puzzle estimé** : le classement de puzzle que le moteur résout 1 fois sur 2 (maximum de vraisemblance)
- **temps par coup**, résultats **par thème** (mat en 2, fourchette, clouage, finale…)
- **test de McNemar** entre modèles (même puzzles → écart significatif ou non)

Joueurs : `NOM@arbreN` = recherche arbre complet profondeur N (comme dans l'app), `PWN@abN` = poids SPAWN + alpha-bêta/quiescence profondeur N.

⚙️ Réglages Kaggle : **Internet ON**, **GPU T4** conseillé (la recherche arbre complet évalue ~1000 positions par coup en lot).
Les résultats sont sauvegardés au fur et à mesure : si la session coupe, relancer reprend où ça s'était arrêté.
'''),
    ("code", INSTALL_ECHECS),
    ("code", MODULE_ECHECS),
    ("code", IMPORT_ECHECS),
    ("code", r'''
# ---- réglages ----
TRANCHES = list(range(400, 2801, 200))   # bornes des tranches de classement puzzle
PAR_TRANCHE = 100                         # puzzles par tranche (12 tranches -> 1200 puzzles)
PAR_TRANCHE_LENT = 40                     # pour les joueurs lents (PWN@ab3)
JOUEURS = [
    "PAWN@arbre1", "PAWN_BIG@arbre1", "CONTROLE@arbre1", "SPAWN@arbre1",
    "PAWN@arbre2", "PAWN_BIG@arbre2", "CONTROLE@arbre2", "SPAWN@arbre2",
    "PWN@ab2", "PWN@ab3",
]
LENTS = {"PWN@ab3"}
GPU = True
NB_WORKERS = os.cpu_count()
THREADS = 1                                # threads onnxruntime par processus (CPU)
SEED = 0
SORTIE = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
'''),
    ("md", r'''
## 1. Base de puzzles Lichess
'''),
    ("code", r'''
import io, urllib.request
FICHIER = f"{SORTIE}/lichess_db_puzzle.csv.zst"
COLS = ["PuzzleId", "FEN", "Moves", "Rating", "RatingDeviation", "Popularity", "NbPlays", "Themes"]
try:
    import zstandard
    if not os.path.exists(FICHIER):
        urllib.request.urlretrieve("https://database.lichess.org/lichess_db_puzzle.csv.zst", FICHIER)
    with open(FICHIER, "rb") as fh:
        texte = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
        puzzles = pd.read_csv(texte, usecols=COLS)
except Exception as e:
    print("database.lichess.org indisponible :", e, "-> Hugging Face")
    from datasets import load_dataset
    puzzles = load_dataset("Lichess/chess-puzzles", split="train").to_pandas()[COLS]
print(f"{len(puzzles):,} puzzles")
'''),
    ("code", r'''
# puzzles fiables : classement stable, bien notés, beaucoup joués
ok = puzzles[(puzzles.RatingDeviation < 90) & (puzzles.Popularity > 80) & (puzzles.NbPlays > 1000)].copy()
ok["tranche"] = pd.cut(ok.Rating, TRANCHES, right=False)
ok = ok.dropna(subset=["tranche"])
ok = ok.sample(frac=1, random_state=SEED)  # mélange, puis les PAR_TRANCHE premiers de chaque tranche
echant = ok[ok.groupby("tranche", observed=True).cumcount() < PAR_TRANCHE].sort_values("Rating").reset_index(drop=True)
# sous-échantillon pour les joueurs lents : les PAR_TRANCHE_LENT premiers de chaque tranche
echant["lent"] = echant.groupby("tranche", observed=True).cumcount() < PAR_TRANCHE_LENT
echant.to_csv(f"{SORTIE}/bench2_puzzles_echantillon.csv", index=False)
print(len(echant), "puzzles,", echant.lent.sum(), "pour les joueurs lents")
echant.groupby("tranche", observed=True).size()
'''),
    ("md", r'''
## 2. Résolution (en parallèle, avec reprise)
'''),
    ("code", r'''
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm

RESULTATS = f"{SORTIE}/bench2_resultats_bruts.csv"
deja = pd.read_csv(RESULTATS) if os.path.exists(RESULTATS) else pd.DataFrame(columns=["joueur", "PuzzleId"])
fait = set(zip(deja.joueur, deja.PuzzleId))

for joueur in JOUEURS:
    lot = echant[echant.lent] if joueur in LENTS else echant
    taches = [(joueur, r.PuzzleId, r.FEN, r.Moves) for r in lot.itertuples() if (joueur, r.PuzzleId) not in fait]
    if not taches:
        print(joueur, ": déjà fait"); continue
    t0 = time.time()
    with ProcessPoolExecutor(NB_WORKERS, mp_context=mp.get_context("spawn"),
                             initializer=ce.init_worker, initargs=(GPU, THREADS)) as ex:
        tampon = []
        for r in tqdm(ex.map(ce.tache_puzzle, taches, chunksize=2), total=len(taches), desc=joueur):
            tampon.append(r)
            if len(tampon) >= 50:
                pd.DataFrame(tampon).to_csv(RESULTATS, mode="a", header=not os.path.exists(RESULTATS), index=False)
                tampon = []
        if tampon:
            pd.DataFrame(tampon).to_csv(RESULTATS, mode="a", header=not os.path.exists(RESULTATS), index=False)
    print(f"{joueur} : {len(taches)} puzzles en {(time.time() - t0) / 60:.1f} min")
'''),
    ("md", r'''
## 3. Résultats
'''),
    ("code", r'''
from scipy.optimize import minimize_scalar

res = pd.read_csv(RESULTATS).drop_duplicates(["joueur", "PuzzleId"]).merge(echant, on="PuzzleId")
res["resolu"] = res["resolu"].astype(bool); res["premier_coup"] = res["premier_coup"].astype(bool)

def elo_puzzle(ratings, ok):
    """classement E tel que P(résolu) = 1 / (1 + 10^((R - E) / 400)) colle le mieux aux résultats"""
    r, y = np.asarray(ratings, float), np.asarray(ok, float)
    def nll(e):
        p = np.clip(1 / (1 + 10 ** ((r - e) / 400)), 1e-9, 1 - 1e-9)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return minimize_scalar(nll, bounds=(-500, 4000), method="bounded").x

def resume(g):
    return pd.Series({
        "puzzles": len(g),
        "résolus (%)": 100 * g.resolu.mean(),
        "premier coup (%)": 100 * g.premier_coup.mean(),
        "Elo puzzle": elo_puzzle(g.Rating, g.resolu),
        "s / coup": g.secondes.sum() / g.coups_joues.sum(),
    })

tableau = res.groupby("joueur").apply(resume).loc[[j for j in JOUEURS if j in set(res.joueur)]]
# même comparaison sur le sous-échantillon commun à tous les joueurs (utile pour PWN@ab3)
tableau["Elo puzzle (sous-éch. commun)"] = res[res.lent].groupby("joueur").apply(lambda g: elo_puzzle(g.Rating, g.resolu))
tableau.to_csv(f"{SORTIE}/bench2_tableau.csv")
tableau.round(2)
'''),
    ("code", r'''
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
par_tranche = res.groupby(["joueur", "tranche"], observed=True).resolu.mean().unstack(0) * 100
par_tranche.index = [str(i.left) if hasattr(i, "left") else str(i).split(",")[0].strip("[") for i in par_tranche.index]
par_tranche.to_csv(f"{SORTIE}/bench2_par_difficulte.csv", index_label="classement puzzle")
par_tranche[[j for j in JOUEURS if j in par_tranche]].plot(ax=axes[0], marker="o")
axes[0].set(xlabel="classement du puzzle", ylabel="résolus (%)", title="Taux de résolution par difficulté")
axes[0].grid(alpha=.3)
tableau["Elo puzzle"].plot.barh(ax=axes[1], color="#58a", title="Elo puzzle estimé")
for i, v in enumerate(tableau["Elo puzzle"]):
    axes[1].text(v, i, f" {v:.0f}", va="center")
plt.tight_layout(); plt.savefig(f"{SORTIE}/bench2_puzzles.png", dpi=120); plt.show()
'''),
    ("code", r'''
THEMES = ["mateIn1", "mateIn2", "mateIn3", "fork", "pin", "skewer", "discoveredAttack", "hangingPiece",
          "sacrifice", "deflection", "attraction", "kingsideAttack", "defensiveMove", "quietMove",
          "opening", "middlegame", "endgame", "pawnEndgame", "rookEndgame", "short", "long", "veryLong"]
lignes = []
for t in THEMES:
    g = res[res.Themes.str.split().apply(lambda l: t in l)]
    if len(g):
        s = g.groupby("joueur").resolu.mean() * 100
        s["n (par joueur)"] = g.groupby("joueur").size().max()
        s.name = t
        lignes.append(s)
themes = pd.DataFrame(lignes)[[j for j in JOUEURS if j in set(res.joueur)] + ["n (par joueur)"]]
themes.to_csv(f"{SORTIE}/bench2_themes.csv")
themes.style.format("{:.0f}").background_gradient(axis=1, cmap="RdYlGn", subset=themes.columns[:-1])
'''),
    ("md", r'''
### Tests de McNemar
Sur les mêmes puzzles : combien A résout que B rate (et inversement), et est-ce significatif ?
'''),
    ("code", r'''
from scipy.stats import binomtest
PAIRES = [("PAWN@arbre2", "PAWN_BIG@arbre2"), ("PAWN_BIG@arbre2", "CONTROLE@arbre2"),
          ("CONTROLE@arbre2", "SPAWN@arbre2"), ("SPAWN@arbre2", "PWN@ab2"), ("PWN@ab2", "PWN@ab3"),
          ("CONTROLE@arbre1", "SPAWN@arbre1")]
piv = res.pivot_table(index="PuzzleId", columns="joueur", values="resolu")
lignes = []
for a, b in PAIRES:
    if a in piv and b in piv:
        x = piv[[a, b]].dropna().astype(bool)
        seul_b, seul_a = int((~x[a] & x[b]).sum()), int((x[a] & ~x[b]).sum())
        p = binomtest(seul_b, seul_a + seul_b).pvalue if seul_a + seul_b else 1.0
        lignes.append({"A": a, "B": b, "puzzles": len(x), "B résout, pas A": seul_b, "A résout, pas B": seul_a,
                       "p-value": p, "significatif (5 %)": p < 0.05})
mcnemar = pd.DataFrame(lignes); mcnemar.to_csv(f"{SORTIE}/bench2_mcnemar.csv", index=False)
mcnemar
'''),
    ("md", r'''
### Comment lire les résultats
- L'**Elo puzzle** est le chiffre le plus parlant : il est sur l'échelle des puzzles Lichess (pas l'Elo de partie !).
- **CONTROLE vs SPAWN** à profondeur égale isole l'apport de Soluce ; **SPAWN@arbre2 vs PWN@ab2** isole l'apport de la recherche alpha-bêta/quiescence.
- Les anciens chiffres (300 puzzles) ne sont pas directement comparables : ici la difficulté est répartie uniformément de 400 à 2800.
'''),
]

# =====================================================================================
# 03 — tournoi + Elo
# =====================================================================================

NB3 = [
    ("md", r'''
# ♟️ Benchmark 3 — Tournoi et classement Elo

Des vraies parties :
1. **tournoi toutes-rondes** entre les modèles (même recherche profondeur 2 pour les réseaux, + PWN alpha-bêta) ;
2. **matchs contre Stockfish 17 bridé** (`UCI_LimitStrength` à 1320 / 1600 / 1900 Elo, et `Skill Level 0` comme dans `stockfish-VS-PWN-prof2/`).

Chaque paire joue chaque **ouverture** de la liste deux fois (une fois avec chaque couleur) : les moteurs sont déterministes,
donc les ouvertures imposées donnent de la variété et rendent le match équitable.

On en déduit un **Elo** par maximum de vraisemblance, **ancré sur les niveaux Stockfish** (1320/1600/1900),
avec un intervalle de confiance à 95 % par bootstrap. Toutes les parties sont exportées en PGN.

⚠️ L'Elo « UCI_Elo » de Stockfish est calibré à cadence longue contre des moteurs (échelle CCRL) : c'est un ordre de grandeur,
pas un Elo Lichess ou FIDE. Les **écarts entre tes modèles** sont eux fiables.

⚙️ Réglages Kaggle : **Internet ON**, **GPU T4** conseillé. Durée : ~2–4 h avec les réglages par défaut (reprise automatique si coupure).
'''),
    ("code", INSTALL_ECHECS),
    ("code", MODULE_ECHECS),
    ("code", IMPORT_ECHECS),
    ("code", r'''
SF = ce.installer_stockfish(os.path.join(os.getcwd(), "stockfish"))
print("Stockfish :", SF)
'''),
    ("code", r'''
# ---- réglages ----
MODELES = ["PAWN@arbre2", "PAWN_BIG@arbre2", "CONTROLE@arbre2", "SPAWN@arbre2", "PWN@ab2"]
# "PWN@ab3" est plus fort mais ~5x plus lent : à ajouter si tu as le temps
STOCKFISH = {"SF@elo1320": 1320, "SF@elo1600": 1600, "SF@elo1900": 1900}   # ancres Elo
STOCKFISH_LIBRES = ["SF@skill0"]                                             # joués mais pas ancrés
SF_TEMPS = 0.1          # secondes par coup pour Stockfish
N_OUVERTURES = 8        # ouvertures par paire (x2 couleurs)
MAX_PLIES = 300         # au-delà : nulle
GPU = True
NB_WORKERS = os.cpu_count()
THREADS = 1
BUDGET_H = 9            # on arrête de lancer des parties après ce temps (limite Kaggle : 12 h)
SEED = 0
SORTIE = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

OUVERTURES = {
    "Italienne": "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5",
    "Espagnole": "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6",
    "Sicilienne Najdorf": "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6",
    "Française": "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6",
    "Caro-Kann": "e2e4 c7c6 d2d4 d7d5 e4e5 c8f5",
    "Gambit dame refusé": "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6",
    "Slave": "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6",
    "Est-indienne": "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6",
    "Nimzo-indienne": "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4",
    "Anglaise": "c2c4 e7e5 b1c3 g8f6 g1f3 b8c6",
    "Scandinave": "e2e4 d7d5 e4d5 d8d5 b1c3 d5a5",
    "Londres": "d2d4 d7d5 g1f3 g8f6 c1f4 c7c5",
    "Pirc": "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6",
    "Hollandaise": "d2d4 f7f5 g1f3 g8f6 g2g3 e7e6",
    "Écossaise": "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 g8f6",
    "Réti": "g1f3 d7d5 g2g3 g8f6 f1g2 e7e6",
}
for nom, coups in OUVERTURES.items():  # vérifie que les ouvertures sont légales
    b = chess.Board()
    for u in coups.split():
        assert chess.Move.from_uci(u) in b.legal_moves, (nom, u)
        b.push_uci(u)
OUV = dict(list(OUVERTURES.items())[:N_OUVERTURES])
'''),
    ("code", r'''
from itertools import combinations
paires = list(combinations(MODELES, 2)) + [(m, s) for m in MODELES for s in list(STOCKFISH) + STOCKFISH_LIBRES]
taches = []
for a, b in paires:
    for nom, coups in OUV.items():
        taches.append((a, b, coups.split(), MAX_PLIES, nom))
        taches.append((b, a, coups.split(), MAX_PLIES, nom))
random.Random(SEED).shuffle(taches)  # si ça coupe, les résultats partiels restent équilibrés
print(len(paires), "paires,", len(taches), "parties")
'''),
    ("md", r'''
## Parties (en parallèle, avec reprise)
'''),
    ("code", r'''
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm

PARTIES = f"{SORTIE}/bench3_parties.jsonl"
fait = set()
if os.path.exists(PARTIES):
    for l in open(PARTIES):
        r = json.loads(l); fait.add((r["blancs"], r["noirs"], r["ouverture"]))
reste = [t for t in taches if (t[0], t[1], t[4]) not in fait]
print(len(fait), "déjà jouées,", len(reste), "à jouer")

t0 = time.time()
ex = ProcessPoolExecutor(NB_WORKERS, mp_context=mp.get_context("spawn"),
                         initializer=ce.init_worker, initargs=(GPU, THREADS, SF, SF_TEMPS))
futurs = [ex.submit(ce.tache_partie, t) for t in reste]
with open(PARTIES, "a") as f:
    for fu in tqdm(as_completed(futurs), total=len(futurs)):
        try:
            f.write(json.dumps(fu.result()) + "\n"); f.flush()
        except Exception as e:
            print("erreur :", e)
        if time.time() - t0 > BUDGET_H * 3600:
            print("budget temps atteint, on s'arrête là")
            for x in futurs:
                x.cancel()
            break
ex.shutdown(wait=False, cancel_futures=True)
'''),
    ("md", r'''
## Résultats
'''),
    ("code", r'''
parties = pd.DataFrame([json.loads(l) for l in open(PARTIES)])
parties = parties.drop_duplicates(["blancs", "noirs", "ouverture"])
parties["score_blancs"] = parties.resultat.map({"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5})
with open(f"{SORTIE}/bench3_parties.pgn", "w") as f:
    f.write("\n\n".join(parties.pgn))
print(len(parties), "parties |", parties.fin.value_counts().to_dict())

joueurs = MODELES + list(STOCKFISH) + STOCKFISH_LIBRES
croise = pd.DataFrame(np.nan, index=joueurs, columns=joueurs)
for a in joueurs:
    for b in joueurs:
        g1 = parties[(parties.blancs == a) & (parties.noirs == b)].score_blancs
        g2 = 1 - parties[(parties.blancs == b) & (parties.noirs == a)].score_blancs
        s = pd.concat([g1, g2])
        if len(s):
            croise.loc[a, b] = s.sum()
print("score de la ligne contre la colonne (sur", 2 * len(OUV), "parties)")
croise.loc[MODELES].dropna(axis=1, how="all")
'''),
    ("code", r'''
from scipy.optimize import minimize

def elo_mle(df, ancres, joueurs, prior_sd=1000):
    """Bradley-Terry / Elo par maximum de vraisemblance (nulle = ½ victoire) ; joueurs ancrés à leur Elo"""
    libres = [j for j in joueurs if j not in ancres]
    idx = {j: i for i, j in enumerate(libres)}
    w, n = df.blancs.to_numpy(), df.noirs.to_numpy()
    s = df.score_blancs.to_numpy()
    moy = np.mean(list(ancres.values()))
    def notes(x):
        return np.array([ancres[j] if j in ancres else x[idx[j]] for j in w]), \
               np.array([ancres[j] if j in ancres else x[idx[j]] for j in n])
    def nll(x):
        rw, rn = notes(x)
        p = np.clip(1 / (1 + 10 ** ((rn - rw) / 400)), 1e-9, 1 - 1e-9)
        return -np.sum(s * np.log(p) + (1 - s) * np.log(1 - p)) + np.sum((x - moy) ** 2) / (2 * prior_sd ** 2)
    x = minimize(nll, np.full(len(libres), moy), method="L-BFGS-B").x
    return {**{j: x[idx[j]] for j in libres}, **ancres}

elo = elo_mle(parties, STOCKFISH, joueurs)
rng = np.random.default_rng(SEED)
boot = []
for _ in range(200):
    boot.append(elo_mle(parties.sample(len(parties), replace=True, random_state=int(rng.integers(1e9))), STOCKFISH, joueurs))
boot = pd.DataFrame(boot)

classement = pd.DataFrame({
    "Elo": pd.Series(elo),
    "IC 95 % bas": boot.quantile(0.025),
    "IC 95 % haut": boot.quantile(0.975),
})
score = {}
for j in joueurs:
    s = pd.concat([parties[parties.blancs == j].score_blancs, 1 - parties[parties.noirs == j].score_blancs])
    score[j] = (s.mean() * 100, len(s))
classement["score (%)"] = [score[j][0] for j in classement.index]
classement["parties"] = [score[j][1] for j in classement.index]
vit = pd.concat([parties.groupby("blancs").s_par_coup_blancs.mean(), parties.groupby("noirs").s_par_coup_noirs.mean()], axis=1).mean(axis=1)
classement["s / coup"] = vit
classement = classement.sort_values("Elo", ascending=False)
classement.to_csv(f"{SORTIE}/bench3_classement.csv")
classement.round(1)
'''),
    ("code", r'''
c = classement.loc[[j for j in classement.index if j in MODELES or j in STOCKFISH_LIBRES]]
fig, ax = plt.subplots(figsize=(9, 4))
ax.barh(c.index, c.Elo, xerr=[c.Elo - c["IC 95 % bas"], c["IC 95 % haut"] - c.Elo], color="#58a", capsize=4)
for niveau, e in STOCKFISH.items():
    ax.axvline(e, ls="--", color="gray", lw=1); ax.text(e, len(c) - 0.4, niveau.replace("SF@", "SF "), fontsize=8, ha="center")
ax.invert_yaxis(); ax.set_xlabel("Elo (ancré sur Stockfish UCI_Elo)"); ax.set_title("Classement Elo des modèles")
plt.tight_layout(); plt.savefig(f"{SORTIE}/bench3_elo.png", dpi=120); plt.show()
'''),
    ("md", r'''
### Comment lire les résultats
- Avec 16 parties par paire l'IC est large (±80–150 Elo) : pour trancher entre deux modèles proches, augmente `N_OUVERTURES`.
- Si un modèle fait 0 % ou 100 % contre un niveau Stockfish, son Elo repose sur les autres niveaux : ajoute un niveau plus fort/faible (`SF@elo2200`, `SF@elo1450`…).
- `fin` = `MAX_PLIES` souvent → les moteurs tournent en rond en finale (signe classique d'une éval sans notion de progrès).
'''),
]

# =====================================================================================
# 04 — MIND (embeddings)
# =====================================================================================

NB4 = [
    ("md", r'''
# 🧠 Benchmark 4 — MIND (embeddings de phrases en français)

**MIND v2** : `Geotrend/distilbert-base-en-fr-cased` (68,8 M paramètres) + mean pooling + Dense 768→256 (tanh) + normalisation,
entraîné avec MultipleNegativesRankingLoss sur 81 829 paires (PAWS-X fr + XNLI fr).

On le compare à des modèles de référence de taille proche :

| Modèle | Paramètres | Dim | Remarque |
|---|---|---|---|
| **MIND v2** | 69 M | 256 | le tien |
| Geotrend/distilbert-base-en-fr-cased | 69 M | 768 | le modèle de départ, **sans** fine-tuning → mesure ce que ton entraînement apporte |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 118 M | 384 | classique multilingue |
| sentence-transformers/distiluse-base-multilingual-cased-v2 | 135 M | 512 | DistilBERT multilingue, architecture la plus proche |
| intfloat/multilingual-e5-small | 118 M | 384 | fort pour sa taille |
| dangvantuan/sentence-camembert-base | 110 M | 768 | spécialiste du français |

**Partie A** — tests rapides maison : STSb-fr, STS croisé fr↔en, vitesse.
**Partie B** — **MTEB français** (`MTEB(fra, v1)`, 25 tâches : classification, clustering, paires, reranking, recherche, STS, résumé),
le benchmark de référence pour les embeddings français.

⚠️ `PawsXPairClassification` (et XNLI) sont **dans le domaine d'entraînement** de MIND : c'est signalé dans les résultats.

⚙️ Réglages Kaggle : **Internet ON**, **GPU T4**. Durée : partie A ~10 min, partie B ~2–4 h (mode `LEGER` ~30 min).
'''),
    ("code", r'''
!pip install -q -U sentence-transformers mteb
'''),
    ("code", r'''
import os, glob, time, gc, urllib.request
import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch
from sentence_transformers import SentenceTransformer
import mteb
from scipy.stats import spearmanr, pearsonr

# jeton Hugging Face (facultatif) : sans lui, HF limite le nombre de requêtes (erreur 429)
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN chargé depuis les secrets Kaggle")
except Exception:
    print("pas de HF_TOKEN : risque d'erreurs 429 (trop de requêtes) sur Hugging Face")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SORTIE = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
print("device :", DEVICE, "| mteb", mteb.__version__)

# ---- réglages ----
NOM_MIND = "MIND v2"   # pour tester une autre version : "FineMIND", "FineMIND-2"... + MIND_DOSSIER ci-dessous
MIND_DOSSIER = None    # ex : "/kaggle/input/<notebook-05>/finemind" (dossier sentence-transformers)
LEGER = False          # True : quelques tâches MTEB seulement (~30 min)
BASELINES = [
    "Geotrend/distilbert-base-en-fr-cased",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/distiluse-base-multilingual-cased-v2",
    "intfloat/multilingual-e5-small",
    "dangvantuan/sentence-camembert-base",
    # "intfloat/multilingual-e5-base", "BAAI/bge-m3",   # plus gros, si tu veux viser haut
]
PREFIXES = {"intfloat/multilingual-e5-small": "query: ", "intfloat/multilingual-e5-base": "query: "}

# Partie B (MTEB) — seul MIND est calculé. Les modèles de comparaison sont repris du
# leaderboard officiel MTEB (scores déjà publiés sur les 25 tâches françaises, téléchargés en ~2 min)
OFFICIELS = [
    "sentence-transformers/all-MiniLM-L6-v2",                        #  23 M, anglais seulement (plancher)
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",   # 118 M
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",   # 278 M
    "intfloat/multilingual-e5-base",                                 # 278 M
    "sentence-transformers/LaBSE",                                   # 471 M
]
# modèles sans score officiel à calculer aussi (~1 h chacun), par ex. le point de départ de MIND :
A_CALCULER = [
    # "Geotrend/distilbert-base-en-fr-cased",
    # "dangvantuan/sentence-camembert-base",
]
FP16 = True            # demi-précision sur GPU : ~2x plus rapide, écart de score négligeable
'''),
    ("md", r'''
## Téléchargement de MIND
Cherché d'abord dans `/kaggle/input` (si tu as ajouté le dossier `mind-v2-final` comme dataset), sinon téléchargé depuis GitHub.
'''),
    ("code", r'''
FICHIERS_MIND = ["README.md", "config.json", "config_sentence_transformers.json", "modules.json",
                 "sentence_bert_config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors",
                 "1_Pooling/config.json", "2_Dense/config.json", "2_Dense/model.safetensors", "3_Normalize/config.json"]
LFS = "https://media.githubusercontent.com/media/LiamLitle/ia-de-liam/main/MIND/mind-v2-final/"

trouve = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/modules.json", recursive=True)
          if os.path.exists(os.path.join(os.path.dirname(p), "2_Dense"))]
if MIND_DOSSIER:
    MIND = MIND_DOSSIER
elif trouve:
    MIND = trouve[0]
else:
    MIND = f"{SORTIE}/mind-v2-final"
    for f in FICHIERS_MIND:
        dest = os.path.join(MIND, f)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if not os.path.exists(dest):
            urllib.request.urlretrieve(LFS + f, dest)
print("MIND :", MIND)
mind = SentenceTransformer(MIND, device=DEVICE)
mind.model_card_data.model_name = "LiamLitle/" + NOM_MIND.lower().replace(" ", "-")  # clé du cache MTEB : un nom par version !
print(mind)
e = mind.encode(["Le chat dort sur le canapé.", "Un chat fait la sieste sur le sofa.", "La bourse a chuté hier."])
print(np.round(e @ e.T, 3))
'''),
    ("md", r'''
## Partie A — tests rapides
### A1. STSb-fr (test) et STS croisé français ↔ anglais
`stsb_multi_mt` contient les **mêmes paires traduites** : on peut comparer une phrase française à une phrase anglaise.
Le test STSb-fr n'a pas servi à l'entraînement de MIND (seul le dev a servi à choisir le checkpoint) → le README annonce **68,2** (Spearman ×100).
'''),
    ("code", r'''
from huggingface_hub import hf_hub_download
def stsb(langue):
    return pd.read_parquet(hf_hub_download("PhilipMay/stsb_multi_mt", f"{langue}/test-00000-of-00001.parquet", repo_type="dataset"))
fr, en = stsb("fr"), stsb("en")
assert len(fr) == len(en)
print(len(fr), "paires"); fr.head(3)
'''),
    ("code", r'''
def encoder(modele, phrases, prefixe=""):
    return modele.encode([prefixe + p for p in phrases], batch_size=64, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)

def sts(modele, s1, s2, gold, prefixe=""):
    a, b = encoder(modele, s1, prefixe), encoder(modele, s2, prefixe)
    return 100 * spearmanr((a * b).sum(1), gold)[0]

def vitesse(modele, phrases, prefixe=""):
    encoder(modele, phrases[:64], prefixe)
    if DEVICE == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter(); encoder(modele, phrases, prefixe)
    if DEVICE == "cuda": torch.cuda.synchronize()
    return len(phrases) / (time.perf_counter() - t0)

phrases_vitesse = (fr.sentence1.tolist() + fr.sentence2.tolist())[:2000]
lignes = []
for nom in [NOM_MIND] + BASELINES:
    m = mind if nom == NOM_MIND else SentenceTransformer(nom, device=DEVICE)
    p = PREFIXES.get(nom, "")
    lignes.append({
        "modèle": nom,
        "paramètres (M)": sum(x.numel() for x in m.parameters()) / 1e6,
        "dim": getattr(m, "get_embedding_dimension", None) and m.get_embedding_dimension() or m.get_sentence_embedding_dimension(),
        "STSb fr": sts(m, fr.sentence1, fr.sentence2, fr.similarity_score, p),
        "STSb en": sts(m, en.sentence1, en.sentence2, en.similarity_score, p),
        "STSb fr↔en": sts(m, fr.sentence1, en.sentence2, fr.similarity_score, p),
        "phrases/s": vitesse(m, phrases_vitesse, p),
    })
    if m is not mind:
        del m; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None
partie_a = pd.DataFrame(lignes).set_index("modèle")
partie_a.to_csv(f"{SORTIE}/bench4_partie_a.csv")
partie_a.round(1)
'''),
    ("md", r'''
### A2. Exemples concrets de recherche sémantique
Petit test qualitatif : pour chaque requête, la phrase la plus proche dans un mini-corpus.
'''),
    ("code", r'''
corpus = [
    "Le train pour Lyon part à 8 h de la gare de Lyon.", "Il pleut depuis trois jours sur la Bretagne.",
    "La recette demande 200 g de farine et trois œufs.", "Le PSG a gagné le match 3-1 hier soir.",
    "Mon ordinateur portable ne démarre plus.", "La Banque centrale a relevé ses taux d'intérêt.",
    "Les chats dorment environ quinze heures par jour.", "Le musée du Louvre est fermé le mardi.",
]
requetes = ["À quelle heure est le TGV ?", "Quel temps fait-il à Rennes ?", "Comment faire un gâteau ?",
            "Résultat du foot", "Mon PC est en panne", "L'inflation et la politique monétaire", "Combien dort un chat ?"]
for nom, m in [(NOM_MIND, mind)]:
    c, q = encoder(m, corpus), encoder(m, requetes)
    for r, s in zip(requetes, q @ c.T):
        print(f"{r:45s} -> {corpus[int(s.argmax())]}  ({s.max():.2f})")
'''),
    ("md", r'''
## Partie B — MTEB français
Les résultats sont mis en cache dans `mteb_cache/` : si la session coupe, relancer reprend où ça s'était arrêté.
'''),
    ("code", r'''
from mteb.cache import ResultCache
CACHE = ResultCache(f"{SORTIE}/mteb_cache")
NOMS = [t.metadata.name for t in mteb.get_benchmark("MTEB(fra, v1)").tasks]
if LEGER:
    NOMS = ["SICKFr", "STS22", "STSBenchmarkMultilingualSTS", "PawsXPairClassification", "OpusparcusPC",
            "SyntecRetrieval", "SyntecReranking", "AlloProfClusteringS2S", "MTOPIntentClassification", "AmazonReviewsClassification"]
TACHES = mteb.get_tasks(tasks=NOMS, languages=["fra"], exclusive_language_filter=True, exclude_superseded=False)
DANS_LE_DOMAINE = {"PawsXPairClassification", "XNLI"}   # vues (train) pendant l'entraînement de MIND
print(len(TACHES), "tâches :", [t.metadata.name for t in TACHES])
'''),
    ("code", r'''
def charger(nom):
    if nom == NOM_MIND:
        m = mind
    elif nom in OFFICIELS:
        return mteb.get_model(nom)  # gère les préfixes (e5 : "query: " / "passage: ")
    else:
        m = SentenceTransformer(nom, device=DEVICE)
        m.model_card_data.model_name = m.model_card_data.model_name or nom
    if FP16 and DEVICE == "cuda":
        m.half()
    return m

def evaluer(nom, taches):
    t0 = time.time()
    m = charger(nom)
    r = mteb.evaluate(m, taches, cache=CACHE, raise_error=False, encode_kwargs={"batch_size": 128})
    s = {tr.task_name: 100 * tr.get_score() for tr in r.task_results}  # tâches déjà restreintes au français
    print(f"{nom} : {len(s)} tâches en {(time.time() - t0) / 60:.1f} min", "| erreurs :", list(r.exceptions or [])[:3])
    if m is not mind:
        del m; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None
    return s

scores = {}
# 1) scores officiels (une révision du modèle peut ne pas avoir toutes les tâches : on fusionne)
try:
    officiels = mteb.load_results(models=OFFICIELS, tasks=TACHES)
    for mr in officiels.model_results:
        for tr in mr.task_results:
            try:
                v = 100 * tr.get_score()
            except Exception:
                continue
            if v == v:  # pas NaN
                scores.setdefault(mr.model_name, {}).setdefault(tr.task_name, v)
except Exception as e:
    print("scores officiels indisponibles, on calcule tout :", e)
for nom in OFFICIELS:
    manque = [t for t in TACHES if t.metadata.name not in scores.get(nom, {})]
    print(f"{nom} : {len(TACHES) - len(manque)} scores officiels, {len(manque)} à calculer")
    if manque:
        scores.setdefault(nom, {}).update(evaluer(nom, manque))

# 2) MIND et les modèles sans score officiel
for nom in [NOM_MIND] + A_CALCULER:
    scores[nom] = evaluer(nom, TACHES)
scores = {n: scores[n] for n in [NOM_MIND] + A_CALCULER + OFFICIELS if n in scores}
'''),
    ("code", r'''
res = pd.DataFrame(scores)
types = {t.metadata.name: t.metadata.type for t in TACHES}
res.insert(0, "type", [types.get(i, "?") for i in res.index])
res.index = [f"{i} ⚠️ (vu à l'entraînement)" if i in DANS_LE_DOMAINE else i for i in res.index]
res = res.sort_values(["type"])
res.to_csv(f"{SORTIE}/bench4_mteb_fr_detail.csv")
modeles = [c for c in res.columns if c != "type"]
res.style.format("{:.1f}", subset=modeles).background_gradient(axis=1, cmap="RdYlGn", subset=modeles)
'''),
    ("code", r'''
# moyenne par type de tâche, puis moyenne des types (comme le leaderboard MTEB) — sans les tâches vues à l'entraînement
hors = res[~res.index.str.contains("⚠️")]
par_type = hors.groupby("type")[modeles].mean()
par_type.loc["MOYENNE (types)"] = par_type.mean()
par_type.loc["MOYENNE (tâches)"] = hors[modeles].mean()
par_type.to_csv(f"{SORTIE}/bench4_mteb_fr_par_type.csv")
display(par_type.T.sort_values("MOYENNE (types)", ascending=False).round(1))

ax = par_type.loc["MOYENNE (types)"].sort_values().plot.barh(figsize=(8, 4), color=["#c55" if m == NOM_MIND else "#58a" for m in par_type.loc["MOYENNE (types)"].sort_values().index])
ax.set_title("MTEB français — moyenne des types de tâches (hors tâches vues)"); ax.set_xlabel("score")
plt.tight_layout(); plt.savefig(f"{SORTIE}/bench4_mteb_fr.png", dpi=120); plt.show()
'''),
    ("md", r'''
### Comment lire les résultats
- **MIND vs Geotrend (sans fine-tuning)** : ce que ton entraînement apporte vraiment.
- **MIND vs distiluse / MiniLM** : modèles de taille proche, entraînés sur beaucoup plus de données → la cible réaliste à battre.
- MIND a été entraîné sur des **paires de paraphrases/NLI** (phrases courtes) : attends-toi à de bons scores en STS / paires,
  et plus faibles en **recherche de documents** (Alloprof, BSARD, Syntec) où les textes sont longs et le couple question→passage différent.
- Dim 256 vs 384–768 : MIND est **2–3× plus compact en stockage** d'index, un vrai avantage à mettre en avant.
'''),
]


# =====================================================================================
# 05 — MIND v3a : fine-tuning de MIND v2 (négatifs difficiles + gros lots)
# =====================================================================================

NB5 = [
    ("md", r"""
# 🧠 FineMIND — fine-tuning de MIND v2

On **repart de MIND v2 tel quel** (mêmes 69 M de paramètres, même architecture, 256 dimensions) et on continue son
entraînement avec deux améliorations, **sans rien changer d'autre** pour savoir exactement ce qu'elles apportent :

1. **Négatifs difficiles** : pour chaque phrase, on donne une phrase qui lui **ressemble mais ne veut pas dire la même chose**
   - XNLI : prémisse → implication ✅ / contradiction ❌ (« Le chat dort » vs « Le chat ne dort pas »)
   - PAWS-X : paraphrase ✅ / mêmes mots dans un autre ordre avec un autre sens ❌
2. **Gros lots** (512 au lieu de 64) grâce à `CachedMultipleNegativesRankingLoss` : plus de comparaisons par pas, même mémoire GPU

Uniquement les parties **train** des datasets (déjà vues par MIND v2), le **dev** STSb-fr sert à garder le meilleur point de contrôle,
le **test** n'est regardé qu'à la fin.

⚙️ Kaggle : **GPU T4**, **Internet ON**. Durée : ~30–60 min. Résultat : le dossier `finemind/` (+ `finemind.zip`) dans l'Output.
Ensuite : notebook 04 avec `NOM_MIND = "FineMIND"` pour le comparer à v2 sur MTEB.
"""),
    ("code", r"""
!pip install -q -U "sentence-transformers[train]" datasets accelerate
"""),
    ("code", r"""
import os, glob, shutil, random, urllib.request
import numpy as np, pandas as pd, torch
from datasets import load_dataset, Dataset, DatasetDict
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
try:
    from sentence_transformers.sentence_transformer.training_args import BatchSamplers, MultiDatasetBatchSamplers
except ImportError:
    from sentence_transformers.training_args import BatchSamplers, MultiDatasetBatchSamplers

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SORTIE = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
print("device :", DEVICE)

# ---- réglages ----
NOM_DEPART = "MIND v2"
DOSSIER_DEPART = None       # None = MIND v2 (GitHub) ; sinon un modèle déjà affiné, ex : "/kaggle/input/finemind/finemind"
NOM_SORTIE = "finemind"   # FineMIND
N_TRIPLETS_XNLI = 150_000   # nombre max de triplets XNLI
TAILLE_LOT = 512            # lot « virtuel » (CachedMNRL le découpe en mini-lots)
MINI_LOT = 64               # ce qui passe réellement dans le GPU d'un coup
EPOQUES = 1
LR = 1e-5                   # plus bas que pour v2 (2e-5) : on affine un modèle déjà entraîné
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
"""),
    ("md", r"""
## Point de départ (MIND v2 par défaut, ou un modèle déjà affiné)
"""),
    ("code", r"""
FICHIERS_MIND = ["README.md", "config.json", "config_sentence_transformers.json", "modules.json",
                 "sentence_bert_config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors",
                 "1_Pooling/config.json", "2_Dense/config.json", "2_Dense/model.safetensors", "3_Normalize/config.json"]
LFS = "https://media.githubusercontent.com/media/LiamLitle/ia-de-liam/main/MIND/mind-v2-final/"
trouve = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/mind-v2-final/modules.json", recursive=True)]
if DOSSIER_DEPART:
    MIND_V2 = DOSSIER_DEPART
elif trouve:
    MIND_V2 = trouve[0]
else:
    MIND_V2 = f"{SORTIE}/mind-v2-final"
    for f in FICHIERS_MIND:
        dest = os.path.join(MIND_V2, f)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if not os.path.exists(dest):
            urllib.request.urlretrieve(LFS + f, dest)
modele = SentenceTransformer(MIND_V2, device=DEVICE)
print(modele)
"""),
    ("md", r"""
## Données d'entraînement (parties train uniquement)
"""),
    ("code", r"""
# XNLI fr : 0 = implication, 1 = neutre, 2 = contradiction
xnli = load_dataset("facebook/xnli", "fr", split="train").to_pandas()
ent = xnli[xnli.label == 0].drop_duplicates("premise")[["premise", "hypothesis"]].rename(columns={"hypothesis": "positive"})
con = xnli[xnli.label == 2].drop_duplicates("premise")[["premise", "hypothesis"]].rename(columns={"hypothesis": "negative"})
triplets_xnli = ent.merge(con, on="premise").rename(columns={"premise": "anchor"})
triplets_xnli = triplets_xnli[triplets_xnli.anchor.str.len() > 15]
triplets_xnli = triplets_xnli.sample(min(N_TRIPLETS_XNLI, len(triplets_xnli)), random_state=SEED)[["anchor", "positive", "negative"]]

# PAWS-X fr : 1 = paraphrase, 0 = mêmes mots mais autre sens (négatif très difficile)
paws = load_dataset("google-research-datasets/paws-x", "fr", split="train").to_pandas()
paws = paws[(paws.sentence1.str.len() > 15) & (paws.sentence2.str.len() > 15)]
pos = paws[paws.label == 1].drop_duplicates("sentence1").rename(columns={"sentence1": "anchor", "sentence2": "positive"})
neg = paws[paws.label == 0].drop_duplicates("sentence1").rename(columns={"sentence1": "anchor", "sentence2": "negative"})
triplets_paws = pos[["anchor", "positive"]].merge(neg[["anchor", "negative"]], on="anchor")
paires_paws = pos[~pos.anchor.isin(triplets_paws.anchor)][["anchor", "positive"]]

train = DatasetDict({
    "xnli_triplets": Dataset.from_pandas(triplets_xnli, preserve_index=False),
    "paws_triplets": Dataset.from_pandas(triplets_paws, preserve_index=False),
    "paws_paires": Dataset.from_pandas(paires_paws, preserve_index=False),
})
print(train)
print(triplets_xnli.iloc[0].to_dict())
"""),
    ("code", r"""
from huggingface_hub import hf_hub_download
def stsb(split):
    d = pd.read_parquet(hf_hub_download("PhilipMay/stsb_multi_mt", f"fr/{split}-00000-of-00001.parquet", repo_type="dataset"))
    return EmbeddingSimilarityEvaluator(d.sentence1.tolist(), d.sentence2.tolist(), (d.similarity_score / 5).tolist(),
                                        name=f"stsb-fr-{split}")
eval_dev, eval_test = stsb("dev"), stsb("test")
avant = {"dev": eval_dev(modele), "test": eval_test(modele)}
print(NOM_DEPART, "— STSb-fr dev :", round(100 * avant["dev"]["stsb-fr-dev_spearman_cosine"], 2),
      "| test :", round(100 * avant["test"]["stsb-fr-test_spearman_cosine"], 2))
"""),
    ("md", r"""
## Entraînement
"""),
    ("code", r"""
perte = losses.CachedMultipleNegativesRankingLoss(modele, mini_batch_size=MINI_LOT)
args = SentenceTransformerTrainingArguments(
    output_dir=f"{SORTIE}/checkpoints",
    num_train_epochs=EPOQUES,
    per_device_train_batch_size=TAILLE_LOT,
    learning_rate=LR,
    warmup_ratio=0.1,
    fp16=DEVICE == "cuda",
    batch_sampler=BatchSamplers.NO_DUPLICATES,          # pas deux fois la même phrase dans un lot
    multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
    eval_strategy="steps", eval_steps=50,
    save_strategy="steps", save_steps=50, save_total_limit=2,
    load_best_model_at_end=True, metric_for_best_model="eval_stsb-fr-dev_spearman_cosine",
    logging_steps=10, report_to="none", seed=SEED,
)
entraineur = SentenceTransformerTrainer(model=modele, args=args, train_dataset=train, loss=perte, evaluator=eval_dev)
entraineur.train()
"""),
    ("md", r"""
## Résultat et sauvegarde
"""),
    ("code", r"""
apres = {"dev": eval_dev(modele), "test": eval_test(modele)}
tab = pd.DataFrame({
    NOM_DEPART: [100 * avant["dev"]["stsb-fr-dev_spearman_cosine"], 100 * avant["test"]["stsb-fr-test_spearman_cosine"]],
    NOM_SORTIE: [100 * apres["dev"]["stsb-fr-dev_spearman_cosine"], 100 * apres["test"]["stsb-fr-test_spearman_cosine"]],
}, index=["STSb-fr dev", "STSb-fr test"]).round(2)
tab["écart"] = (tab[NOM_SORTIE] - tab[NOM_DEPART]).round(2)
print(tab)

DOSSIER = f"{SORTIE}/{NOM_SORTIE}"
modele.save(DOSSIER)
shutil.make_archive(DOSSIER, "zip", DOSSIER)
shutil.rmtree(f"{SORTIE}/checkpoints", ignore_errors=True)   # les points de contrôle prennent beaucoup de place
print("modèle sauvegardé :", DOSSIER, "et", DOSSIER + ".zip")
e = modele.encode(["Le chat dort sur le canapé.", "Un chat fait la sieste sur le sofa.", "Le chat ne dort pas sur le canapé."])
print(np.round(e @ e.T, 3))
"""),
    ("md", r"""
### Et après ?
- Le STSb-fr **test** ne doit être regardé qu'une fois : c'est le verdict. Un gain sur le **dev** seul ne suffit pas.
- Pour le vrai verdict (25 tâches) : dans le notebook 04, **Add Input → l'Output de ce notebook**, puis
  `NOM_MIND = "FineMIND"` et `MIND_DOSSIER = "/kaggle/input/<nom>/finemind"` (adapter le chemin), `LEGER = False`.
- Regarde surtout **PairClassification / STS / Reranking** : c'est là que les négatifs difficiles doivent aider.
"""),
]


# =====================================================================================
# 06 — FineMIND-2 : questions → réponses, sans oublier le reste
# =====================================================================================

NB6 = [
    ("md", r"""
# 🧠 FineMIND-2 — apprendre les questions → réponses

On repart de **FineMIND** (mêmes 69 M de paramètres) pour corriger son plus gros point faible sur MTEB français :
la **recherche de documents** (Retrieval 37,7, contre 44 à 53 pour les meilleurs).

1. **Nouvelles données** : des paires **question → paragraphe qui contient la réponse** (PIAF, FrenchQA : Wikipédia en français).
   MIND n'a jamais appris ça, il n'a vu que des paraphrases.
2. **Rappel des anciennes données** (STSb, XNLI, PAWS-X) mélangées aux nouvelles, pour qu'il **n'oublie pas** ce qu'il savait.
   FineMIND avait perdu ~2 points en regroupement de textes : c'est ce qu'on veut éviter cette fois.
3. Le meilleur point de contrôle est choisi sur **deux juges à la fois** : STSb-fr dev **et** une recherche question → paragraphe
   sur des paragraphes jamais vus à l'entraînement.

Aucun jeu de données de MTEB n'est utilisé (pas d'Alloprof, Mintaka, XPQA, BSARD, Syntec) : le notebook 04 reste honnête.

⚙️ Kaggle : **GPU T4**, **Internet ON**, **Add Input → ton dataset `finemind`**. Durée : ~1 h.
Résultat : `finemind-2/` (+ `finemind-2.zip`) dans l'Output. Ensuite : notebook 04 avec `NOM_MIND = "FineMIND-2"`.
"""),
    ("code", r"""
!pip install -q -U "sentence-transformers[train]" datasets accelerate
"""),
    ("code", r"""
import os, glob, shutil, random
import numpy as np, pandas as pd, torch
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, InformationRetrievalEvaluator, SequentialEvaluator
try:
    from sentence_transformers.sentence_transformer.training_args import BatchSamplers, MultiDatasetBatchSamplers
except ImportError:
    from sentence_transformers.training_args import BatchSamplers, MultiDatasetBatchSamplers

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SORTIE = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
print("device :", DEVICE)

# ---- réglages ----
NOM_DEPART = "FineMIND"
DOSSIER_DEPART = None        # None = cherche tout seul un dossier « finemind » dans /kaggle/input
NOM_SORTIE = "finemind-2"    # FineMIND-2
SOURCES_QR = ["CATIE-AQ/frenchQA", "AgentPublic/piaf", "etalab-ia/piaf"]   # questions → paragraphes (on prend ce qui marche)
N_QR = 100_000               # paires question → paragraphe max
N_DEV_QR = 1_000             # paragraphes mis de côté pour juger la recherche
N_RAPPEL_XNLI = 40_000       # paires XNLI « implication » (données de MIND v2)
N_RAPPEL_TRIPLETS = 20_000   # triplets XNLI avec négatif (données de FineMIND)
TAILLE_LOT = 512
MINI_LOT = 32                # plus petit qu'au 05 : les paragraphes sont plus longs que des phrases
EPOQUES = 1
LR = 1e-5
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
"""),
    ("md", r"""
## Point de départ : FineMIND
"""),
    ("code", r"""
if not DOSSIER_DEPART:
    trouve = sorted(os.path.dirname(p) for p in glob.glob("/kaggle/input/**/modules.json", recursive=True)
                    if "finemind" in p.lower() and "finemind-2" not in p.lower())
    assert trouve, "FineMIND introuvable : Add Input → ton dataset finemind (ou mets son chemin dans DOSSIER_DEPART)"
    DOSSIER_DEPART = trouve[0]
print("départ :", DOSSIER_DEPART)
modele = SentenceTransformer(DOSSIER_DEPART, device=DEVICE)
print("longueur max :", modele.max_seq_length, "tokens")
"""),
    ("md", r"""
## Nouvelles données : questions → paragraphes
"""),
    ("code", r"""
def charger_qr(nom):
    ds = None
    for options in ({}, {"revision": "refs/convert/parquet"}):
        try:
            ds = load_dataset(nom, split="train", **options)
            break
        except Exception as e:
            erreur = e
    if ds is None:
        print(f"  ✗ {nom} : {type(erreur).__name__} {str(erreur)[:150]}")
        return None
    df = ds.to_pandas()
    q = next((c for c in ["question", "query"] if c in df.columns), None)
    c = next((c for c in ["context", "passage", "paragraph", "positive"] if c in df.columns), None)
    if not q or not c:
        print(f"  ✗ {nom} : colonnes inconnues {list(df.columns)}")
        return None
    # questions « sans réponse dans le paragraphe » (style SQuAD v2) : ce ne sont pas de vraies paires
    if "answer" in df.columns:
        df = df[df.answer.notna() & (df.answer.astype(str).str.strip() != "")]
    elif "answers" in df.columns:
        df = df[df.answers.map(lambda a: len(a["text"]) > 0 if isinstance(a, dict) else True)]
    print(f"  ✓ {nom} : {len(df):,} lignes")
    return df[[q, c]].rename(columns={q: "anchor", c: "positive"}).astype(str)

qr = pd.concat([d for d in map(charger_qr, SOURCES_QR) if d is not None], ignore_index=True)
qr["anchor"], qr["positive"] = qr.anchor.str.strip(), qr.positive.str.strip()
# paragraphes trop longs coupés à 256 tokens : la réponse risquerait de disparaître
qr = qr[(qr.anchor.str.len() > 10) & (qr.positive.str.len().between(80, 1200))].drop_duplicates("anchor")
print(f"{len(qr):,} paires question → paragraphe, {qr.positive.nunique():,} paragraphes différents")

# on met de côté des PARAGRAPHES entiers (et toutes leurs questions) : jamais vus à l'entraînement
paragraphes = qr.positive.drop_duplicates().sample(frac=1, random_state=SEED)
dev_p = set(paragraphes.iloc[:N_DEV_QR])
dev_qr = qr[qr.positive.isin(dev_p)].drop_duplicates("positive")
train_qr = qr[~qr.positive.isin(dev_p)]
train_qr = train_qr.sample(min(N_QR, len(train_qr)), random_state=SEED)

corpus_txt = list(dev_p) + paragraphes.iloc[N_DEV_QR:N_DEV_QR + 4_000].tolist()   # + 4000 paragraphes pièges
corpus = {f"p{i}": t for i, t in enumerate(corpus_txt)}
id_de = {t: k for k, t in corpus.items()}
questions = {f"q{i}": q for i, q in enumerate(dev_qr.anchor)}
pertinents = {f"q{i}": {id_de[p]} for i, p in enumerate(dev_qr.positive)}
eval_qr = InformationRetrievalEvaluator(questions, corpus, pertinents, name="qr-dev",
                                        show_progress_bar=False, batch_size=64)
print("exemple :", train_qr.iloc[0].to_dict())
"""),
    ("md", r"""
## Rappel des anciennes données (pour ne pas oublier)
"""),
    ("code", r"""
def stsb_df(split):
    return pd.read_parquet(hf_hub_download("PhilipMay/stsb_multi_mt", f"fr/{split}-00000-of-00001.parquet", repo_type="dataset"))

# STSb train : les paires très proches (note ≥ 4 sur 5)
sts = stsb_df("train")
sts = sts[sts.similarity_score >= 4][["sentence1", "sentence2"]].rename(columns={"sentence1": "anchor", "sentence2": "positive"})

# XNLI : implications (comme MIND v2) + triplets avec contradiction (comme FineMIND)
xnli = load_dataset("facebook/xnli", "fr", split="train").to_pandas()
ent = xnli[xnli.label == 0].drop_duplicates("premise")[["premise", "hypothesis"]].rename(columns={"hypothesis": "positive"})
con = xnli[xnli.label == 2].drop_duplicates("premise")[["premise", "hypothesis"]].rename(columns={"hypothesis": "negative"})
ent = ent[ent.premise.str.len() > 15]
triplets = ent.merge(con, on="premise").rename(columns={"premise": "anchor"})
triplets = triplets.sample(min(N_RAPPEL_TRIPLETS, len(triplets)), random_state=SEED)[["anchor", "positive", "negative"]]
paires_xnli = ent[~ent.premise.isin(triplets.anchor)].rename(columns={"premise": "anchor"})
paires_xnli = paires_xnli.sample(min(N_RAPPEL_XNLI, len(paires_xnli)), random_state=SEED)

# PAWS-X : paraphrases
paws = load_dataset("google-research-datasets/paws-x", "fr", split="train").to_pandas()
paws = paws[(paws.label == 1) & (paws.sentence1.str.len() > 15) & (paws.sentence2.str.len() > 15)]
paws = paws.drop_duplicates("sentence1")[["sentence1", "sentence2"]].rename(columns={"sentence1": "anchor", "sentence2": "positive"})

train = DatasetDict({
    "questions_reponses": Dataset.from_pandas(train_qr, preserve_index=False),
    "stsb": Dataset.from_pandas(sts, preserve_index=False),
    "xnli_paires": Dataset.from_pandas(paires_xnli, preserve_index=False),
    "xnli_triplets": Dataset.from_pandas(triplets, preserve_index=False),
    "paws_paires": Dataset.from_pandas(paws, preserve_index=False),
})
total = sum(len(d) for d in train.values())
for k, d in train.items():
    print(f"{k:20s} {len(d):>7,}  ({100 * len(d) / total:.0f} %)")
"""),
    ("code", r"""
def eval_sts(split):
    d = stsb_df(split)
    return EmbeddingSimilarityEvaluator(d.sentence1.tolist(), d.sentence2.tolist(), (d.similarity_score / 5).tolist(),
                                        name=f"stsb-fr-{split}")
eval_dev, eval_test = eval_sts("dev"), eval_sts("test")
# juge pendant l'entraînement : moyenne de STSb dev (Spearman) et de la recherche (nDCG@10)
juge = SequentialEvaluator([eval_dev, eval_qr], main_score_function=lambda s: float(np.mean(s)))

def bilan(m):
    r = {**eval_dev(m), **eval_test(m), **eval_qr(m)}
    return {"STSb-fr dev": 100 * r["stsb-fr-dev_spearman_cosine"],
            "STSb-fr test": 100 * r["stsb-fr-test_spearman_cosine"],
            "Q→R nDCG@10": 100 * r["qr-dev_cosine_ndcg@10"],
            "Q→R trouvé en 1ᵉʳ": 100 * r["qr-dev_cosine_accuracy@1"]}
avant = bilan(modele)
print(NOM_DEPART, {k: round(v, 2) for k, v in avant.items()})
"""),
    ("md", r"""
## Entraînement
"""),
    ("code", r"""
perte = losses.CachedMultipleNegativesRankingLoss(modele, mini_batch_size=MINI_LOT)
args = SentenceTransformerTrainingArguments(
    output_dir=f"{SORTIE}/checkpoints",
    num_train_epochs=EPOQUES,
    per_device_train_batch_size=TAILLE_LOT,
    learning_rate=LR,
    warmup_ratio=0.1,
    fp16=DEVICE == "cuda",
    batch_sampler=BatchSamplers.NO_DUPLICATES,          # pas deux fois le même paragraphe dans un lot
    multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
    eval_strategy="steps", eval_steps=0.1,              # 10 évaluations réparties sur tout l'entraînement
    save_strategy="steps", save_steps=0.1, save_total_limit=2,
    load_best_model_at_end=True, metric_for_best_model="eval_sequential_score",
    logging_steps=10, report_to="none", seed=SEED,
)
entraineur = SentenceTransformerTrainer(model=modele, args=args, train_dataset=train, loss=perte, evaluator=juge)
entraineur.train()
"""),
    ("md", r"""
## Résultat et sauvegarde
"""),
    ("code", r"""
apres = bilan(modele)
tab = pd.DataFrame({NOM_DEPART: avant, NOM_SORTIE: apres}).round(2)
tab["écart"] = (tab[NOM_SORTIE] - tab[NOM_DEPART]).round(2)
print(tab)

DOSSIER = f"{SORTIE}/{NOM_SORTIE}"
modele.save(DOSSIER)
shutil.make_archive(DOSSIER, "zip", DOSSIER)
shutil.rmtree(f"{SORTIE}/checkpoints", ignore_errors=True)
print("modèle sauvegardé :", DOSSIER, "et", DOSSIER + ".zip")

q = modele.encode(["Comment faire un gâteau au chocolat ?"])
p = modele.encode(["Faire fondre le chocolat avec le beurre, ajouter les œufs, le sucre et la farine, puis cuire 25 minutes à 180 °C.",
                   "Le chocolat est produit à partir des fèves de cacao, cultivées surtout en Afrique de l'Ouest.",
                   "La bourse de Paris a fortement chuté ce matin."])
print("question → recette / histoire du chocolat / bourse :", np.round(q @ p.T, 3))
e = modele.encode(["Le chat dort sur le canapé.", "Un chat fait la sieste sur le sofa.", "Le chat ne dort pas sur le canapé."])
print(np.round(e @ e.T, 3))
"""),
    ("md", r"""
### Et après ?
- **Q→R** doit monter nettement (c'est le but), et **STSb-fr** doit rester à peu près au niveau de FineMIND.
- Le vrai verdict : notebook 04, **Add Input → l'Output de ce notebook** (ou un dataset `finemind-2`), puis
  `NOM_MIND = "FineMIND-2"`, `MIND_DOSSIER = "/kaggle/input/<chemin>/finemind-2"`, `LEGER = False`.
- À regarder : **Retrieval** et **Reranking** doivent monter, **Clustering** ne doit pas redescendre.
"""),
]


# =====================================================================================
# 07 — PWN@ab3 contre une échelle de Stockfish
# =====================================================================================

NB7 = [
    ("md", r'''
# ♟️ PWN3 vs Stockfish — jusqu'où monte PWN@ab3 ?

**PWN@ab3** (poids SPAWN + alpha-bêta profondeur 3 + quiescence) affronte **Stockfish 17 à plusieurs niveaux**,
du plus faible au plus fort. Chaque niveau joue les mêmes **ouvertures**, une fois avec chaque couleur.

On obtient :
- le **score contre chaque niveau** (la courbe doit descendre de ~100 % à ~0 %) ;
- l'**Elo en partie** de PWN@ab3 (maximum de vraisemblance, ancré sur les `UCI_Elo` de Stockfish) avec un intervalle à 95 % ;
- toutes les parties en **PGN** + **toutes les positions** (FEN après chaque coup) en CSV.

Pendant que ça tourne : les **échiquiers des parties en cours en direct**, le **tableau des scores** par niveau,
une ligne par partie terminée (qui a gagné), et un **bilan dès qu'un niveau est fini**.
Les fichiers de chaque partie sont écrits au fur et à mesure dans `pwn3_parties/stockfish_eloXXXX/`.

Repères : PWN@ab2 a fait **1904** en parties (notebook 03), PWN@ab3 **2345** aux puzzles (notebook 02).

⚠️ L'`UCI_Elo` de Stockfish est calibré contre des moteurs (échelle CCRL) : c'est un ordre de grandeur, pas un Elo Lichess ou FIDE.

⚙️ Kaggle : **Internet ON**. L'alpha-bêta tourne sur **CPU**, le GPU n'est pas nécessaire.
Durée : ~4–6 h (ab3 réfléchit ~5–15 s par coup) → lance-le en **Save & Run All (Commit)**.
Arrêt propre après `BUDGET_H` heures : les parties jouées sont gardées et analysées.
'''),
    ("code", INSTALL_ECHECS),
    ("code", MODULE_ECHECS),
    ("code", IMPORT_ECHECS),
    ("code", r'''
SF = ce.installer_stockfish(os.path.join(os.getcwd(), "stockfish"))
print("Stockfish :", SF)
'''),
    ("code", r'''
# ---- réglages ----
JOUEUR = "PWN@ab3"
NIVEAUX = [1500, 1800, 2100, 2400, 2700]   # UCI_Elo de Stockfish (de 1320 à 3190)
SF_TEMPS = 0.1          # secondes par coup pour Stockfish
N_OUVERTURES = 8        # parties par niveau = 2 x N_OUVERTURES
MAX_PLIES = 200         # au-delà : nulle (évite les finales interminables)
NB_WORKERS = os.cpu_count()
BUDGET_H = 11           # on arrête de lancer des parties après ce temps (limite Kaggle : 12 h)
SEED = 0
SORTIE = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

OUVERTURES = {
    "Italienne": "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5",
    "Espagnole": "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6",
    "Sicilienne Najdorf": "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6",
    "Française": "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6",
    "Caro-Kann": "e2e4 c7c6 d2d4 d7d5 e4e5 c8f5",
    "Gambit dame refusé": "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6",
    "Slave": "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6",
    "Est-indienne": "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6",
    "Nimzo-indienne": "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4",
    "Anglaise": "c2c4 e7e5 b1c3 g8f6 g1f3 b8c6",
    "Scandinave": "e2e4 d7d5 e4d5 d8d5 b1c3 d5a5",
    "Londres": "d2d4 d7d5 g1f3 g8f6 c1f4 c7c5",
    "Pirc": "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6",
    "Hollandaise": "d2d4 f7f5 g1f3 g8f6 g2g3 e7e6",
    "Écossaise": "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 g8f6",
    "Réti": "g1f3 d7d5 g2g3 g8f6 f1g2 e7e6",
}
for nom, coups in OUVERTURES.items():  # vérifie que les ouvertures sont légales
    b = chess.Board()
    for u in coups.split():
        assert chess.Move.from_uci(u) in b.legal_moves, (nom, u)
        b.push_uci(u)
OUV = dict(list(OUVERTURES.items())[:N_OUVERTURES])

STOCKFISH = {f"SF@elo{e}": e for e in NIVEAUX}
taches = []
for sf in STOCKFISH:
    for nom, coups in OUV.items():
        taches.append((JOUEUR, sf, coups.split(), MAX_PLIES, nom))
        taches.append((sf, JOUEUR, coups.split(), MAX_PLIES, nom))
random.Random(SEED).shuffle(taches)  # si ça coupe, chaque niveau a à peu près le même nombre de parties
print(len(STOCKFISH), "niveaux,", len(taches), "parties")
'''),
    ("md", r'''
## Parties (en parallèle, avec reprise)
'''),
    ("code", r'''
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

PARTIES = f"{SORTIE}/pwn3_parties.jsonl"
DOSSIER = f"{SORTIE}/pwn3_parties"          # un sous-dossier par niveau, un fichier par partie
PAR_NIVEAU = 2 * len(OUV)
EMOJI = {"victoire": "✅", "nulle": "🤝", "défaite": "❌"}

def infos(r):
    pwn_blancs = r["blancs"] == JOUEUR
    adv = r["noirs"] if pwn_blancs else r["blancs"]
    couleur = "blancs" if pwn_blancs else "noirs"
    if r["resultat"] == "1/2-1/2":
        res = "nulle"
    else:
        res = "victoire" if (r["resultat"] == "1-0") == pwn_blancs else "défaite"
    return adv, couleur, res

def enregistrer(r):
    # fichiers lisibles pour chaque partie : PGN (à ouvrir sur lichess.org/paste) + CSV de toutes les positions
    adv, couleur, res = infos(r)
    d = os.path.join(DOSSIER, adv.replace("SF@", "stockfish_"))
    os.makedirs(d, exist_ok=True)
    nom = f"{r['ouverture']}_PWN-{couleur}".replace(" ", "-")
    with open(os.path.join(d, nom + ".pgn"), "w") as f:
        f.write(r["pgn"])
    pos = pd.DataFrame(r["positions"])
    pos.insert(0, "ouverture", r["ouverture"]); pos.insert(1, "PWN_joue", couleur); pos["resultat_PWN"] = res
    pos.to_csv(os.path.join(d, nom + "_positions.csv"), index=False)

def bilan_niveau(adv, lignes):
    res = [infos(r)[2] for r in lignes]
    v, n, p = res.count("victoire"), res.count("nulle"), res.count("défaite")
    print(f"\n🏁 NIVEAU {adv} TERMINÉ : {v} victoires, {n} nulles, {p} défaites → score {100 * (v + n / 2) / len(res):.0f} %")
    for r in sorted(lignes, key=lambda r: (r["ouverture"], infos(r)[1])):
        a, c, x = infos(r)
        print(f"   {EMOJI[x]} {r['ouverture']:20s} PWN avec les {c:6s} → {x:8s} ({r['fin']}, {r['plies'] // 2} coups)")
    print()

deja = []
if os.path.exists(PARTIES):
    deja = [json.loads(l) for l in open(PARTIES)]
fait = {(r["blancs"], r["noirs"], r["ouverture"]) for r in deja}
par_niveau = {sf: [r for r in deja if infos(r)[0] == sf] for sf in STOCKFISH}
reste = [t for t in taches if (t[0], t[1], t[4]) not in fait]
print(len(fait), "déjà jouées,", len(reste), "à jouer")
for sf, lignes in par_niveau.items():
    if len(lignes) >= PAR_NIVEAU:
        bilan_niveau(sf, lignes)

# ---- affichage en direct : une carte par partie en cours + tableau des scores ----
import glob, shutil
import chess.svg
from concurrent.futures import wait, FIRST_COMPLETED
from IPython.display import display, HTML

SUIVI = "/tmp/pwn3_en_cours"
shutil.rmtree(SUIVI, ignore_errors=True); os.makedirs(SUIVI)
derniers = []     # parties finies (HTML), les plus récentes à la fin
vu = {}           # partie -> dernier demi-coup affiché (pour n'animer que les nouveaux coups)
VALEUR = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
SYMB = {chess.WHITE: {chess.PAWN: "♙", chess.KNIGHT: "♘", chess.BISHOP: "♗", chess.ROOK: "♖", chess.QUEEN: "♕"},
        chess.BLACK: {chess.PAWN: "♟", chess.KNIGHT: "♞", chess.BISHOP: "♝", chess.ROOK: "♜", chess.QUEEN: "♛"}}
DEPART = {chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2, chess.ROOK: 2, chess.QUEEN: 1}
COULEURS = {"square light": "#eeeed2", "square dark": "#769656",
            "square light lastmove": "#f6f669", "square dark lastmove": "#baca2b", "margin": "#312e2b", "coord": "#ddd"}
TAILLE = 45   # taille d'une case dans le repère SVG de chess.svg (marge 15)

STYLE = """<style>
.pw{font-family:system-ui,sans-serif;color:#222;max-width:1250px}
.pw h2{margin:4px 0 2px}.pw .aide{background:#f4f6f8;border-radius:8px;padding:8px 12px;font-size:13px;margin:8px 0}
.pw .niv{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0}
.pw .nv{border:1px solid #ddd;border-radius:10px;padding:8px 10px;width:190px;background:#fff;font-size:13px}
.pw .nv.fini{border-color:#2e7d32;background:#eef7ee}
.pw .barre{display:flex;height:10px;border-radius:5px;overflow:hidden;background:#eee;margin:5px 0}
.pw .cartes{display:flex;flex-wrap:wrap;gap:12px}
.pw .carte{border:1px solid #ccc;border-radius:12px;padding:8px;background:#fff;width:250px;box-shadow:0 1px 4px #0001}
.pw .titre{font-size:13px;color:#555;margin-bottom:6px}
.pw .joueur{display:flex;align-items:center;justify-content:space-between;padding:4px 6px;border-radius:6px;font-size:13px;white-space:nowrap}
.pw .joueur.pwn{background:#e3f0ff}.pw .joueur.sf{background:#f3f3f3}
.pw .pion{display:inline-block;width:13px;height:13px;border-radius:50%;border:1px solid #333;margin-right:6px;vertical-align:-2px}
.pw .reflechit{color:#d35400;font-weight:bold;animation:clig 1s infinite}
@keyframes clig{50%{opacity:.25}}
.pw .pris{font-size:15px;letter-spacing:-2px;color:#555;min-height:18px}
.pw .info{font-size:13px;margin-top:6px}
.pw .mat{height:8px;border-radius:4px;background:#ddd;position:relative;margin-top:4px}
.pw .fin{font-size:13px;padding:3px 0;border-bottom:1px solid #eee}
</style>"""

def case_xy(sq, retourne):
    f, r = chess.square_file(sq), chess.square_rank(sq)
    if retourne:
        return 15 + TAILLE * (7 - f), 15 + TAILLE * r
    return 15 + TAILLE * f, 15 + TAILLE * (7 - r)

def echiquier(bd, coup, retourne, animer):
    svg = chess.svg.board(bd, lastmove=coup, size=250, flipped=retourne, colors=COULEURS,
                          check=bd.king(bd.turn) if bd.is_check() else None)
    if coup and animer:   # la pièce glisse de sa case de départ à sa case d'arrivée
        xa, ya = case_xy(coup.from_square, retourne)
        xb, yb = case_xy(coup.to_square, retourne)
        cible = f'transform="translate({xb}, {yb})" />'
        anim = (f'transform="translate({xb}, {yb})"><animateTransform attributeName="transform" type="translate" '
                f'from="{xa} {ya}" to="{xb} {yb}" dur="0.7s" fill="freeze" /></use>')
        svg = svg.replace(cible, anim, 1)
    return svg

def pieces_prises(bd, couleur):
    # pièces de `couleur` qui ont disparu de l'échiquier
    return "".join(SYMB[couleur][p] * max(0, DEPART[p] - len(bd.pieces(p, couleur))) for p in DEPART)

def materiel(bd, couleur):
    return sum(v * len(bd.pieces(p, couleur)) for p, v in VALEUR.items())

def barre_joueur(nom, classe, couleur_pieces, a_son_tour, prises_adverses):
    pastille = "#fff" if couleur_pieces == chess.WHITE else "#222"
    txt_couleur = "blancs" if couleur_pieces == chess.WHITE else "noirs"
    etat = "<span class='reflechit'>⏳ joue…</span>" if a_son_tour else ""
    return (f"<div class='joueur {classe}'><span><span class='pion' style='background:{pastille}'></span>"
            f"<b>{nom}</b> <small>({txt_couleur})</small></span>{etat}</div>"
            f"<div class='pris' title='pièces capturées'>{prises_adverses}</div>")

def carte(e):
    pwn_blancs = e["blancs"] == JOUEUR
    adv = e["noirs"] if pwn_blancs else e["blancs"]
    c_pwn = chess.WHITE if pwn_blancs else chess.BLACK
    bd = chess.Board(e["fen"])
    coup = chess.Move.from_uci(e["dernier"]) if e["dernier"] else None
    cle = (e["blancs"], e["noirs"], e["ouverture"])
    animer = vu.get(cle) != e["ply"]
    vu[cle] = e["ply"]
    ecart = materiel(bd, c_pwn) - materiel(bd, not c_pwn)
    if ecart > 0:
        mat_txt, mat_coul = f"PWN mène de <b>+{ecart}</b>", "#1e88e5"
    elif ecart < 0:
        mat_txt, mat_coul = f"Stockfish mène de <b>+{-ecart}</b>", "#e53935"
    else:
        mat_txt, mat_coul = "matériel <b>égal</b>", "#999"
    pos = 50 + max(-50, min(50, ecart * 5))
    num = (e["ply"] + 1) // 2
    if coup:
        qui = "PWN" if (e["ply"] % 2 == 1) == pwn_blancs else "Stockfish"
        dernier = f"{num}{'.' if e['ply'] % 2 else '…'} <b>{e['san']}</b> ({qui})"
    else:
        dernier = "–"
    duree = (time.time() - e["debut"]) / 60
    haut = barre_joueur(f"Stockfish {adv[6:]}", "sf", not c_pwn, bd.turn != c_pwn, pieces_prises(bd, c_pwn))
    bas = barre_joueur(JOUEUR, "pwn", c_pwn, bd.turn == c_pwn, pieces_prises(bd, not c_pwn))
    return (f"<div class='carte'><div class='titre'>📖 {e['ouverture']} · coup {num} · {duree:.0f} min</div>"
            f"{haut}{echiquier(bd, coup, not pwn_blancs, animer)}{bas}"
            f"<div class='info'>Dernier coup : {dernier}</div>"
            f"<div class='info'>⚖️ {mat_txt}</div>"
            f"<div class='mat'><div style='position:absolute;left:{pos}%;top:-3px;width:4px;height:14px;background:{mat_coul}'></div>"
            f"<div style='position:absolute;left:50%;top:0;width:1px;height:8px;background:#555'></div></div></div>")

def ecran():
    fait_n = sum(len(v) for v in par_niveau.values())
    total = PAR_NIVEAU * len(STOCKFISH)
    en_cours = []
    for fich in sorted(glob.glob(f"{SUIVI}/*.json")):
        try:
            en_cours.append(json.load(open(fich)))
        except Exception:
            pass
    h = [STYLE, "<div class='pw'>",
         f"<h2>♟️ {JOUEUR} contre Stockfish 17</h2>",
         f"<div>{fait_n}/{total} parties terminées · {len(en_cours)} en cours · {(time.time() - t0) / 60:.0f} min</div>",
         f"<div class='barre' style='width:600px'><div style='width:{100 * fait_n / total:.1f}%;background:#1e88e5'></div></div>",
         "<div class='aide'>Chaque carte = une partie en cours. <b style='color:#1e88e5'>PWN</b> (ton IA) est toujours "
         "<b>en bas</b>, <b>Stockfish</b> en haut. La pastille ⚪/⚫ indique la couleur des pièces. Les cases jaunes = dernier coup. "
         "Sous chaque joueur : les pièces qu'il a capturées. Les niveaux sont joués dans un ordre mélangé exprès.</div>",
         "<div class='niv'>"]
    for sf, lignes in par_niveau.items():
        res = [infos(r)[2] for r in lignes]
        v, n, d = res.count("victoire"), res.count("nulle"), res.count("défaite")
        cours = sum(1 for e in en_cours if sf in (e["blancs"], e["noirs"]))
        sc = f"{100 * (v + n / 2) / len(res):.0f} %" if res else "–"
        fini = " fini" if len(res) == PAR_NIVEAU else ""
        w = lambda x: 100 * x / PAR_NIVEAU
        h.append(f"<div class='nv{fini}'><b>Stockfish {sf[6:]}</b>{' ✔️' if fini else ''}"
                 f"<div class='barre'><div style='width:{w(v)}%;background:#43a047'></div>"
                 f"<div style='width:{w(n)}%;background:#fbc02d'></div><div style='width:{w(d)}%;background:#e53935'></div></div>"
                 f"<small>✅ {v} · 🤝 {n} · ❌ {d} · {len(res)}/{PAR_NIVEAU}"
                 f"{f' · {cours} en cours' if cours else ''}</small><br>Score PWN : <b>{sc}</b></div>")
    h.append("</div><div class='cartes'>")
    h += [carte(e) for e in en_cours]
    h.append("</div><h3>Dernières parties terminées</h3>")
    h.append("".join(derniers[-10:][::-1]) or "<i>aucune pour l'instant (une partie dure 10–20 min)</i>")
    h.append("</div>")
    return HTML("".join(h))

def ligne_fin(r):
    adv, couleur, res = infos(r)
    coul = {"victoire": "#43a047", "nulle": "#f9a825", "défaite": "#e53935"}[res]
    return (f"<div class='fin'>{EMOJI[res]} <b style='color:{coul}'>{res.upper()}</b> de PWN ({couleur}) contre "
            f"<b>Stockfish {adv[6:]}</b> · {r['ouverture']} · {r['plies'] // 2} coups · {r['fin'].lower()}</div>")

t0 = time.time()
ecran_id = display(ecran(), display_id=True)
ex = ProcessPoolExecutor(NB_WORKERS, mp_context=mp.get_context("spawn"),
                         initializer=ce.init_worker, initargs=(False, 1, SF, SF_TEMPS))
en_attente = {ex.submit(ce.tache_partie, t + (SUIVI,)) for t in reste}
k = 0
with open(PARTIES, "a") as f:
    while en_attente:
        finis, en_attente = wait(en_attente, timeout=2, return_when=FIRST_COMPLETED)
        for fu in finis:
            k += 1
            try:
                r = fu.result()
            except Exception as e:
                print("erreur :", e)
                continue
            f.write(json.dumps(r) + "\n"); f.flush()
            enregistrer(r)
            adv, couleur, res = infos(r)
            par_niveau[adv].append(r)
            ligne = (f"{EMOJI[res]} contre {adv} ({len(par_niveau[adv])}/{PAR_NIVEAU}) : {r['ouverture']}, "
                     f"PWN avec les {couleur} → {res} ({r['fin']}, {r['plies'] // 2} coups)")
            derniers.append(ligne_fin(r))
            print(f"[{k}/{len(reste)} | {(time.time() - t0) / 60:.0f} min] {ligne}")
            if len(par_niveau[adv]) == PAR_NIVEAU:
                bilan_niveau(adv, par_niveau[adv])
        ecran_id.update(ecran())
        if time.time() - t0 > BUDGET_H * 3600:
            print("budget temps atteint, on s'arrête là")
            for x in en_attente:
                x.cancel()
            break
ex.shutdown(wait=False, cancel_futures=True)
'''),
    ("md", r'''
## Résultats
'''),
    ("code", r'''
parties = pd.DataFrame([json.loads(l) for l in open(PARTIES)])
parties = parties.drop_duplicates(["blancs", "noirs", "ouverture"])
parties = parties[parties.blancs.isin(STOCKFISH) | parties.noirs.isin(STOCKFISH)]
parties["adversaire"] = np.where(parties.blancs == JOUEUR, parties.noirs, parties.blancs)
parties["couleur"] = np.where(parties.blancs == JOUEUR, "blancs", "noirs")
gagne = {"1-0": "blancs", "0-1": "noirs"}
parties["resultat_pwn"] = [
    "nulle" if r == "1/2-1/2" else ("victoire" if gagne[r] == c else "défaite")
    for r, c in zip(parties.resultat, parties.couleur)]
parties["score"] = parties.resultat_pwn.map({"victoire": 1.0, "nulle": 0.5, "défaite": 0.0})
with open(f"{SORTIE}/pwn3_parties.pgn", "w") as f:
    f.write("\n\n".join(parties.pgn))
toutes = []
for _, r in parties.iterrows():
    pos = pd.DataFrame(r.positions)
    pos.insert(0, "adversaire", r.adversaire); pos.insert(1, "ouverture", r.ouverture)
    pos.insert(2, "PWN_joue", r.couleur); pos["resultat_PWN"] = r.resultat_pwn
    toutes.append(pos)
pd.concat(toutes).to_csv(f"{SORTIE}/pwn3_positions.csv", index=False)
liste = parties[["adversaire", "ouverture", "couleur", "resultat_pwn", "fin", "plies"]].copy()
liste["coups"] = liste.pop("plies") // 2
liste.sort_values(["adversaire", "ouverture", "couleur"]).to_csv(f"{SORTIE}/pwn3_liste_parties.csv", index=False)

tab = parties.groupby("adversaire").agg(
    victoires=("resultat_pwn", lambda s: (s == "victoire").sum()),
    nulles=("resultat_pwn", lambda s: (s == "nulle").sum()),
    defaites=("resultat_pwn", lambda s: (s == "défaite").sum()),
    parties=("score", "size"),
    score_pct=("score", lambda s: 100 * s.mean()),
    score_blancs=("score", lambda s: 100 * s[parties.loc[s.index, "couleur"] == "blancs"].mean()),
    score_noirs=("score", lambda s: 100 * s[parties.loc[s.index, "couleur"] == "noirs"].mean()),
)
tab["Elo Stockfish"] = tab.index.map(STOCKFISH)
tab = tab.sort_values("Elo Stockfish")
print(len(parties), "parties | fins :", parties.fin.value_counts().to_dict())
print("temps moyen par coup de", JOUEUR, ":",
      round(pd.concat([parties[parties.blancs == JOUEUR].s_par_coup_blancs,
                       parties[parties.noirs == JOUEUR].s_par_coup_noirs]).mean(), 2), "s")
tab.round(1)
'''),
    ("code", r'''
from scipy.optimize import minimize_scalar

def elo_mle(df):
    """Elo de PWN par maximum de vraisemblance, les niveaux Stockfish étant fixés (nulle = ½ victoire)"""
    e_adv, s = df.adversaire.map(STOCKFISH).to_numpy(), df.score.to_numpy()
    def nll(r):
        p = np.clip(1 / (1 + 10 ** ((e_adv - r) / 400)), 1e-9, 1 - 1e-9)
        return -np.sum(s * np.log(p) + (1 - s) * np.log(1 - p))
    return minimize_scalar(nll, bounds=(0, 4000), method="bounded").x

elo = elo_mle(parties)
rng = np.random.default_rng(SEED)
boot = [elo_mle(parties.sample(len(parties), replace=True, random_state=int(rng.integers(1e9)))) for _ in range(500)]
bas, haut = np.percentile(boot, [2.5, 97.5])
print(f"Elo de {JOUEUR} en partie : {elo:.0f}  (IC 95 % : {bas:.0f} – {haut:.0f})")
pd.DataFrame([{"joueur": JOUEUR, "Elo": elo, "IC 95 % bas": bas, "IC 95 % haut": haut, "parties": len(parties)}]) \
  .to_csv(f"{SORTIE}/pwn3_elo.csv", index=False)
tab.to_csv(f"{SORTIE}/pwn3_par_niveau.csv")
'''),
    ("code", r'''
fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.linspace(min(NIVEAUX) - 100, max(NIVEAUX) + 100, 200)
ax.plot(x, 100 / (1 + 10 ** ((x - elo) / 400)), color="gray", lw=1, ls="--", label=f"attendu pour Elo {elo:.0f}")
ax.bar(tab["Elo Stockfish"], tab.score_pct, width=120, color="#58a", label="score réel de " + JOUEUR)
for e, v, n in zip(tab["Elo Stockfish"], tab.score_pct, tab.parties):
    ax.text(e, v + 2, f"{v:.0f} %", ha="center", fontsize=9)
ax.axhline(50, color="black", lw=0.5)
ax.axvline(elo, color="#c55", lw=1.5, label=f"Elo estimé {elo:.0f} [{bas:.0f}–{haut:.0f}]")
ax.set_xticks(NIVEAUX); ax.set_xlabel("niveau de Stockfish (UCI_Elo)"); ax.set_ylabel("score de PWN@ab3 (%)")
ax.set_ylim(0, 110); ax.set_title(f"{JOUEUR} contre Stockfish 17"); ax.legend(loc="upper right", fontsize=8)
plt.tight_layout(); plt.savefig(f"{SORTIE}/pwn3_vs_stockfish.png", dpi=120); plt.show()
'''),
    ("md", r'''
### Les fichiers (onglet Output)
| Fichier | Contenu |
|---|---|
| `pwn3_parties/stockfish_eloXXXX/<ouverture>_PWN-<couleur>.pgn` | chaque partie, à coller sur **lichess.org/paste** pour la rejouer |
| `pwn3_parties/stockfish_eloXXXX/<ouverture>_PWN-<couleur>_positions.csv` | chaque coup de la partie : qui joue, le coup, la **position (FEN)** obtenue, le temps de réflexion |
| `pwn3_positions.csv` | toutes les positions de toutes les parties dans un seul fichier |
| `pwn3_liste_parties.csv` | la liste des parties : adversaire, ouverture, couleur de PWN, résultat, comment ça finit, nombre de coups |
| `pwn3_par_niveau.csv`, `pwn3_elo.csv`, `pwn3_vs_stockfish.png`, `pwn3_parties.pgn` | les bilans, l'Elo, la courbe, toutes les parties en un PGN |

### Comment lire les résultats
- Le niveau où le score passe sous **50 %**, c'est à peu près l'Elo de PWN@ab3.
- Si PWN fait encore > 50 % contre le niveau le plus fort, ajoute des niveaux plus hauts dans `NIVEAUX` (ex. 2700, 2900) et relance :
  les parties déjà jouées sont gardées.
- Avec 16 parties par niveau (80 au total), l'intervalle fait environ ±70–120 Elo. Pour plus de précision : `N_OUVERTURES = 12` (plus long).
- Beaucoup de fins `MAX_PLIES` ou de nulles par répétition → PWN ne sait pas convertir un avantage en finale.
'''),
]


if __name__ == "__main__":
    ecrire("01_echecs_precision_evaluation.ipynb", NB1)
    ecrire("02_echecs_puzzles_lichess.ipynb", NB2)
    ecrire("03_echecs_tournoi_elo.ipynb", NB3)
    ecrire("04_mind_embeddings_mteb_fr.ipynb", NB4)
    ecrire("05_finemind_finetuning.ipynb", NB5)
    ecrire("06_finemind2_questions_reponses.ipynb", NB6)
    ecrire("07_pwn3_vs_stockfish.ipynb", NB7)
