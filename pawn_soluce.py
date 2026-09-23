"""P.A.W.N. + Soluce 1 : le réseau pawn_big avec les cartes tactiques en entrée."""
import torch, torch.nn as nn, torch.nn.functional as F
from soluce import Soluce


class Block(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.c1 = nn.Conv2d(f, f, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(f)
        self.c2 = nn.Conv2d(f, f, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(f)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        return F.relu(x + self.b2(self.c2(y)))


class Pawn(nn.Module):
    def __init__(self, blocks, f, entrees=17):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(entrees, f, 3, padding=1, bias=False), nn.BatchNorm2d(f), nn.ReLU())
        self.body = nn.Sequential(*[Block(f) for _ in range(blocks)])
        self.head = nn.Sequential(nn.Conv2d(f, 8, 1), nn.ReLU(), nn.Flatten(),
                                  nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        return self.head(self.body(self.stem(x))).squeeze(1)


class PawnSoluce(Pawn):
    """entrée : les 17 cartes habituelles ; les 10 cartes Soluce sont calculées ici même"""
    def __init__(self, blocks, f):
        super().__init__(blocks, f, entrees=27)
        self.soluce = Soluce()

    def forward(self, x):
        return super().forward(torch.cat([x, self.soluce(x)], 1))


def depuis_pawn(chemin_pt):
    """charge pawn_big.pt ; les poids des 10 nouvelles cartes sont mis à zéro -> même réseau au départ"""
    ck = torch.load(chemin_pt, map_location="cpu")
    blocs, f = ck["cfg"]
    m = PawnSoluce(blocs, f)
    sd = dict(ck["model"])
    w = sd["stem.0.weight"]
    sd["stem.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], 10, 3, 3)], 1)
    m.load_state_dict(sd)
    return m.eval(), ck
