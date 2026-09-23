"""Soluce 1 : cartes tactiques calculées dans le réseau (torch pur, exportable en ONNX).

Entrée : les 17 cartes de P.A.W.N. (B, 17, 8, 8), vues du camp qui joue.
  cartes 0-5 = nos P N B R Q K, cartes 6-11 = leurs P N B R Q K.
  ligne 0 = notre rangée de départ, nos pions montent (+1 en ligne).
Sortie : (B, 10, 8, 8)
  0 cases attaquées par nous          1 cases attaquées par eux
  2 nb d'attaquants nous (0-3)/3      3 nb d'attaquants eux (0-3)/3
  4 nos pièces en prise               5 leurs pièces en prise
  6 nos pièces clouées au roi         7 leurs pièces clouées au roi
  8 pression sur notre roi            9 pression sur leur roi
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

VALEURS = (1.0, 3.0, 3.0, 5.0, 9.0, 0.0)
ORTHO = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))
CAVALIER = ((2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2))
ROI = ORTHO + DIAG


def dec(x, dr, dc):
    """décale la carte de dr lignes et dc colonnes (le vide entre, ce qui sort disparaît)"""
    p = F.pad(x, (2, 2, 2, 2))
    return p[..., 2 - dr:10 - dr, 2 - dc:10 - dc]


def rayons(src, occ, dirs):
    """cases atteintes par des pièces glissantes ; la première pièce rencontrée est atteinte puis le rayon s'arrête"""
    tot = 0
    for d in dirs:
        f = src
        for _ in range(7):
            f = dec(f, *d)
            tot = tot + f
            f = f * (1 - occ)
    return tot


def attaques(P, occ, avant):
    """P : (B,6,8,8) pièces d'un camp -> (B,6,8,8) nombre d'attaquants de chaque type par case"""
    pion = dec(P[:, 0:1], avant, -1) + dec(P[:, 0:1], avant, 1)
    cav = sum(dec(P[:, 1:2], *d) for d in CAVALIER)
    roi = sum(dec(P[:, 5:6], *d) for d in ROI)
    fou = rayons(P[:, 2:3], occ, DIAG)
    tour = rayons(P[:, 3:4], occ, ORTHO)
    dame = rayons(P[:, 4:5], occ, ORTHO + DIAG)
    return torch.cat([pion, cav, fou, tour, dame, roi], 1)


def clouees(propres, roi, tour_dame, fou_dame, occ, occ_propre):
    """pièces de notre camp clouées à notre roi par une tour/fou/dame adverse"""
    res = 0
    for d in ORTHO + DIAG:
        gl = tour_dame if d in ORTHO else fou_dame
        cur, prem = roi, 0
        for _ in range(7):
            cur = dec(cur, *d)
            prem = prem + cur * occ
            cur = cur * (1 - occ)
        prem = prem * occ_propre
        cur, sec = prem, 0
        for _ in range(7):
            cur = dec(cur, *d)
            sec = sec + cur * occ
            cur = cur * (1 - occ)
        drap = ((sec * gl).sum((1, 2, 3), keepdim=True) > 0).float()
        res = res + prem * drap
    return res


class Soluce(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("val", torch.tensor(VALEURS).view(1, 6, 1, 1), persistent=False)
        # valeur pour classer les attaquants : le roi passe en dernier
        self.register_buffer("val_att", torch.tensor(VALEURS[:5] + (50.0,)).view(1, 6, 1, 1), persistent=False)

    def forward(self, x):
        nous, eux = x[:, 0:6], x[:, 6:12]
        occ_n, occ_e = nous.sum(1, keepdim=True), eux.sum(1, keepdim=True)
        occ = occ_n + occ_e
        An, Ae = attaques(nous, occ, 1), attaques(eux, occ, -1)
        cn, ce = An.sum(1, keepdim=True), Ae.sum(1, keepdim=True)

        v = self.val_att
        pn = torch.where(An > 0, v.expand_as(An), torch.full_like(An, 99.0)).amin(1, keepdim=True)
        pe = torch.where(Ae > 0, v.expand_as(Ae), torch.full_like(Ae, 99.0)).amin(1, keepdim=True)

        val_n = (nous * self.val).sum(1, keepdim=True)
        val_e = (eux * self.val).sum(1, keepdim=True)
        sans_roi_n, sans_roi_e = nous[:, :5].sum(1, keepdim=True), eux[:, :5].sum(1, keepdim=True)
        pr_n = sans_roi_n * (ce > 0).float() * (((cn == 0).float() + (pe < val_n).float()) > 0).float()
        pr_e = sans_roi_e * (cn > 0).float() * (((ce == 0).float() + (pn < val_e).float()) > 0).float()

        cl_n = clouees(nous, nous[:, 5:6], eux[:, 3:4] + eux[:, 4:5], eux[:, 2:3] + eux[:, 4:5], occ, occ_n)
        cl_e = clouees(eux, eux[:, 5:6], nous[:, 3:4] + nous[:, 4:5], nous[:, 2:3] + nous[:, 4:5], occ, occ_e)

        zone_n = F.max_pool2d(nous[:, 5:6], 3, 1, 1)
        zone_e = F.max_pool2d(eux[:, 5:6], 3, 1, 1)
        return torch.cat([
            (cn > 0).float(), (ce > 0).float(),
            cn.clamp(max=3) / 3, ce.clamp(max=3) / 3,
            pr_n, pr_e, cl_n, cl_e,
            zone_n * ce.clamp(max=3) / 3, zone_e * cn.clamp(max=3) / 3,
        ], 1)
