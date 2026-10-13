"""Exporte les poids de P.O.W.N. au format binaire lu par le moteur C++ (pown.bin).

    python exporter_poids.py pown.onnx pown.bin     (nécessite le paquet onnx)
    python exporter_poids.py pown.pt   pown.bin     (nécessite torch)

Format (little-endian) : "POWN", version=1, dims = [22528, 256, 32, 32], puis en float32 :
transformateur (22528x256), biais (256), c1.weight (32x512), c1.bias, c2.weight (32x32), c2.bias, sortie.weight (1x32), sortie.bias
"""
import struct
import sys

import numpy as np

ORDRE = ["transformateur.weight", "biais", "c1.weight", "c1.bias", "c2.weight", "c2.bias", "sortie.weight", "sortie.bias"]
FORMES = [(22528, 256), (256,), (32, 512), (32,), (32, 32), (32,), (1, 32), (1,)]


def ecrire_bin(poids, chemin_bin):
    with open(chemin_bin, "wb") as f:
        f.write(b"POWN")
        f.write(struct.pack("<i", 1))
        f.write(struct.pack("<4i", 22528, 256, 32, 32))
        for nom, forme in zip(ORDRE, FORMES):
            a = np.asarray(poids[nom], dtype=np.float32)
            assert a.shape == forme, (nom, a.shape, forme)
            f.write(np.ascontiguousarray(a).tobytes())
    print("écrit", chemin_bin)


def poids_depuis_modele(modele):
    return {k: v.detach().cpu().numpy() for k, v in modele.state_dict().items()}


def poids_depuis_onnx(chemin):
    import onnx
    from onnx import numpy_helper
    return {i.name: numpy_helper.to_array(i) for i in onnx.load(chemin).graph.initializer}


def exporter(modele, chemin_bin):
    ecrire_bin(poids_depuis_modele(modele), chemin_bin)


if __name__ == "__main__":
    src, dest = sys.argv[1], sys.argv[2]
    if src.endswith(".onnx"):
        ecrire_bin(poids_depuis_onnx(src), dest)
    else:
        import os
        import torch
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Pown"))
        import commun_pown as pown
        m = pown.ReseauPOWN()
        pt = torch.load(src, map_location="cpu")
        m.load_state_dict(pt["modele"] if "modele" in pt else pt)
        exporter(m, dest)
