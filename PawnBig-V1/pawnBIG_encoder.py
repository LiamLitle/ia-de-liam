BASE = {c: i + 1 for i, c in enumerate("PNBRQK")}


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
            out[(r if noir else 7 - r) * 8 + c] = BASE[ch.upper()] + (0 if nous else 6)
            c += 1
    ours, theirs = ("kq", "KQ") if noir else ("KQ", "kq")
    out[64] = ours[0] in f[2]
    out[65] = ours[1] in f[2]
    out[66] = theirs[0] in f[2]
    out[67] = theirs[1] in f[2]
    if f[3] != "-":
        out[68] = ord(f[3][0]) - 96
    return bytes(out)
