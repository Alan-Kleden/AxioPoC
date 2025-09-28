# -*- coding: utf-8 -*-
"""
Vérifie l'alignement des thread_id entre:
 - features_axio_cmv.csv (CMV features)
 - cmv_vader_by_thread.csv (VADER agrégé par thread)

Imprime :
- exemples de thread_id des deux fichiers
- comptage par format (t3_* vs numérique)
- taille des ensembles d'ID
- intersections exactes
- intersections après conversion numeric -> t3_<base36>
(ne modifie aucun fichier)
"""
import pandas as pd
from pathlib import Path

FEAT = Path(r".\artifacts_xfer\features_axio_cmv.csv")
VDR  = Path(r".\artifacts_xfer\cmv_vader_by_thread.csv")  # tel que produit par compute_vader_by_thread.py

def to_base36(n: int) -> str:
    if n == 0: return "0"
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    out = []
    while n > 0:
        n, r = divmod(n, 36)
        out.append(digits[r])
    return "".join(reversed(out))

def to_t3(s):
    """convertit une string s en t3_<base36> si s est numérique; sinon renvoie s tel quel."""
    try:
        if isinstance(s, str) and s.startswith("t3_"):
            return s
        n = int(float(str(s)))
        return "t3_" + to_base36(n)
    except Exception:
        return str(s)

def main():
    if not FEAT.exists(): raise SystemExit(f"Introuvable: {FEAT}")
    if not VDR.exists():  raise SystemExit(f"Introuvable: {VDR}")

    fx = pd.read_csv(FEAT, usecols=["thread_id"])
    vd = pd.read_csv(VDR,  usecols=["thread_id"])

    fx["thread_id"] = fx["thread_id"].astype(str)
    vd["thread_id"] = vd["thread_id"].astype(str)

    # Stats de format
    fx_t3   = (fx["thread_id"].str.startswith("t3_")).sum()
    fx_num  = (fx["thread_id"].str.fullmatch(r"\d+")).sum()
    vd_t3   = (vd["thread_id"].str.startswith("t3_")).sum()
    vd_num  = (vd["thread_id"].str.fullmatch(r"\d+")).sum()

    print("== échantillons ==")
    print("CMV features (5):", fx["thread_id"].head(5).tolist())
    print("VADER by thread (5):", vd["thread_id"].head(5).tolist())
    print()

    print("== formats ==")
    print(f"CMV features: total={len(fx)} | t3_*= {fx_t3} | numeriques= {fx_num}")
    print(f"VADER       : total={len(vd)} | t3_*= {vd_t3} | numeriques= {vd_num}")
    print()

    set_fx = set(fx["thread_id"])
    set_vd = set(vd["thread_id"])

    inter_exact = len(set_fx & set_vd)
    print(f"== intersections ==")
    print(f"Exact match (telles quelles): {inter_exact}")

    # Hypothèse: VD en numérique → conversion en t3_<base36>
    vd_conv = vd["thread_id"].map(to_t3)
    inter_conv = len(set_fx & set(vd_conv))
    print(f"Après conversion numeric->t3_<base36> côté VADER: {inter_conv}")

    # Taux de couverture potentielle après conversion
    cover_fx = inter_conv / len(set_fx) * 100 if len(set_fx) else 0.0
    cover_vd = inter_conv / len(set(vd_conv)) * 100 if len(vd_conv) else 0.0
    print(f"Couverture CMV features par VADER converti: {cover_fx:.2f}%")
    print(f"Couverture VADER converti par CMV features: {cover_vd:.2f}%")

if __name__ == "__main__":
    main()
