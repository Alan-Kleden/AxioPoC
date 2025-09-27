# -*- coding: utf-8 -*-
"""
make_pragmatics_afd.py
Calcule des marqueurs pragmatiques à partir d'un CSV (thread_id, <text-col>),
agrégés par thread, puis fusionne (si présent) avec un fichier de features
qui a le même chemin que --out-feat (pour ne pas perdre les colonnes déjà existantes).

Usage (exemple) :
  python scripts/make_pragmatics_afd.py ^
    --in C:\AxioPoC\artifacts_xfer\afd_messages_labeled.csv ^
    --text-col text ^
    --out-feat C:\AxioPoC\artifacts_xfer\features_axio_afd.csv
"""
import argparse
import os
from pathlib import Path
import re
import numpy as np
import pandas as pd


# -----------------------
# Lexiques très simples
# -----------------------
POLITE = [
    "please", "thanks", "thank you", "appreciate", "sorry", "excuse me", "would you mind"
]
HEDGES = [
    "maybe", "perhaps", "seems", "appear", "appears", "i think", "i guess",
    "might", "could", "somewhat", "kind of", "sort of", "probably", "possibly", "likely", "unsure"
]
AGREE_PHRASES = [
    "i agree", "good point", "you're right", "you are right", "makes sense", "i concur"
]
NEG_MARKERS = [
    "not", "no", "never", "don't", "doesn't", "can't", "won't", "shouldn't",
    "bad", "wrong", "fail", "fails", "failure"
]


def norm_text(s):
    """Normalise une cellule texte de manière robuste (NaN -> '')."""
    if not isinstance(s, str):
        return ""
    return s.lower()


def tokenize_words(text):
    """Tokenise en mots ASCII basiques."""
    return re.findall(r"[a-z']+", text)


def count_list_terms(tokens, terms):
    """Compte le nombre d'occurrences de termes 1-mot dans tokens."""
    ts = set([t for t in terms if " " not in t])
    return sum(1 for t in tokens if t in ts)


def count_phrase_terms(text, phrases):
    """Compte le nombre d'occurrences (overlap autorisé) de phrases multi-mots dans le texte brut."""
    cnt = 0
    for p in phrases:
        if " " in p:
            # recherche insensible à la casse sur le texte normalisé
            cnt += len(re.findall(re.escape(p), text))
    return cnt


def features_for_thread(texts):
    """
    Calcule les features pragmatiques pour un thread à partir d'une liste de messages str.
    Retourne un dict avec:
      polite_ratio, hedge_ratio, you_i_ratio, agree_markers, neg_markers
    """
    # normalisation + concat
    texts = [norm_text(t) for t in texts]
    concat = " ".join(texts)
    tokens = tokenize_words(concat)
    n_tok = max(1, len(tokens))  # éviter div / 0

    # you/i ratio
    you_cnt = sum(1 for t in tokens if t == "you")
    i_cnt   = sum(1 for t in tokens if t == "i")
    you_i_ratio = you_cnt / max(1, i_cnt)

    # polite (mots + phrases)
    polite_cnt = count_list_terms(tokens, POLITE) + count_phrase_terms(concat, POLITE)
    # hedges
    hedge_cnt  = count_list_terms(tokens, HEDGES) + count_phrase_terms(concat, HEDGES)
    # agree markers
    agree_cnt  = count_phrase_terms(concat, AGREE_PHRASES)
    # negative markers (mots)
    neg_cnt    = count_list_terms(tokens, NEG_MARKERS)

    return {
        "polite_ratio": polite_cnt / n_tok,
        "hedge_ratio":  hedge_cnt  / n_tok,
        "you_i_ratio":  you_i_ratio,
        "agree_markers": agree_cnt / n_tok,
        "neg_markers":   neg_cnt   / n_tok,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="CSV avec au moins thread_id,<text-col>")
    ap.add_argument("--text-col", required=True, help="Nom de la colonne texte dans --in")
    ap.add_argument("--out-feat", required=True, help="CSV de sortie des features (merge si fichier déjà existant)")
    args = ap.parse_args()

    inp = Path(args.inp)
    outp = Path(args.out_feat)
    text_col = args.text_col

    if not inp.exists():
        raise SystemExit(f"Fichier introuvable: {inp}")

    df = pd.read_csv(inp)
    if "thread_id" not in df.columns:
        raise SystemExit("Le CSV d'entrée doit contenir 'thread_id'.")
    if text_col not in df.columns:
        raise SystemExit(f"Le CSV d'entrée ne contient pas la colonne texte '{text_col}'.")

    # Normalisation robuste du texte avant groupby
    df[text_col] = df[text_col].fillna("").astype(str)

    # Agrégation par thread: liste de textes par thread_id
    grp = df.groupby("thread_id")[text_col].apply(list)

    # Calcul des features pragmatiques par thread
    feats = []
    for tid, texts in grp.items():
        vals = features_for_thread(texts)
        vals["thread_id"] = tid
        feats.append(vals)
    PR = pd.DataFrame(feats)

    # Si un fichier de features existe déjà au chemin --out-feat, on merge (pour ne rien perdre)
    if outp.exists():
        base = pd.read_csv(outp)

        # Harmonise le type de clé
        if "thread_id" in base.columns:
            base["thread_id"] = base["thread_id"].astype("int64")
        PR["thread_id"] = PR["thread_id"].astype("int64")

        # Supprime dans "base" les colonnes pragmatiques qui vont être réécrites
        overlap = list(set(PR.columns) & set(base.columns) - {"thread_id"})
        base = base.drop(columns=overlap, errors="ignore")

        # Merge propre (ajoute/remplace les colonnes de PR)
        merged = base.merge(PR, on="thread_id", how="left")

        merged.to_csv(outp, index=False)
        print(f"Updated (merged) features -> {outp}  | rows={len(merged)}")

        print(f"Updated (merged) features -> {outp}  | rows={len(merged)}")
    else:
        # sinon, on écrit juste nos colonnes pragmatiques
        PR.to_csv(outp, index=False)
        print(f"Wrote pragmatics-only features -> {outp}  | rows={len(PR)}")

    # Petit récap de couverture (non-zéro)
    out_df = pd.read_csv(outp)
    cols_check = ["polite_ratio", "hedge_ratio", "you_i_ratio", "agree_markers", "neg_markers"]
    avail = [c for c in cols_check if c in out_df.columns]
    cov = {c: round(float((out_df[c].fillna(0) != 0).mean()) * 100, 2) for c in avail}
    print("Coverage (% non-zéro):", cov)


if __name__ == "__main__":
    main()
