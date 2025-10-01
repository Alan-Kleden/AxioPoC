#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Face-validity de l'axe télotopique (AfD)
- Projette chaque message: score = <embedding, axis>
- Trie globalement, extrait TOP K et BOTTOM K
- Option --labeled-only : ne garder que des messages dont le thread est labellé
- Test quanti: chi2 | fisher | auto (par défaut auto)
- Seuils: --p-thresh (def 0.01) et --delta-pp (def 20.0 points)

Entrées attendues :
  --emb    : Parquet des embeddings messages (colonnes: '__msg_id__','thread_id', 'e0'...'e{d-1}')
  --axis   : NPY du vecteur axe (dim = nb de colonnes 'e*')
  --texts  : CSV des textes (ordre identique aux embeddings, colonnes: 'thread_id','text')
  --labels : CSV labels AfD (colonnes: 'thread_id','label')
"""

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import chi2_contingency, fisher_exact


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, help="Parquet embeddings")
    ap.add_argument("--axis", required=True, help="NPY axis vector")
    ap.add_argument("--texts", required=True, help="CSV texts (thread_id,text) couplé")
    ap.add_argument("--labels", required=True, help="CSV labels (thread_id,label)")
    ap.add_argument("--outdir", required=True, help="Dossier de sortie")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--thread-col", default="thread_id")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--msg-id-col", default="__msg_id__")
    ap.add_argument("--emb-prefix", default="e")
    ap.add_argument("--labeled-only", action="store_true",
                    help="Enforcer que TOP/BOTTOM contiennent uniquement des messages de threads labellés")
    ap.add_argument("--test", choices=["auto", "chi2", "fisher"], default="auto")
    ap.add_argument("--p-thresh", type=float, default=0.01)
    ap.add_argument("--delta-pp", type=float, default=20.0)
    return ap.parse_args()


def load_embeddings(path_parquet, emb_prefix, msg_id_col, thread_col):
    tbl = pq.read_table(path_parquet)
    df = tbl.to_pandas()
    # colonnes embedding
    ecols = [c for c in df.columns if isinstance(c, str) and c.startswith(emb_prefix)]
    # vérifs colonnes clés
    for col in (msg_id_col, thread_col):
        if col not in df.columns:
            raise ValueError(f"Colonne manquante dans embeddings: {col}")
    return df[[msg_id_col, thread_col] + ecols], ecols


def load_axis(path_npy, dim):
    u = np.load(path_npy)
    if u.ndim != 1 or u.shape[0] != dim:
        raise ValueError(f"Dimension axe invalide. Attendu {dim}, reçu {u.shape}.")
    # normaliser pour score cos-like (optionnel)
    n = np.linalg.norm(u) + 1e-9
    return (u / n).astype(np.float32)


def project_scores(df_emb, ecols, axis_vec):
    # normaliser chaque embedding avant dot pour être proche d’un cosinus
    E = df_emb[ecols].to_numpy(dtype=np.float32)
    nE = np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    E = E / nE
    s = (E @ axis_vec.astype(np.float32))
    return s


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ------- load -------
    df_emb, ecols = load_embeddings(args.emb, args.emb_prefix, args.msg_id_col, args.thread_col)
    axis = load_axis(args.axis, dim=len(ecols))
    scores = project_scores(df_emb, ecols, axis)

    # Texts : on suppose même ordre que msg_id (= index)
    # On ajoute un index implicite sur le CSV des textes pour retrouver par __msg_id__
    df_txt = pd.read_csv(args.texts, usecols=[args.thread_col, "text"])
    if len(df_txt) <= df_emb[args.msg_id_col].max():
        raise ValueError("Le CSV des textes ne correspond pas en longueur aux embeddings.")
    # labels
    df_lab = pd.read_csv(args.labels, usecols=[args.thread_col, args.label_col])
    df_lab[args.thread_col] = df_lab[args.thread_col].astype(int)

    # ------- assemble master -------
    # On s'appuie sur l'alignement par __msg_id__ (ligne = message)
    master = pd.DataFrame({
        args.msg_id_col: df_emb[args.msg_id_col].values,
        args.thread_col: df_emb[args.thread_col].astype(int).values,
        "score": scores,
    })
    # récupérer le texte par index (msg_id)
    # NOTE: on utilise .iloc sur df_txt (même ordre que embeddings/messages)
    texts_series = df_txt["text"]
    master["text"] = texts_series.iloc[master[args.msg_id_col].values].values

    # joindre les labels par thread_id
    master = master.merge(df_lab, on=args.thread_col, how="left")

    # filtrage labeled-only si demandé
    if args.labeled_only:
        master = master[master[args.label_col].notna()].copy()

    # ------- top/bottom -------
    # trier par score
    m_sorted = master.sort_values("score", ascending=False)
    top = m_sorted.head(args.topk).copy()
    bot = m_sorted.tail(args.topk).copy()

    # convertir labels en entiers si possible
    def _to_int(x):
        try:
            return int(x)
        except Exception:
            return np.nan

    top["label_int"] = top[args.label_col].map(_to_int)
    bot["label_int"] = bot[args.label_col].map(_to_int)

    # ------- test quanti -------
    # On construit table 2x2 : group (top/bot) x label (1/0)
    def counts01(df_part):
        c1 = int((df_part["label_int"] == 1).sum())
        c0 = int((df_part["label_int"] == 0).sum())
        return c1, c0

    t1, t0 = counts01(top)
    b1, b0 = counts01(bot)

    # delta pp (en points de %)
    def pct1(c1, c0):
        tot = c1 + c0
        return 100.0 * c1 / tot if tot > 0 else np.nan

    delta_pp = pct1(t1, t0) - pct1(b1, b0)

    table = np.array([[t1, t0],
                      [b1, b0]], dtype=np.int64)

    test_used = "chi2"
    p_value = np.nan
    chi2_val = None
    odds_ratio = None

    if args.test == "auto":
        # si une case attendue proche de 0 → fisher
        if (table < 5).any():
            test_used = "fisher"
        else:
            test_used = "chi2"
    else:
        test_used = args.test

    if test_used == "chi2":
        try:
            chi2_val, p_value, dof, exp = chi2_contingency(table, correction=False)
        except ValueError:
            # fallback sur fisher si chi2 impossible (attendus nuls)
            test_used = "fisher"
            odds_ratio, p_value = fisher_exact(table, alternative="two-sided")
    else:
        odds_ratio, p_value = fisher_exact(table, alternative="two-sided")

    # critères
    pass_flag = (p_value < args.p_thresh) and (abs(delta_pp) >= args.delta_pp)

    # ------- sauvegardes -------
    # CSV top/bottom
    keep_cols = [args.msg_id_col, args.thread_col, "score", args.label_col, "text"]
    top_out = outdir / f"top{args.topk}_messages.csv"
    bot_out = outdir / f"bottom{args.topk}_messages.csv"
    top[keep_cols].to_csv(top_out, index=False)
    bot[keep_cols].to_csv(bot_out, index=False)

    # Résumé JSON
    summary = {
        "emb_path": str(Path(args.emb).resolve()),
        "axis_path": str(Path(args.axis).resolve()),
        "texts_path": str(Path(args.texts).resolve()),
        "labels_path": str(Path(args.labels).resolve()),
        "n_total_msgs": int(len(master)),
        "topk": int(args.topk),
        "top_counts_[label1,label0]": [int(t1), int(t0)],
        "bot_counts_[label1,label0]": [int(b1), int(b0)],
        "test": test_used,
        "p_value": float(p_value) if p_value == p_value else None,
        "chi2": float(chi2_val) if chi2_val is not None else None,
        "odds_ratio": float(odds_ratio) if odds_ratio is not None else None,
        "delta_pp_label1_top_minus_bottom": float(delta_pp) if delta_pp == delta_pp else None,
        "criteria": {
            "p_threshold": float(args.p_thresh),
            "delta_pp_threshold": float(args.delta_pp)
        },
        "pass_face_validity_quant": bool(pass_flag)
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[OK] Face-validity dumps -> {outdir}")
    print(" top:", top_out)
    print(" bot:", bot_out)
    print(f" {test_used} p= {p_value} | Δpp(label=1)= {round(delta_pp,1)} | pass= {pass_flag}")


if __name__ == "__main__":
    main()
