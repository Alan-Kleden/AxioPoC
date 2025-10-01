# -*- coding: utf-8 -*-
"""
Face-validity by THREAD for AfD axis.
- Projette chaque message sur l'axe u (dot ou cos).
- Agrège au niveau thread (mean|max).
- Sélectionne K threads top/bottom.
- Teste l’écart de proportions de label=1 (chi2 + odds ratio).
- Exporte tous les messages des fils sélectionnés.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import chi2_contingency
from math import isfinite

def load_axis(path: Path) -> np.ndarray:
    u = np.load(path).astype("float32")
    n = float(np.linalg.norm(u))
    if n == 0:
        raise ValueError("Axis has zero norm.")
    return u / n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, help="Parquet with columns: thread_id, e0..eD-1")
    ap.add_argument("--axis", required=True, help=".npy axis")
    ap.add_argument("--labels", required=True, help="CSV with thread_id,label (0/1)")
    ap.add_argument("--texts", required=True, help="CSV with thread_id,text (pour export)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--emb_prefix", default="e")
    ap.add_argument("--thread_col", default="thread_id")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--score", choices=["dot","cos"], default="dot",
                    help="dot=produit scalaire, cos=cosine (normalise e)")
    ap.add_argument("--agg", choices=["mean","max"], default="mean",
                    help="agrégation message->thread")
    ap.add_argument("--min_msgs", type=int, default=1, help="filtrer threads avec >=N msgs")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) embeddings
    t = pq.read_table(args.emb)
    df = t.to_pandas()
    thr = args.thread_col

    # Colonnes d'embeddings (CORRECTIF: emb_prefix au lieu de emb-prefix)
    emcols = [c for c in df.columns if c.startswith(args.emb_prefix)]
    emcols = sorted(emcols, key=lambda c: int(str(c).replace(args.emb_prefix, "")))
    if not emcols:
        raise ValueError(f"Aucune colonne d'embeddings trouvée avec le préfixe '{args.emb_prefix}'.")

    E = df[emcols].to_numpy(dtype="float32")

    # 2) axis
    u = load_axis(Path(args.axis))
    if E.shape[1] != u.shape[0]:
        raise ValueError(f"Dim mismatch: emb={E.shape[1]} vs axis={u.shape[0]}")

    # 3) scoring par message
    if args.score == "cos":
        nrm = np.linalg.norm(E, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        E = E / nrm
    s = (E @ u).astype("float32")
    df["_score"] = s

    # 4) agrégation au niveau thread
    gb = df.groupby(thr)["_score"]
    per_thr = gb.mean() if args.agg == "mean" else gb.max()
    counts = df.groupby(thr).size()

    # 5) filtre min_msgs
    valid_threads = counts[counts >= args.min_msgs].index
    per_thr = per_thr.loc[valid_threads]

    # 6) labels
    lab = pd.read_csv(args.labels, usecols=[args.thread_col, args.label_col]).dropna()
    lab[args.thread_col] = lab[args.thread_col].astype(int)
    lab[args.label_col] = lab[args.label_col].astype(int)

    per_thr = per_thr.reset_index().rename(columns={"_score":"score"})
    per_thr[thr] = per_thr[thr].astype(int)
    per_thr = per_thr.merge(lab, left_on=thr, right_on=args.thread_col, how="inner")
    per_thr = per_thr[[thr,"score",args.label_col]].rename(columns={args.label_col:"label"})

    # 7) top/bottom K fils
    per_thr = per_thr.sort_values("score", ascending=False)
    top = per_thr.head(args.topk).copy()
    bot = per_thr.tail(args.topk).copy()

    # 8) export messages des fils sélectionnés
    texts = pd.read_csv(args.texts, usecols=[args.thread_col,"text"])
    texts[args.thread_col] = texts[args.thread_col].astype(int)

    def dump_threads(df_sel, name):
        thr_set = set(df_sel[thr].tolist())
        msgs = df[df[thr].isin(thr_set)][[thr,"_score"]].copy()
        msgs = msgs.merge(texts, left_on=thr, right_on=args.thread_col, how="left")
        msgs = msgs.merge(lab, on=args.thread_col, how="left")
        msgs = msgs.rename(columns={args.thread_col:"thread_id"})
        msgs = msgs[["thread_id","_score","label","text"]].sort_values(
            ["thread_id","_score"], ascending=[True,False]
        )
        msgs.to_csv(outdir / f"{name}_threads_messages.csv", index=False)

    dump_threads(top, "top")
    dump_threads(bot, "bottom")

    # 9) test χ² + OR sur proportions label=1 (niveau thread)
    def counts_01(df_sel):
        v = df_sel["label"].astype(int)
        return int((v==1).sum()), int((v==0).sum())
    t1,t0 = counts_01(top)
    b1,b0 = counts_01(bot)

    table = np.array([[t1,t0],[b1,b0]], dtype="float64")
    chi2, p, _, _ = chi2_contingency(table, correction=False)

    eps = 0.5
    OR = ((t1+eps)*(b0+eps))/((t0+eps)*(b1+eps))
    delta_pp = (t1/(t1+t0+1e-9) - b1/(b1+b0+1e-9)) * 100.0

    summary = {
        "emb_path": str(args.emb),
        "axis_path": str(args.axis),
        "texts_path": str(args.texts),
        "labels_path": str(args.labels),
        "score_metric": args.score,
        "agg": args.agg,
        "min_msgs": int(args.min_msgs),
        "n_threads_scored": int(len(per_thr)),
        "topk": int(args.topk),
        "top_counts_[label1,label0]": [t1,t0],
        "bot_counts_[label1,label0]": [b1,b0],
        "test": "chi2",
        "p_value": float(p),
        "chi2": float(chi2),
        "odds_ratio": float(OR) if isfinite(OR) else None,
        "delta_pp_label1_top_minus_bottom": float(delta_pp)
    }
    (Path(args.outdir)/"summary_by_thread.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[OK] Face-validity (by thread) -> {args.outdir}")
    print(f" top: {(Path(args.outdir)/'top_threads_messages.csv')}")
    print(f" bot: {(Path(args.outdir)/'bottom_threads_messages.csv')}")
    print(f" chi2 p={p:.6g} | Δpp(label=1)={delta_pp:.1f} | OR≈{OR:.3f}")

if __name__ == "__main__":
    main()
