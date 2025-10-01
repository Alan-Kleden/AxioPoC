# -*- coding: utf-8 -*-
"""
eval_thread_models.py
- Reprend le split train/test du hold-out (train_ids.txt / test_ids.txt)
- Joint:
   * scores_axis (scores_train.csv / scores_test.csv)  -> feature: axis_mean
   * baselines (thread_baselines.csv) -> len_chars, len_tokens, vader_mean
- Modèle A = baselines
- Modèle B = baselines + axis_mean
- Compare AUC sur TEST + SAFE "ΔAUC" en shuffle la colonne axis_mean (N permutations)

Usage:
python eval_thread_models.py ^
  --holdout-dir "C:\AxioPoC\REPORTS\holdout_threads" ^
  --baselines "C:\AxioPoC\REPORTS\thread_baselines.csv" ^
  --iters 500 ^
  --out "C:\AxioPoC\REPORTS\model_comp_summary.json"
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

def load_holdout(holdout_dir):
    d = Path(holdout_dir)
    tr = pd.read_csv(d/"scores_train.csv", usecols=["thread_id","label","score_axis_mean"])
    te = pd.read_csv(d/"scores_test.csv",  usecols=["thread_id","label","score_axis_mean"])
    return tr, te

def fit_predict_auc(Xtr, ytr, Xte, yte):
    clf = LogisticRegression(max_iter=200, solver="liblinear")
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:,1]
    return roc_auc_score(yte, p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-dir", required=True)
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tr, te = load_holdout(args.holdout-dir)
    base = pd.read_csv(args.baselines)

    def join(df):
        x = df.merge(base, on="thread_id", how="left")
        x = x.dropna(subset=["long_mean_chars","long_mean_tokens","vader_mean"]).copy()
        return x

    tr = join(tr); te = join(te)
    # Features
    Xa = tr[["long_mean_chars","long_mean_tokens","vader_mean"]].values
    ya = tr["label"].values
    Xb = np.column_stack([Xa, tr["score_axis_mean"].values])

    Ta = te[["long_mean_chars","long_mean_tokens","vader_mean"]].values
    yt = te["label"].values
    Tb = np.column_stack([Ta, te["score_axis_mean"].values])

    auc_A = fit_predict_auc(Xa, ya, Ta, yt)
    auc_B = fit_predict_auc(Xb, ya, Tb, yt)
    delta = auc_B - auc_A

    # SAFE-like: shuffle axis_mean de TEST et recompute ΔAUC (fixe train)
    rng = np.random.default_rng(123)
    null = []
    for _ in range(args.iters):
        Tb_shuf = Tb.copy()
        Tb_shuf[:,-1] = rng.permutation(Tb_shuf[:,-1])
        auc_Bs = fit_predict_auc(Xb, ya, Tb_shuf, yt)
        null.append(auc_Bs - auc_A)
    null = np.array(null, dtype=float)
    p = float((np.sum(null >= delta) + 1) / (len(null) + 1))

    out = {
        "holdout_dir": str(Path(args.holdout-dir)),
        "baselines": str(Path(args.baselines)),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "AUC_A_baselines": float(auc_A),
        "AUC_B_plus_axis": float(auc_B),
        "delta_AUC": float(delta),
        "SAFE_perms": int(len(null)),
        "SAFE_p_for_delta": p,
        "note": "A=longueur+sentiment; B=A+axis_mean (même split que hold-out)"
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Models A vs B -> {args.out}")
    print(f" AUC_A={auc_A:.3f} | AUC_B={auc_B:.3f} | ΔAUC={delta:.3f} | p_SAFE={p:.4f}")
if __name__ == "__main__":
    main()
