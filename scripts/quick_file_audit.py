# scripts/quick_file_audit.py
import argparse, os, json, unicodedata as u
import pandas as pd

def pct(x): return round(float(x)*100, 2)

def audit_messages(path, out_dir, id_col="thread_id", text_col="text", sample_n=5):
    rep = {"path": path, "exists": os.path.exists(path)}
    if not rep["exists"]:
        return rep, None, None

    usecols = [c for c in [id_col, text_col] if c]
    df = pd.read_csv(path, usecols=usecols, dtype=str)
    rep.update({
        "n_rows": int(len(df)),
        "n_threads": int(df[id_col].nunique()),
        "threads_lt_rows": bool(df[id_col].nunique() < len(df)),
        "empty_text_rate_pct": pct((df[text_col].astype(str).str.strip()=="").mean()),
    })
    le = df[text_col].astype(str).str.len()
    rep["len_stats"] = {
        "min": int(le.min()), "median": float(le.median()),
        "mean": round(float(le.mean()), 2), "max": int(le.max())
    }
    # caractères de contrôle / non-ascii “suspects”
    rep["weird_char_rate_pct"] = pct(df[text_col].map(
        lambda s: any((ord(ch)>127 and u.category(ch)[0] in "CZ") for ch in (s or ""))
    ).mean())

    # échantillons
    sample_rows = df.sample(min(sample_n, len(df)), random_state=1)
    sample_rows.to_csv(os.path.join(out_dir, "audit_sample_rows.csv"), index=False)

    # multi-messages si possible (nécessite >1 par thread)
    vc = df[id_col].value_counts()
    multi = vc[vc>=3].index[:3]
    rep["sample_threads_ge3"] = list(map(str, multi.tolist()))
    if len(multi)>0:
        df[df[id_col].isin(multi)].head(50).to_csv(os.path.join(out_dir, "audit_threads_ge3_head.csv"), index=False)

    return rep, df, usecols

def audit_parquet_embeddings(path, out_dir, id_col="thread_id", time_cols=("__t__","created_utc","time","timestamp")):
    rep = {"path": path, "exists": os.path.exists(path)}
    if not rep["exists"]:
        return rep, None
    # lis léger: id + temps + 10 emb pour signature
    df = pd.read_parquet(path)
    rep.update({
        "n_rows": int(len(df)),
        "n_threads": int(df[id_col].nunique()) if id_col in df.columns else None,
        "has_time_col": any(c in df.columns for c in time_cols),
        "emb_cols_head": [c for c in df.columns if str(c).startswith("e")][:5]
    })
    return rep, df

def audit_features_vs_labels(feat_path, lab_path, out_dir, id_col="thread_id"):
    rep = {"feat_path": feat_path, "lab_path": lab_path,
           "feat_exists": os.path.exists(feat_path),
           "lab_exists": os.path.exists(lab_path)}
    if not (rep["feat_exists"] and rep["lab_exists"]):
        return rep
    f = pd.read_csv(feat_path, usecols=[id_col])
    l = pd.read_csv(lab_path, usecols=[id_col, "label"])
    rep.update({
        "feat_n": int(len(f)),
        "lab_n": int(len(l)),
        "feat_threads": int(f[id_col].nunique()),
        "lab_threads": int(l[id_col].nunique()),
        "coverage_pct": pct(f[id_col].isin(set(l[id_col])).mean()),
        "feat_dtype": str(f[id_col].dtype),
        "lab_dtype": str(l[id_col].dtype)
    })
    # Echantillon d’intersection
    inter = pd.Series(f[id_col].unique())[:50]
    inter.to_csv(os.path.join(out_dir, "audit_threads_intersection_sample.csv"), index=False, header=[id_col])
    return rep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--messages", help="CSV messages (thread_id,text)", default=None)
    ap.add_argument("--embeddings", help="Parquet embeddings", default=None)
    ap.add_argument("--features", help="CSV features", default=None)
    ap.add_argument("--labels", help="CSV labels", default=None)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    report = {}

    if args.messages:
        rep, dfm, _ = audit_messages(args.messages, args.outdir)
        report["messages"] = rep

    if args.embeddings:
        rep, dfe = audit_parquet_embeddings(args.embeddings, args.outdir)
        report["embeddings"] = rep

    if args.features and args.labels:
        rep = audit_features_vs_labels(args.features, args.labels, args.outdir)
        report["feat_vs_labels"] = rep

    with open(os.path.join(args.outdir, "audit_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("[OK] audit_report.json écrit dans", args.outdir)
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
