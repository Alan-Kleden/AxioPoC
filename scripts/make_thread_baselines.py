# -*- coding: utf-8 -*-
"""
make_thread_baselines.py
- Construit des baselines par fil depuis un CSV textes (thread_id, text)
- long_mean_chars, long_mean_tokens, vader_mean (compound)
- Joint aux labels

Usage:
python make_thread_baselines.py ^
  --texts "C:\AxioPoC\artifacts_xfer\afd_messages_filtered.csv" ^
  --labels "C:\AxioPoC\data\wikipedia\afd\afd_eval.csv" ^
  --min-msgs 5 ^
  --out "C:\AxioPoC\REPORTS\thread_baselines.csv"
"""
import argparse
from pathlib import Path
import pandas as pd

def safe_vader(series_text):
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        import nltk
        try:
            _ = SentimentIntensityAnalyzer()
        except:
            nltk.download('vader_lexicon')
            _ = SentimentIntensityAnalyzer()
        sia = _
        vals = []
        for t in series_text.fillna(""):
            try:
                vals.append(sia.polarity_scores(str(t))["compound"])
            except:
                vals.append(0.0)
        return pd.Series(vals, index=series_text.index, dtype=float)
    except Exception:
        # si nltk indisponible -> baseline neutre
        return pd.Series([0.0]*len(series_text), index=series_text.index, dtype=float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--thread-col", default="thread_id")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--min-msgs", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.texts, usecols=[args.thread-col, args.text-col])
    df[args.thread-col] = df[args.thread-col].astype("int64")
    txt = df[args.text-col].astype(str)
    lens_chars = txt.str.len().astype("int64")
    lens_tokens = txt.str.split().apply(len).astype("int64")
    vader = safe_vader(txt)

    df_feat = pd.DataFrame({
        "thread_id": df[args.thread-col].values,
        "len_chars": lens_chars.values,
        "len_tokens": lens_tokens.values,
        "vader": vader.values
    })
    agg = df_feat.groupby("thread_id", as_index=False).agg(
        long_mean_chars=("len_chars","mean"),
        long_mean_tokens=("len_tokens","mean"),
        vader_mean=("vader","mean"),
        n_msgs=("vader","size")
    )
    agg = agg[agg["n_msgs"]>=args.min_msgs].copy()

    labels = pd.read_csv(args.labels, usecols=["thread_id","label"])
    labels["thread_id"] = labels["thread_id"].astype("int64")
    labels["label"] = labels["label"].astype(int)

    out = agg.merge(labels, on="thread_id", how="inner")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[OK] baselines -> {args.out} | rows={len(out)}")
if __name__ == "__main__":
    main()
