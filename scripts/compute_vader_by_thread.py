# scripts/compute_vader_by_thread.py
# -*- coding: utf-8 -*-
import argparse, sys
import pandas as pd

def get_analyzer():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except Exception:
        return None

def score_texts(texts, analyzer):
    def score_one(s):
        s = "" if pd.isna(s) else str(s)
        if analyzer is None:
            return {"pos":0.0,"neu":0.0,"neg":0.0}
        sc = analyzer.polarity_scores(s)
        return {"pos":sc["pos"], "neu":sc["neu"], "neg":sc["neg"]}
    sv = texts.map(score_one).apply(pd.Series)
    sv = sv.rename(columns={"pos":"vader_pos","neu":"vader_neu","neg":"vader_neg"})
    return sv

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True, help="CSV messages avec thread_id et colonne texte")
    p.add_argument("--text-col", default="text")
    p.add_argument("--out", required=True, help="CSV de sortie: vader_*_mean par thread_id")
    args = p.parse_args()

    df = pd.read_csv(args.inp)
    if "thread_id" not in df.columns or args.text_col not in df.columns:
        raise SystemExit(f"Input must contain 'thread_id' and '{args.text_col}'")

    analyzer = get_analyzer()
    if analyzer is None:
        print("WARN: vaderSentiment non disponible -> sortie = zéros", file=sys.stderr)

    sv = score_texts(df[args.text_col], analyzer)
    tmp = pd.concat([df[["thread_id"]], sv], axis=1)
    agg = tmp.groupby("thread_id", as_index=False).mean(numeric_only=True)
    agg = agg.rename(columns={
        "vader_pos":"vader_pos_mean",
        "vader_neu":"vader_neu_mean",
        "vader_neg":"vader_neg_mean"
    })
    agg.to_csv(args.out, index=False)
    nz = {c: float((agg[c].ne(0).mean()*100)) for c in ["vader_pos_mean","vader_neu_mean","vader_neg_mean"]}
    print(f"WROTE {args.out} rows={len(agg)} nonzero%={nz}")

if __name__ == "__main__":
    main()
