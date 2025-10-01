import argparse, os, hashlib
import pandas as pd

def md5(path, chunk=2**20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()

def quick_profile(csv_path, max_rows=None):
    info = {"path": csv_path, "size_bytes": os.path.getsize(csv_path), "md5": md5(csv_path)}
    # charge colonnes minimales
    df = pd.read_csv(csv_path, usecols=["thread_id","text"], dtype={"thread_id":str}, nrows=max_rows)
    info["rows"] = len(df)
    info["uniq_threads"] = df["thread_id"].nunique()
    info["empty_texts"] = int((df["text"].astype(str).str.strip()=="").sum())
    info["mean_len"] = float(df["text"].astype(str).str.len().mean())
    return info, df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="CSV A (ex: data\\wikipedia\\afd\\afd_messages.csv)")
    ap.add_argument("--b", required=True, help="CSV B (ex: artifacts_xfer\\afd_messages.csv)")
    ap.add_argument("--emb-parquet", help="(optionnel) embeddings parquet pour croiser les thread_id")
    ap.add_argument("--sample", type=int, default=None, help="nrows to read for speed (None = full)")
    args = ap.parse_args()

    infoA, dfA = quick_profile(args.a, args.sample)
    infoB, dfB = quick_profile(args.b, args.sample)

    print("\n=== PROFILE A ==="); [print(f"{k}: {v}") for k,v in infoA.items()]
    print("\n=== PROFILE B ==="); [print(f"{k}: {v}") for k,v in infoB.items()]

    # chevauchement de threads
    setA, setB = set(dfA.thread_id), set(dfB.thread_id)
    inter = len(setA & setB)
    union = len(setA | setB)
    jacc = inter/union if union else 0.0
    print(f"\nThread overlap: intersection={inter} | union={union} | Jaccard={jacc:.4f}")

    # différences de contenu (échantillon)
    onlyA = list(setA - setB)[:10]
    onlyB = list(setB - setA)[:10]
    print(f"Example thread_ids only in A: {onlyA}")
    print(f"Example thread_ids only in B: {onlyB}")

    if args.emb_parquet:
        import pyarrow.parquet as pq
        t = pq.read_table(args.emb_parquet, columns=["thread_id"]).to_pandas()
        t["thread_id"] = t["thread_id"].astype(str)
        setE = set(t.thread_id.unique())
        covA = len(setA & setE) / (len(setA) or 1)
        covB = len(setB & setE) / (len(setB) or 1)
        print(f"\nCoverage vs embeddings ({args.emb_parquet}):")
        print(f"A: {covA*100:.2f}% | B: {covB*100:.2f}%")

if __name__ == "__main__":
    main()
