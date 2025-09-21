# scripts/check_csv.py
import sys
from pathlib import Path
import pandas as pd
import numpy as np

def main(path):
    p = Path(path)
    if not p.exists():
        print(f"ERROR: file not found: {p}")
        sys.exit(1)

    df = pd.read_csv(p)
    print(f"\nLoaded: {p}  |  rows={len(df)}  cols={list(df.columns)}")

    # 1) colonnes minimales
    required = {"thread_id","stance","outcome"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: missing columns: {missing}")
        sys.exit(1)

    # 2) parser timestamp (si présent)
    if "timestamp" in df.columns:
        df["timestamp_parsed"] = pd.to_datetime(df["timestamp"], errors="coerce")
        n_NaT = df["timestamp_parsed"].isna().sum()
        print(f"timestamp: present | unparsable = {n_NaT}")
        if n_NaT > 0:
            print("  -> Inspect a few unparsable timestamps:")
            print(df.loc[df["timestamp_parsed"].isna(), "timestamp"].head().to_string(index=False))

    # 3) domaines stance/outcome
    def norm_stance(x):
        try:
            v = float(x)
            return 1 if v>0 else (-1 if v<0 else 0)
        except Exception:
            sx = str(x).strip().lower()
            if any(k in sx for k in ["support","pour","+"]): return 1
            if any(k in sx for k in ["oppose","contre","-"]): return -1
            if any(k in sx for k in ["neutral","neutre","abstain"]): return 0
            return 0

    def norm_outcome(x):
        try:
            return 1 if float(x)>0 else 0
        except Exception:
            sx = str(x).strip().lower()
            return 1 if sx in {"accept","accepted","success","yes","true","1"} else 0

    df["stance_n"] = df["stance"].map(norm_stance)
    df["outcome_n"] = df["outcome"].map(norm_outcome)

    bad_stance = df.loc[~df["stance_n"].isin([-1,0,1])]
    bad_outcome = df.loc[~df["outcome_n"].isin([0,1])]
    print(f"stance domain ok? {bad_stance.empty} | outcome domain ok? {bad_outcome.empty}")

    # 4) cohérence outcome par thread
    incoh = (df.groupby("thread_id")["outcome_n"]
               .nunique()
               .reset_index(name="n_labels"))
    incoh = incoh[incoh["n_labels"]>1]
    if not incoh.empty:
        print("\nWARNING: threads with mixed outcome labels:")
        print(incoh.to_string(index=False))
    else:
        print("Outcome per thread is consistent (one label/thread).")

    # 5) résumé par thread & classes
    summ = (df.groupby("thread_id")
              .agg(n_msgs=("stance_n","size"),
                   outcome=("outcome_n","first"),
                   t_min=("timestamp_parsed","min") if "timestamp_parsed" in df.columns else ("stance_n","size"),
                   t_max=("timestamp_parsed","max") if "timestamp_parsed" in df.columns else ("stance_n","size"))
              .reset_index())
    print("\nBy-thread summary (first 10):")
    print(summ.head(10).to_string(index=False))

    # classes
    cls = df.drop_duplicates("thread_id")[["thread_id","outcome_n"]]["outcome_n"].value_counts().to_dict()
    print(f"\nClass distribution (by thread): {cls}  (need both 0 and 1 for CV)")

    # 6) option: sauvegarder une version "clean" avec timestamp ISO
    if "timestamp_parsed" in df.columns:
        out = p.parent / (p.stem + "_clean.csv")
        df_out = df[["thread_id","timestamp_parsed","stance_n","outcome_n"]].rename(
            columns={"timestamp_parsed":"timestamp","stance_n":"stance","outcome_n":"outcome"}
        )
        df_out.to_csv(out, index=False)
        print(f"\nWrote cleaned CSV to: {out}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_csv.py path/to/data.csv")
        sys.exit(1)
    main(sys.argv[1])
