#!/usr/bin/env python
import argparse, json, pathlib, sys
import pandas as pd

KEEP = ["sample_frac","min_msgs","B_effective","n_threads_used",
        "cos_min","cos_p05","cos_p25","cos_mean","cos_median","cos_p75","cos_p95","cos_max"]

def load_one(path: pathlib.Path):
    try:
        j = json.loads(path.read_text(encoding="utf-8"))
        row = {"file": path.name}
        for k in KEEP:
            row[k] = j.get(k, None)
        return row
    except Exception as e:
        return {"file": path.name, "error": str(e)}

def main():
    ap = argparse.ArgumentParser(description="Compare multiple axis_stability*.json runs.")
    ap.add_argument("--jsons", nargs="+", required=True, help="List of JSON files to compare")
    ap.add_argument("--outcsv", help="Optional CSV to write")
    args = ap.parse_args()

    rows = []
    for jpath in args.jsons:
        p = pathlib.Path(jpath)
        if not p.exists():
            print(f"[WARN] Missing: {p}", file=sys.stderr)
            continue
        rows.append(load_one(p))

    if not rows:
        print("[ERR] No valid inputs.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    num_cols = [c for c in KEEP if c in df.columns]
    df_sorted = df.sort_values(by=["min_msgs","sample_frac","file"], na_position="last")
    print(df_sorted.to_string(index=False))

    if args.outcsv:
        pathlib.Path(args.outcsv).parent.mkdir(parents=True, exist_ok=True)
        df_sorted.to_csv(args.outcsv, index=False)
        print(f"[OK] CSV -> {args.outcsv}")

if __name__ == "__main__":
    main()
