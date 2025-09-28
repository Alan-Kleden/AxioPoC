#!/usr/bin/env python3
import argparse, json, glob, pandas as pd, os, datetime as dt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="Glob des JSON: ex. artifacts_xfer/metrics_*.json")
    ap.add_argument("--outcsv", required=True, help="Chemin du CSV consolidé")
    args = ap.parse_args()

    rows = []
    for path in glob.glob(args.inputs):
        with open(path, "r", encoding="utf-8") as f:
            try:
                obj = json.load(f)
            except Exception as e:
                print(f"[WARN] Skip {path}: {e}")
                continue
        m = obj.get("metrics", {})
        rows.append({
            "file": os.path.basename(path),
            "dataset": obj.get("dataset",""),
            "setting": obj.get("setting",""),
            "direction": obj.get("direction",""),
            "signature": obj.get("signature",""),
            "model": obj.get("model",""),
            "cv": obj.get("cv",""),
            "reps": obj.get("reps",""),
            "n": obj.get("n",""),
            "auc_mean": m.get("auc_mean",""),
            "auc_sd": m.get("auc_sd",""),
            "acc_mean": m.get("acc_mean",""),
            "acc_sd": m.get("acc_sd",""),
            "timestamp": obj.get("timestamp","")
        })
    df = pd.DataFrame(rows)
    if not len(df):
        print("[INFO] Aucun fichier trouvé.")
        return
    df.sort_values(["setting","direction","signature","dataset","model"], inplace=True)
    os.makedirs(os.path.dirname(args.outcsv), exist_ok=True)
    df.to_csv(args.outcsv, index=False)
    print(f"[OK] Consolidé {len(df)} runs -> {args.outcsv}")
    print(df[["setting","direction","signature","dataset","auc_mean","acc_mean"]])
    # petit mémo horodaté
    with open(args.outcsv.replace(".csv",".txt"),"w",encoding="utf-8") as g:
        g.write(f"Consolidation: {dt.datetime.now().isoformat()}\n{args.outcsv}\n")

if __name__ == "__main__":
    main()
