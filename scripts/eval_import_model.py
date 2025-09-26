# -*- coding: utf-8 -*-
import argparse, joblib, pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feat", required=True)
    p.add_argument("--raw", required=True)
    p.add_argument("--import-model", required=True)
    p.add_argument("--cols", required=True)
    args = p.parse_args()

    bundle = joblib.load(args.import_model)
    model, scaler, trained_cols = bundle["model"], bundle["scaler"], bundle["cols"]
    cols = [c.strip() for c in args.cols.split(",")]
    if cols != trained_cols:
        raise SystemExit(f"Column order mismatch.\nProvided: {cols}\nTrained:  {trained_cols}")

    dfX = pd.read_csv(args.feat)
    dfY = pd.read_csv(args.raw)
    if "thread_id" not in dfX or "thread_id" not in dfY:
        raise SystemExit("Both feat and raw must contain 'thread_id'.")
    merge_cols = ["thread_id","label"] if "label" in dfY.columns else ["thread_id"]
    df = dfX.merge(dfY[merge_cols], on="thread_id", how="inner")

    X = df[cols].to_numpy()
    if scaler is not None:
        X = scaler.transform(X)

    out = {"n": int(len(df))}
    if "label" in df.columns:
        y = df["label"].to_numpy().astype(int)
        if hasattr(model, "predict_proba"):
            p1 = model.predict_proba(X)[:,1]
        else:
            s = model.decision_function(X); p1 = (s - s.min())/(s.max()-s.min()+1e-9)
        out["AUC"] = float(roc_auc_score(y, p1))
        out["ACC"] = float(np.mean(model.predict(X) == y))
    print("Eval:", out)

if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-import argparse, joblib, pandas as pd, numpy as npfrom sklearn.metrics import roc_auc_score, accuracy_scoredef main():    p = argparse.ArgumentParser()    p.add_argument("--feat", required=True)    p.add_argument("--raw", required=True)    p.add_argument("--import-model", required=True)    p.add_argument("--cols", required=True)    args = p.parse_args()    bundle = joblib.load(args.import_model)    model  = bundle["model"]; scaler = bundle["scaler"]; trained_cols = bundle["cols"]    cols = [c.strip() for c in args.cols.split(",")]    if cols != trained_cols:        raise SystemExit(f"Column order mismatch.\nProvided: {cols}\nTrained:  {trained_cols}")    dfX = pd.read_csv(args.feat)    dfY = pd.read_csv(args.raw)    if "thread_id" not in dfX or "thread_id" not in dfY:        raise SystemExit("Both feat and raw must contain 'thread_id'.")    merge_cols = ["thread_id","label"] if "label" in dfY.columns else ["thread_id"]    df = dfX.merge(dfY[merge_cols], on="thread_id", how="inner")    X = df[cols].to_numpy()    if scaler is not None: X = scaler.transform(X)    out = {"n": int(len(df))}    if "label" in df.columns:        y = df["label"].to_numpy().astype(int)        if hasattr(model, "predict_proba"):            p1 = model.predict_proba(X)[:,1]        else:            s = model.decision_function(X); p1 = (s - s.min())/(s.max()-s.min()+1e-9)        out["AUC"] = float(roc_auc_score(y, p1))        out["ACC"] = float(np.mean((model.predict(X)==y)))    print("Eval:", out)if __name__ == "__main__":    main()