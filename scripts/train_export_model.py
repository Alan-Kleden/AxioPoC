# -*- coding: utf-8 -*-
import argparse, joblib, pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

def get_model(name: str):
    if name == "logreg": return LogisticRegression(max_iter=2000)
    if name == "rf":     return RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    raise ValueError(f"Unknown model {name}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feat", required=True)
    p.add_argument("--raw", required=True)
    p.add_argument("--cols", required=True)
    p.add_argument("--model", choices=["logreg","rf"], default="logreg")
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--scale", action="store_true")
    p.add_argument("--export-model", required=True)
    args = p.parse_args()

    cols = [c.strip() for c in args.cols.split(",")]
    dfX = pd.read_csv(args.feat)
    dfY = pd.read_csv(args.raw)
    if "thread_id" not in dfX or "thread_id" not in dfY:
        raise SystemExit("Both feat and raw CSV must contain a 'thread_id' column.")
    if "label" not in dfY:
        raise SystemExit("RAW must contain a 'label' column (0/1).")

    df = dfX.merge(dfY[["thread_id","label"]], on="thread_id", how="inner")
    miss = [c for c in cols if c not in df.columns]
    if miss: raise SystemExit(f"Feature columns missing: {miss}")

    X = df[cols].to_numpy()
    y = df["label"].to_numpy().astype(int)
    if len(np.unique(y)) < 2:
        print("WARNING: only one class present; logreg may fail.")

    scaler = StandardScaler() if args.scale else None
    Xs = scaler.fit_transform(X) if scaler is not None else X

    aucs, accs = [], []
    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
    for _ in range(args.reps):
        for tr, te in skf.split(Xs, y):
            m = get_model(args.model).fit(Xs[tr], y[tr])
            if hasattr(m, "predict_proba"):
                p1 = m.predict_proba(Xs[te])[:,1]
            else:
                s = m.decision_function(Xs[te]); p1 = (s - s.min())/(s.max()-s.min()+1e-9)
            aucs.append(roc_auc_score(y[te], p1))
            accs.append(accuracy_score(y[te], m.predict(Xs[te])))
    print(f"CV {args.model} | AUC mean±sd: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"CV {args.model} | ACC mean±sd: {np.mean(accs):.3f} ± {np.std(accs):.3f}")

    final_model = get_model(args.model).fit(Xs, y)
    joblib.dump({"model": final_model, "scaler": scaler, "cols": cols}, args.export_model)
    print(f"Model exported to: {args.export_model}")

if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-import argparse, joblib, pandas as pd, numpy as npfrom sklearn.linear_model import LogisticRegressionfrom sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifierfrom sklearn.preprocessing import StandardScalerfrom sklearn.model_selection import StratifiedKFoldfrom sklearn.metrics import roc_auc_score, accuracy_scoredef get_model(name:str):    if name == "logreg": return LogisticRegression(max_iter=2000)    if name == "rf":     return RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)    if name == "hgb":    return HistGradientBoostingClassifier(random_state=42)    raise ValueError(f"Unknown model {name}")def main():    p = argparse.ArgumentParser()    p.add_argument("--feat", required=True)    p.add_argument("--raw", required=True)    p.add_argument("--cols", required=True)    p.add_argument("--model", choices=["logreg","rf","hgb"], default="logreg")    p.add_argument("--cv", type=int, default=5)    p.add_argument("--reps", type=int, default=3)    p.add_argument("--scale", action="store_true")    p.add_argument("--export-model", required=True)    args = p.parse_args()    cols = [c.strip() for c in args.cols.split(",")]    dfX = pd.read_csv(args.feat)    dfY = pd.read_csv(args.raw)    if "thread_id" not in dfX or "thread_id" not in dfY:        raise SystemExit("Both feat and raw CSV must contain a 'thread_id' column.")    if "label" not in dfY:        raise SystemExit("RAW must contain a 'label' column (0/1).")    df = dfX.merge(dfY[["thread_id","label"]], on="thread_id", how="inner")    missing = [c for c in cols if c not in df.columns]    if missing:        raise SystemExit(f"Feature columns missing: {missing}")    X = df[cols].to_numpy()    y = df["label"].to_numpy().astype(int)    if len(np.unique(y)) < 2:        print("WARNING: only one class present; logreg may fail.")    scaler = StandardScaler() if args.scale else None    Xs = scaler.fit_transform(X) if scaler is not None else X    aucs, accs = [], []    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)    for _ in range(args.reps):        for tr, te in skf.split(Xs, y):            m = get_model(args.model).fit(Xs[tr], y[tr])            if hasattr(m, "predict_proba"):                p1 = m.predict_proba(Xs[te])[:,1]            else:                s = m.decision_function(Xs[te]); p1 = (s - s.min())/(s.max()-s.min()+1e-9)            aucs.append(roc_auc_score(y[te], p1))            accs.append(accuracy_score(y[te], m.predict(Xs[te])))    import numpy as _np    print(f"CV {args.model} | AUC mean±sd: {_np.mean(aucs):.3f} ± {_np.std(aucs):.3f}")    print(f"CV {args.model} | ACC mean±sd: {_np.mean(accs):.3f} ± {_np.std(accs):.3f}")    final_model = get_model(args.model).fit(Xs, y)    joblib.dump({"model": final_model, "scaler": scaler, "cols": cols}, args.export_model)    print(f"Model exported to: {args.export_model}")if __name__ == "__main__":    main()