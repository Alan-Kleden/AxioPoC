import argparse, csv, statistics as st
from collections import defaultdict
from math import isnan
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

def load_y(raw_path):
    y = {}
    with open(raw_path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            tid = row['thread_id']
            o = row['outcome']
            if o == '': 
                continue
            y[tid] = int(o)
    return y

def load_X(feat_path, cols):
    X_map = {}
    with open(feat_path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            tid = row['thread_id']
            vals = []
            ok = True
            for c in cols:
                if c not in row or row[c] == '':
                    ok = False; break
                vals.append(float(row[c]))
            if ok:
                X_map[tid] = vals
    return X_map

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--feat', required=True)
    ap.add_argument('--raw',  required=True)
    ap.add_argument('--cols', required=True, help='comma-separated feature names')
    ap.add_argument('--balanced', action='store_true', help='use balanced class weights')
    ap.add_argument('--scale', action='store_true', help='standardize features before LR')
    args = ap.parse_args()

    cols = [c.strip() for c in args.cols.split(',') if c.strip()]
    y_map = load_y(args.raw)
    X_map = load_X(args.feat, cols)

    # align
    keys = sorted(set(X_map.keys()) & set(y_map.keys()))
    X = np.array([X_map[k] for k in keys], dtype=float)
    y = np.array([y_map[k] for k in keys], dtype=int)

    print(f'Threads: {len(y)} | Positives: {int(y.sum())} ({y.mean()*100:.1f}%)')
    print('Cols:', cols)

    # option: scaling
    if args.scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # LR
    cw = 'balanced' if args.balanced else None
    lr = LogisticRegression(max_iter=2000, solver='liblinear', class_weight=cw)

    aucs, accs = [], []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in skf.split(X, y):
        lr.fit(X[tr], y[tr])
        p = lr.predict_proba(X[te])[:,1]
        aucs.append(roc_auc_score(y[te], p))
        accs.append(accuracy_score(y[te], (p>=0.5).astype(int)))

    mA, sA = st.mean(aucs), st.pstdev(aucs)
    mC, sC = st.mean(accs), st.pstdev(accs)
    print(f'AUC meansd: {mA:.3f}  {sA:.3f}')
    print(f'ACC meansd: {mC:.3f}  {sC:.3f}')
