import argparse, csv, statistics as st
import numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

def load_y(raw_path):
    y = {}
    with open(raw_path, encoding='utf-8', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            tid = row['thread_id']
            if tid not in y:
                y[tid] = int(row['outcome'])
    return y

def load_X(feat_path, cols):
    Xmap = {}
    with open(feat_path, encoding='utf-8', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            vals=[]
            ok=True
            for c in cols:
                if c not in row or row[c] in ('', None):
                    ok=False; break
                vals.append(float(row[c]))
            if ok:
                Xmap[row['thread_id']] = vals
    return Xmap

parser = argparse.ArgumentParser()
parser.add_argument('--feat', required=True)
parser.add_argument('--raw', required=True)
parser.add_argument('--cols', required=True)  # comma list
parser.add_argument('--model', choices=['logreg','rf','hgb'], default='logreg')
args = parser.parse_args()

cols = [c.strip() for c in args.cols.split(',') if c.strip()]
y_map = load_y(args.raw)
X_map = load_X(args.feat, cols)
common = sorted(set(X_map.keys()) & set(y_map.keys()))
X = np.array([X_map[t] for t in common], dtype=float)
y = np.array([y_map[t] for t in common], dtype=int)

if args.model == 'logreg':
    clf = make_pipeline(StandardScaler(with_mean=True, with_std=True),
                        LogisticRegression(max_iter=1000, solver='liblinear'))
elif args.model == 'rf':
    clf = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=10, random_state=42)
else:
    clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08,
                                         max_bins=255, min_samples_leaf=20,
                                         l2_regularization=0.0, random_state=42)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc, acc = [], []
for tr, te in skf.split(X, y):
    clf.fit(X[tr], y[tr])
    proba = (clf.predict_proba(X[te])[:,1]
             if hasattr(clf, 'predict_proba')
             else clf.predict(X[te]))  # HGB has predict_proba; fallback just in case
    if proba.ndim==1 and set(np.unique(proba))<= {0,1}:  # fallback safety
        proba = proba.astype(float)
    pred = (proba >= 0.5).astype(int) if proba.ndim==1 else (proba >= 0.5)
    auc.append(roc_auc_score(y[te], proba))
    acc.append(accuracy_score(y[te], pred))

print(f'Threads: {len(y)} | Positives: {int(y.sum())} ({100*y.mean():.1f}%)')
print('Cols:', cols)
name = args.model.upper()
print(f'{name} | AUC meansd: {st.mean(auc):.3f}  {st.pstdev(auc):.3f}')
print(f'{name} | ACC meansd: {st.mean(acc):.3f}  {st.pstdev(acc):.3f}')
