import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
import numpy as np

FEAT = r"artifacts_benchAfd_final_vader/features_eval.csv"
RAW  = r"data/wikipedia/afd/afd_eval.csv"
COLS = ["R_mean_w20","R_last_w20","R_slope_w20","R_mean_w10","R_last_w10","R_slope_w10"]

X = pd.read_csv(FEAT, usecols=["thread_id"]+COLS)
y = pd.read_csv(RAW)
df = X.merge(y, on="thread_id", how="inner")
X, y = df[COLS].values, df["outcome"].values

Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
clf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
clf.fit(Xtr,ytr)
auc = roc_auc_score(yte, clf.predict_proba(Xte)[:,1])
print("AUC holdout:", round(auc,3))

pi = permutation_importance(clf, Xte, yte, n_repeats=10, random_state=42, n_jobs=-1)
order = np.argsort(-pi.importances_mean)
for idx in order:
    print(COLS[idx], "=>", round(pi.importances_mean[idx],4))
