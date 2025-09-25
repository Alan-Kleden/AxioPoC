import pandas as pd, os, sys

FEAT = r"artifacts_benchAfd_final_vader/features.csv"          # threads présents côté X
RAW  = r"data/wikipedia/afd/afd_prepared.csv"                  # source des labels
OUT  = r"data/wikipedia/afd/afd_balanced_for_eval.csv"         # nouveau raw aligné

# 1) threads présents dans les features (type str)
f = pd.read_csv(FEAT, usecols=["thread_id"])
f["thread_id"] = f["thread_id"].astype(str)
fid = set(f["thread_id"].unique())
if not fid: sys.exit("Alerte: aucun thread_id dans features.csv")

# 2) label majoritaire par thread depuis le prepared (type str)
df = pd.read_csv(RAW, usecols=["thread_id","outcome_bin"])
df["thread_id"] = df["thread_id"].astype(str)
lab = df.groupby("thread_id")["outcome_bin"].mean().round().astype(int).reset_index()
lab = lab.rename(columns={"outcome_bin":"outcome"})

# 3) intersection stricte avec les features
lab = lab[lab["thread_id"].isin(fid)]

# 4) équilibrage 1:1 (optionnel mais utile pour AUC stable)
counts = lab["outcome"].value_counts().to_dict()
n_pos, n_neg = counts.get(1,0), counts.get(0,0)
if n_pos == 0 or n_neg == 0:
    sys.exit(f"Classes non équilibrables après alignement (pos={n_pos}, neg={n_neg}).")
n = min(n_pos, n_neg)
pos = lab[lab.outcome==1].sample(n=n, random_state=42)
neg = lab[lab.outcome==0].sample(n=n, random_state=42)
bal = pd.concat([pos,neg]).sample(frac=1, random_state=42).reset_index(drop=True)

# 5) sortie finale: colonnes exactement 'thread_id,outcome', encodage UTF-8
os.makedirs(os.path.dirname(OUT), exist_ok=True)
bal.to_csv(OUT, index=False, encoding="utf-8")
print("OK ->", OUT, "| counts:", bal['outcome'].value_counts().to_dict(), "| n_threads=", len(bal))
