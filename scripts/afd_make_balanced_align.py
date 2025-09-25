import pandas as pd, os, sys

FEAT = r"artifacts_benchAfd_final_vader/features.csv"
RAW  = r"data/wikipedia/afd/afd_prepared.csv"
OUT  = r"data/wikipedia/afd/afd_balanced_for_features.csv"

# Charger les IDs présents dans les features (str)
f = pd.read_csv(FEAT, usecols=["thread_id"])
f["thread_id"] = f["thread_id"].astype(str)
fid = set(f["thread_id"].unique())
if not fid:
    sys.exit("Alerte: aucun thread_id dans features.csv")

# Charger les labels bruts et agréger par thread (majorité), forcer str
df = pd.read_csv(RAW, usecols=["thread_id","outcome_bin"])
df["thread_id"] = df["thread_id"].astype(str)
lab = df.groupby("thread_id")["outcome_bin"].mean().round().astype(int).reset_index()

# Garder uniquement les threads présents dans features
lab = lab[lab["thread_id"].isin(fid)]
cnt = lab["outcome_bin"].value_counts().to_dict()
n_pos, n_neg = cnt.get(1,0), cnt.get(0,0)
if n_pos == 0 or n_neg == 0:
    sys.exit(f"Classes non équilibrables après alignement (pos={n_pos}, neg={n_neg}).")

# Équilibrage 1:1
n = min(n_pos, n_neg)
pos = lab[lab.outcome_bin==1].sample(n=n, random_state=42)
neg = lab[lab.outcome_bin==0].sample(n=n, random_state=42)
bal = pd.concat([pos,neg]).sample(frac=1, random_state=42).reset_index(drop=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
bal.to_csv(OUT, index=False, encoding="utf-8")
print("OK ->", OUT, "| counts:", bal['outcome_bin'].value_counts().to_dict(), "| n_threads=", len(bal))
