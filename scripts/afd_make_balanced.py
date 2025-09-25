import pandas as pd, os, sys

RAW = r"data/wikipedia/afd/afd_prepared.csv"
OUT = r"data/wikipedia/afd/afd_balanced.csv"

if not os.path.exists(RAW):
    sys.exit(f"Introuvable: {RAW}")

# Label par thread = majorité (moyenne arrondie)
df = pd.read_csv(RAW, usecols=["thread_id","outcome_bin"])
lab = df.groupby("thread_id")["outcome_bin"].mean().round().astype(int).reset_index()

counts = lab["outcome_bin"].value_counts().to_dict()
n_pos = counts.get(1, 0)
n_neg = counts.get(0, 0)
if n_pos == 0 or n_neg == 0:
    sys.exit(f"Alerte: classes déséquilibrées irréparables (pos={n_pos}, neg={n_neg}). Vérifie outcome_bin.")

n = min(n_pos, n_neg)
pos = lab[lab.outcome_bin==1].sample(n=n, random_state=42)
neg = lab[lab.outcome_bin==0].sample(n=n, random_state=42)
bal = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
bal.to_csv(OUT, index=False, encoding="utf-8")

print(f"OK -> {OUT} | counts:", bal['outcome_bin'].value_counts().to_dict(), "| n_threads=", len(bal))
