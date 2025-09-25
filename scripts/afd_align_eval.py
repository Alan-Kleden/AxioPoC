import pandas as pd, os, sys

FEAT_IN   = r"artifacts_benchAfd_final_vader/features.csv"
RAW_SRC   = r"data/wikipedia/afd/afd_prepared.csv"              # contient outcome_bin par message
RAW_OUT   = r"data/wikipedia/afd/afd_eval.csv"                  # -> thread_id,outcome (0/1), aligné/intersecté/équilibré
FEAT_OUT  = r"artifacts_benchAfd_final_vader/features_eval.csv" # -> features filtrés à l'intersection

# ----- 1) Charger features et normaliser IDs -> int64
f = pd.read_csv(FEAT_IN)
try:
    f["thread_id"] = pd.to_numeric(f["thread_id"], errors="raise").astype("int64")
except Exception as e:
    sys.exit(f"[features] Impossible de convertir thread_id en int : {e}")
f_ids = set(f["thread_id"].unique())
print(f"[features] n_rows={len(f)} | n_threads={len(f_ids)}")

# ----- 2) Construire labels par thread depuis le prepared (majorité) + normaliser IDs
df = pd.read_csv(RAW_SRC, usecols=["thread_id","outcome_bin"])
try:
    df["thread_id"] = pd.to_numeric(df["thread_id"], errors="raise").astype("int64")
except Exception as e:
    sys.exit(f"[raw] Impossible de convertir thread_id en int : {e}")

lab = (df.groupby("thread_id")["outcome_bin"].mean()
         .round().astype(int).reset_index()
         .rename(columns={"outcome_bin":"outcome"}))
r_ids = set(lab["thread_id"].unique())
print(f"[raw] n_threads={len(r_ids)} | counts={lab['outcome'].value_counts().to_dict()}")

# ----- 3) Intersection stricte des IDs
ids = f_ids.intersection(r_ids)
if not ids:
    sys.exit("[align] Intersection vide entre features et labels.")
lab = lab[lab["thread_id"].isin(ids)]
f   = f[f["thread_id"].isin(ids)]
print(f"[align] intersection n_threads={len(ids)}")

# ----- 4) Équilibrage 1:1 après alignement
counts = lab["outcome"].value_counts().to_dict()
n_pos, n_neg = counts.get(1,0), counts.get(0,0)
if n_pos==0 or n_neg==0:
    sys.exit(f"[balance] Impossible d'équilibrer (pos={n_pos}, neg={n_neg}).")
n = min(n_pos, n_neg)
pos = lab[lab.outcome==1].sample(n=n, random_state=42)
neg = lab[lab.outcome==0].sample(n=n, random_state=42)
lab_bal = pd.concat([pos,neg]).sample(frac=1, random_state=42).reset_index(drop=True)

# ----- 5) Restreindre features à ces threads équilibrés
keep_ids = set(lab_bal["thread_id"].unique())
f = f[f["thread_id"].isin(keep_ids)]
print(f"[final] labels counts={lab_bal['outcome'].value_counts().to_dict()} | n_threads={len(keep_ids)} | feat_rows={len(f)}")

# ----- 6) Sauvegarde
os.makedirs(os.path.dirname(RAW_OUT), exist_ok=True)
os.makedirs(os.path.dirname(FEAT_OUT), exist_ok=True)
lab_bal.to_csv(RAW_OUT, index=False, encoding="utf-8")         # cols: thread_id,outcome
f.to_csv(FEAT_OUT, index=False, encoding="utf-8")              # mêmes colonnes features que l'entrée, filtrées
print(f"OK -> {RAW_OUT}")
print(f"OK -> {FEAT_OUT}")
