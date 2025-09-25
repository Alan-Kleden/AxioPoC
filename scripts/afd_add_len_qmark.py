import pandas as pd
PREP = r"data/wikipedia/afd/afd_prepared.csv"
FEAT_IN  = r"artifacts_benchAfd_final_vader/features_eval.csv"
FEAT_OUT = r"artifacts_benchAfd_final_vader/features_eval_plus.csv"

df = pd.read_csv(PREP, usecols=["thread_id","text"])
df["thread_id"] = pd.to_numeric(df["thread_id"], errors="coerce").astype("int64")
df["len"] = df["text"].astype(str).str.len()
df["has_q"] = df["text"].astype(str).str.contains(r"\?", regex=True).astype(int)
agg = df.groupby("thread_id").agg(len_mean=("len","mean"), qmark_ratio=("has_q","mean")).reset_index()

f = pd.read_csv(FEAT_IN)
f["thread_id"] = pd.to_numeric(f["thread_id"], errors="raise").astype("int64")
g = f.merge(agg, on="thread_id", how="left")
g.to_csv(FEAT_OUT, index=False)
print("OK ->", FEAT_OUT, g.shape)
