import pandas as pd
a = pd.read_csv(r"data/wikipedia/afd/afd.csv", low_memory=False)
t = pd.read_csv(r"data/wikipedia/afd/afd_threads.csv")  # thread_id,outcome
m = a.merge(t, on="thread_id", how="left", suffixes=("","_thread"))
# outcome final = outcome si non vide, sinon outcome_thread
m["outcome"] = (m["outcome"].fillna("").astype(str))
mask = m["outcome"].str.strip()==""
m.loc[mask, "outcome"] = m.loc[mask, "outcome_thread"].fillna("")
m = m.drop(columns=[c for c in ["outcome_thread"] if c in m.columns])
m.to_csv(r"data/wikipedia/afd/afd_with_outcome.csv", index=False, encoding="utf-8")
print("OK -> data/wikipedia/afd/afd_with_outcome.csv | n=", len(m))
