import os, pandas as pd
os.makedirs(r"artifacts_benchAfd_final_vader", exist_ok=True)
f20 = pd.read_csv(r"artifacts_benchAfd_vader_w20/features.csv").add_suffix("_w20"); f20=f20.rename(columns={"thread_id_w20":"thread_id"})
f10 = pd.read_csv(r"artifacts_benchAfd_vader_w10/features.csv").add_suffix("_w10"); f10=f10.rename(columns={"thread_id_w10":"thread_id"})
features = pd.merge(f20, f10, on="thread_id", how="inner")
features.to_csv(r"artifacts_benchAfd_final_vader/features.csv", index=False)
print("OK -> artifacts_benchAfd_final_vader/features.csv", features.shape)
