# -*- coding: utf-8 -*-
import pandas as pd, re
from pathlib import Path

SRC = Path("data/wikipedia/afd/afd_with_outcome.csv")
DST = Path("data/wikipedia/afd/afd_prepared.csv")

if not SRC.exists():
    raise SystemExit(f"Introuvable: {SRC}")

df = pd.read_csv(SRC)

# Normalisation colonnes -> thread_id, text, timestamp, outcome
need = ["thread_id", "text", "timestamp", "outcome"]
lc = {c.lower(): c for c in df.columns}
aliases = {
    "thread_id": ["thread_id","thread","conversation_id","discussion_id","page_id","conv_id"],
    "text":      ["text","body","message","content"],
    "timestamp": ["timestamp","time","created_utc","date"],
    "outcome":   ["outcome","decision","result","final_decision","label"]
}
rename = {}
for target, opts in aliases.items():
    for o in opts:
        if o in lc:
            rename[lc[o]] = target
            break
df = df.rename(columns=rename)
missing = [c for c in need if c not in df.columns]
if missing:
    raise SystemExit(f"Colonnes manquantes: {missing}")

# Nettoyage texte
df = df.dropna(subset=["text"])
df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

# Timestamps: fallback ordinal par thread si NaN / non-numérique
if df["timestamp"].isna().any():
    df["timestamp"] = df.groupby("thread_id").cumcount()
df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
df.loc[df["timestamp"].isna(), "timestamp"] = df.groupby("thread_id").cumcount()

# Binarisation outcome: Delete-like = 1 ; sinon 0
def to_bin(x) -> int:
    s = str(x).strip().lower() if pd.notna(x) else ""
    if not s:
        return 0
    if re.search(r"\b(delete|speedy\s*delete|prod|salt)\b", s):
        return 1
    if re.search(r"\b(keep|speedy\s*keep|redirect|merge|userfy|transwiki|no\s*consensus|withdrawn|relist)\b", s):
        return 0
    if "del" in s and "keep" not in s:
        return 1
    return 0

df["outcome_bin"] = df["outcome"].map(to_bin).astype(int)

# Sauvegarde
DST.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(DST, index=False, encoding="utf-8")
print(f"OK -> {DST} | n={len(df)} | p_delete={df['outcome_bin'].mean():.3f}")

