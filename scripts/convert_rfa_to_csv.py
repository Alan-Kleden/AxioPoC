# scripts/convert_rfa_to_csv.py
import gzip, re, csv
from pathlib import Path

IN  = Path("data/wiki-RfA.txt.gz")
OUT = Path("data/rfa.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Chaque entrée est séparée par une ligne vide; les champs sont préfixés (SRC/TGT/VOT/RES/YEA/DAT/TXT)
def parse_entry(block: str):
    # on segmente sur " <KEY>:" pour éviter de casser les commentaires TXT
    parts = re.split(r'\s(?=(?:SRC|TGT|VOT|RES|YEA|DAT|TXT):)', block.strip())
    data = {}
    for p in parts:
        if ':' in p:
            k, v = p.split(':', 1)
            data[k] = v.strip()
    return data

rows = []
with gzip.open(IN, 'rt', encoding='utf-8', errors='ignore') as f:
    content = f.read()
for block in re.split(r'\n\s*\n', content):  # entrée = bloc séparé par ligne vide
    if not block.strip():
        continue
    d = parse_entry(block)
    # Champs indispensables
    if not all(k in d for k in ("TGT","VOT","RES","YEA","DAT")):
        continue
    # stance / outcome
    try:
        stance = int(d["VOT"])   # -1 / 0 / +1
        outcome = 1 if int(d["RES"]) > 0 else 0  # 1=accepted, 0=rejected
    except ValueError:
        continue
    # thread_id simple : candidat + année (suffisant pour un PoC)
    thread_id = f'{d["TGT"]}_{d["YEA"]}'
    # timestamp : on laisse la chaîne DAT (pandas la parsera)
    timestamp = d["DAT"]
    rows.append((thread_id, timestamp, stance, outcome))

with OUT.open("w", newline="", encoding="utf-8") as g:
    w = csv.writer(g)
    w.writerow(["thread_id","timestamp","stance","outcome"])
    w.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUT}")
