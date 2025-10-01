# scripts/mask_wp_jargon_and_reproject.py
# But: produire un CSV "afd_messages_masked.csv" où les marqueurs WP/JARGON
#      sont neutralisés pour une lecture humaine sans biais.
#
# Usage minimal (mask-only):
#   python scripts/mask_wp_jargon_and_reproject.py \
#     --texts "G:\Mon Drive\AxioPoC\artifacts_xfer\afd_messages.csv" \
#     --outdir C:\AxioPoC\REPORTS\face_validity_masked
#
# Sorties:
#   C:\AxioPoC\REPORTS\face_validity_masked\afd_messages_masked.csv
#   + un petit log de stats sur les remplacements.

import re
import argparse
from pathlib import Path
import pandas as pd

PATTERNS = [
    r"\[\[WP:[A-Z0-9_-]+\]\]",   # [[WP:GNG]], [[WP:SPAM]], [[WP:CORP]]...
    r"\bWP:[A-Z0-9_-]+\b",       # WP:GNG sans crochets
    r"\bper\s+nom\b",            # "per nom"
    r"\bper\s+policy\b",
    r"\bsee\s+\[\[WP:[A-Z0-9_-]+\]\]\b",
    r"\bper\s+\[\[WP:[A-Z0-9_-]+\]\]\b",
]
REPLACEMENTS = ["[WP_RULE]"] * len(PATTERNS)

def mask_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    out = s
    for pat, rep in zip(PATTERNS, REPLACEMENTS):
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True, help="CSV avec colonnes: thread_id,text")
    ap.add_argument("--outdir", required=True, help="Dossier de sortie")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "afd_messages_masked.csv"
    log_txt = outdir / "mask_stats.txt"

    df = pd.read_csv(args.texts, usecols=["thread_id", "text"])
    before = df["text"].astype(str).fillna("").copy()

    df["text"] = before.apply(mask_text)

    # Stats simples
    changed = (before.values != df["text"].values).sum()
    total = len(df)
    with open(log_txt, "w", encoding="utf-8") as f:
        f.write(f"rows={total}\nchanged={changed}\nchanged_ratio={changed/total:.4f}\n")

    df.to_csv(out_csv, index=False)
    print(f"[OK] Masked CSV -> {out_csv}")
    print(f"[stats] -> {log_txt}")

if __name__ == "__main__":
    main()
