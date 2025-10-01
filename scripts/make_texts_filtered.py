# C:\AxioPoC\scripts\make_texts_filtered.py
import re, sys
import pandas as pd
from pathlib import Path
import argparse

PATTERNS = [
    r'^\s*\*?\s*VOTE\b',                 # lignes de vote
    r'\bper\s+nom\b',                    # "per nom"
    r'\[\[\s*WP:[A-Z][A-Z0-9_-]+',       # jargon WP:
    r'\[\[\s*Wikipedia:Articles for deletion/',  # liens AfD
    r'\{\{[A-Za-z]+-?vote',              # templates vote
]

def compile_patterns(pats):
    return [re.compile(p, flags=re.IGNORECASE) for p in pats]

def keep_text(t:str, regs):
    s = (t or "").strip()
    if not s: 
        return False
    for rg in regs:
        if rg.search(s):
            return False
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp",  required=True, help="CSV with columns: thread_id,text")
    ap.add_argument("--out",  required=True, help="Output filtered CSV")
    ap.add_argument("--idcol", default="thread_id")
    ap.add_argument("--textcol", default="text")
    args = ap.parse_args()

    regs = compile_patterns(PATTERNS)
    df = pd.read_csv(args.inp, usecols=[args.idcol, args.textcol])
    before = len(df)
    df = df[df[args.textcol].astype(str).map(lambda x: keep_text(x, regs))]
    after = len(df)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[OK] filtered texts -> {args.out} | kept {after}/{before} rows ({after/before:.2%})")

if __name__ == "__main__":
    main()
