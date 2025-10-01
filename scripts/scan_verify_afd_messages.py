#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scan & Verify AfD messages CSV candidates.
- Cherche des fichiers (pattern par défaut: *afd*messages*.csv) sous des racines données.
- Pour chaque fichier: calcule stats clés (rows, n_threads, msgs/thread, textes vides)
  et la couverture vs afd_eval.csv (labels).
- Écrit un résumé CSV + affiche les meilleurs candidats.
"""

import argparse, os, sys, hashlib, datetime
from pathlib import Path
from collections import Counter
import pandas as pd

def sha1_of_file(path, blocksize=1<<20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(blocksize), b""):
            h.update(b)
    return h.hexdigest()

def analyze_messages_csv(csv_path: Path, labels_path: Path, thread_col="thread_id", text_col="text"):
    # lecture par chunks pour rester robuste en RAM
    rows = 0
    empty_texts = 0
    thread_counts = Counter()
    usecols = [thread_col, text_col]

    try:
        for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=50_000):
            chunk[thread_col] = chunk[thread_col].astype(str)
            # textes vides
            empty_texts += (chunk[text_col].astype(str).str.strip() == "").sum()
            # compteur par thread
            thread_counts.update(chunk[thread_col].value_counts().to_dict())
            rows += len(chunk)
    except Exception as e:
        return {"error": f"read_error: {e}"}

    n_threads = len(thread_counts)
    # stats msgs/thread
    if n_threads > 0:
        counts = list(thread_counts.values())
        counts.sort()
        msgs_min = counts[0]
        msgs_max = counts[-1]
        # median
        m = len(counts)//2
        msgs_med = (counts[m] if len(counts)%2==1 else (counts[m-1]+counts[m])/2)
        msgs_mean = sum(counts)/n_threads
    else:
        msgs_min = msgs_med = msgs_mean = msgs_max = 0

    # couverture labels
    coverage_pct = None
    missing_labels = None
    try:
        lbl = pd.read_csv(labels_path, usecols=[thread_col])
        lbl[thread_col] = lbl[thread_col].astype(str)
        covered = sum(1 for tid in thread_counts.keys() if tid in set(lbl[thread_col]))
        coverage_pct = 100.0 * covered / max(1, n_threads)
        missing_labels = n_threads - covered
    except Exception as e:
        coverage_pct = None

    return {
        "rows": rows,
        "n_threads": n_threads,
        "threads_lt_rows": bool(n_threads < rows),
        "msgs_per_thread_min": msgs_min,
        "msgs_per_thread_median": msgs_med,
        "msgs_per_thread_mean": round(msgs_mean, 3) if n_threads > 0 else 0.0,
        "msgs_per_thread_max": msgs_max,
        "empty_texts": int(empty_texts),
        "coverage_pct_labels": None if coverage_pct is None else round(coverage_pct, 2),
        "missing_labels": missing_labels,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True,
                    help="Racines à explorer (ex: C:\\AxioPoC C:\\AxioPoC_Data)")
    ap.add_argument("--labels", required=True,
                    help="Chemin vers afd_eval.csv (labels AfD par thread)")
    ap.add_argument("--pattern", default="*afd*messages*.csv",
                    help="Pattern glob (par défaut: *afd*messages*.csv)")
    ap.add_argument("--out", default="C:\\AxioPoC\\artifacts_xfer\\scan_afd_messages_summary.csv",
                    help="CSV résumé de sortie")
    args = ap.parse_args()

    labels_path = Path(args.labels)
    if not labels_path.exists():
        sys.exit(f"[ERR] labels introuvables: {labels_path}")

    candidates = []
    for root in args.roots:
        rootp = Path(root)
        if not rootp.exists():
            continue
        for p in rootp.rglob(args.pattern):
            if p.is_file():
                candidates.append(p)

    if not candidates:
        print("[INFO] Aucun fichier trouvé avec ce pattern.")
        print("Essayez d'élargir --roots ou de changer --pattern.")
        return

    rows_out = []
    seen_hashes = {}
    print(f"[INFO] {len(candidates)} fichier(s) candidat(s) trouvé(s). Analyse...")

    for p in candidates:
        try:
            h = sha1_of_file(p)
        except Exception as e:
            print(f"[WARN] hash fail: {p} -> {e}")
            h = None

        try:
            st = p.stat()
            size_mb = round(st.st_size / (1024*1024), 2)
            mtime = datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
        except Exception:
            size_mb = None
            mtime = None

        print(f" - {p}")
        res = analyze_messages_csv(p, labels_path)
        row = {
            "path": str(p),
            "sha1": h,
            "size_mb": size_mb,
            "mtime": mtime,
        }
        row.update(res)
        rows_out.append(row)

        # arrêter de spam si on a déjà vu le même fichier (dedup par hash)
        if h and h in seen_hashes:
            pass
        else:
            if h:
                seen_hashes[h] = str(p)

    df = pd.DataFrame(rows_out).sort_values(
        by=["threads_lt_rows","coverage_pct_labels","msgs_per_thread_median","rows"],
        ascending=[False, False, False, False]
    )

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False, encoding="utf-8")
    print(f"[OK] Résumé -> {outp}")

    # Affiche top 5
    print("\n[TOP 5 CANDIDATS]")
    cols = ["path","size_mb","threads_lt_rows","rows","n_threads",
            "msgs_per_thread_min","msgs_per_thread_median","msgs_per_thread_mean","msgs_per_thread_max",
            "empty_texts","coverage_pct_labels","missing_labels","mtime"]
    print(df[cols].head(5).to_string(index=False))

    # Conseils
    print("\n[GUIDE]")
    print("- Choisis un fichier où 'threads_lt_rows' == True, 'coverage_pct_labels' élevé (~100%),")
    print("  et 'msgs_per_thread_median' >= 2. Évite ceux où median==1.")
    print("- Si plusieurs fichiers ont le même sha1 -> c'est une duplication exacte.")
    print("- Une fois le bon CSV identifié, copie-le en canonique : C:\\AxioPoC\\artifacts_xfer\\afd_messages.csv")

if __name__ == "__main__":
    main()
