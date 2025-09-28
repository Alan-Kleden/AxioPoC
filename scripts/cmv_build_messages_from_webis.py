# scripts/cmv_build_messages_from_webis.py
# -*- coding: utf-8 -*-
"""
Construit un CSV messages CMV (thread_id,text) à partir du corpus Webis CMV 2020.

Entrées (décompressées de .bz2) :
  - threads.jsonl                (soumissions : titre + selftext)
  - posts_malleability.jsonl     (posts / commentaires)

Sortie :
  - cmv_messages.csv  avec colonnes : thread_id,text
    (une ligne par message ; le group-by par thread sera fait plus tard)

Usage :
  python scripts/cmv_build_messages_from_webis.py \
    --threads E:\Extraction\threads.jsonl \
    --posts   E:\Extraction\posts_malleability.jsonl \
    --out     .\artifacts_xfer\cmv_messages.csv
"""

import argparse
import json
import csv
from pathlib import Path
from typing import Dict, Iterable, Tuple, Optional


# ---------- Utilitaires ----------

def to_thread_int(tid) -> int:
    """
    Convertit un id Reddit en entier via base36.
    Accepte : 't3_3i1y4z', '3i1y4z', ou déjà int.
    Fallback : hash positif si conversion impossible.
    """
    if isinstance(tid, int):
        return tid
    s = str(tid) if tid is not None else ""
    if s.startswith("t3_"):
        s = s[3:]
    try:
        return int(s, 36)
    except Exception:
        return abs(hash(s)) % (2**31)


def read_jsonl(path: Path) -> Iterable[Dict]:
    """Yield un dict par ligne JSON valide depuis path."""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # ligne corrompue : on ignore
                continue


def pick_first(d: Dict, keys: Iterable[str], default=None):
    """Renvoie la première valeur non-vide trouvée parmi `keys` dans `d`."""
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def clean_text(x) -> str:
    """Nettoyage très léger : cast en str, strip; garde la ponctuation."""
    if x is None:
        return ""
    s = str(x)
    return s.strip()


# ---------- Extraction threads / posts ----------

def collect_from_threads(path: Path) -> Iterable[Tuple[int, str]]:
    """
    Extrait (thread_id_int, text) depuis threads.jsonl.
    thread_id : id/ thread_id / root_id (si présent)
    texte     : combinaison title + selftext (ou fallback text/body/content)
    """
    for obj in read_jsonl(path):
        tid_raw = pick_first(obj, ["id", "thread_id", "root_id"])
        title   = pick_first(obj, ["title"])
        body    = pick_first(obj, ["selftext", "text", "body", "content"])

        tid_int = to_thread_int(tid_raw)
        parts = []
        t = clean_text(title)
        b = clean_text(body)
        if t:
            parts.append(t)
        if b:
            parts.append(b)
        if not parts:
            # rien à dire ? on skip
            continue
        yield (tid_int, " ".join(parts))


def collect_from_posts(path: Path) -> Iterable[Tuple[int, str]]:
    """
    Extrait (thread_id_int, text) depuis posts_malleability.jsonl.
    thread_id : link_id / thread_id / root_id / link / thread
    texte     : body / text / content
    """
    for obj in read_jsonl(path):
        tid_raw = pick_first(obj, ["link_id", "thread_id", "root_id", "link", "thread"])
        text    = pick_first(obj, ["body", "text", "content", "selftext"])
        txt = clean_text(text)
        if not txt:
            continue
        tid_int = to_thread_int(tid_raw)
        yield (tid_int, txt)


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", required=True, help="Chemin threads.jsonl")
    ap.add_argument("--posts",   required=True, help="Chemin posts_malleability.jsonl")
    ap.add_argument("--out",     required=True, help="CSV de sortie (thread_id,text)")
    args = ap.parse_args()

    threads_p = Path(args.threads)
    posts_p   = Path(args.posts)
    out_p     = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    n_threads, n_posts = 0, 0

    with out_p.open("w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["thread_id", "text"])

        # Threads (titre + selftext)
        for tid_int, txt in collect_from_threads(threads_p):
            w.writerow([tid_int, txt])
            n_threads += 1

        # Posts (commentaires)
        for tid_int, txt in collect_from_posts(posts_p):
            w.writerow([tid_int, txt])
            n_posts += 1

    print(f"WROTE {out_p}  | rows_threads={n_threads}  rows_posts={n_posts}  total={n_threads + n_posts}")


if __name__ == "__main__":
    main()
