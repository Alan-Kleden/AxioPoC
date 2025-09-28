# -*- coding: utf-8 -*-
"""
Convertit le corpus Webis-CMV-20 (threads.jsonl[.bz2]) en CSV minimal `thread_id,text`.
Accepte un .jsonl ou .jsonl.bz2. Aucune dépendance externe.

Usage:
  python scripts/cmv_threads_to_messages.py --in C:\path\to\threads.jsonl.bz2 --out .\artifacts_xfer\cmv_messages.csv
"""
import argparse, json, bz2, io, os, re, sys
from typing import Any, Dict, Iterable, List, Union

def open_maybe_bz2(path: str):
    if path.lower().endswith(".bz2"):
        return io.TextIOWrapper(bz2.open(path, mode="rb"), encoding="utf-8", errors="replace")
    return open(path, mode="r", encoding="utf-8", errors="replace")

WS = re.compile(r"\s+")

def norm_text(s: Union[str, None]) -> str:
    if not s:
        return ""
    # VADER préfère la casse originale ; on se contente de nettoyer les espaces.
    return WS.sub(" ", str(s)).strip()

def collect_texts(obj: Any, out: List[str]):
    """
    Parcourt récursivement les champs plausibles contenant du texte.
    Adapte-toi aux variations de schéma (title, selftext, body, text, content, etc.).
    """
    if obj is None:
        return
    if isinstance(obj, str):
        t = norm_text(obj)
        if t:
            out.append(t)
        return
    if isinstance(obj, dict):
        # Champs textuels fréquents
        for k in ("title", "selftext", "body", "text", "content"):
            if k in obj and isinstance(obj[k], (str, type(None))):
                t = norm_text(obj[k])
                if t:
                    out.append(t)
        # Plongée récursive sur les sous-objets potentiels (posts, comments, children, etc.)
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                collect_texts(v, out)
        return
    if isinstance(obj, list):
        for it in obj:
            collect_texts(it, out)

def get_thread_id(obj: Dict[str, Any]) -> Union[str, None]:
    """
    Cherche un identifiant de thread plausible.
    Priorité à 'id' puis 'thread_id', 'conversation_id'.
    """
    for k in ("id", "thread_id", "conversation_id"):
        if k in obj and obj[k] is not None:
            return str(obj[k])
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Chemin vers threads.jsonl ou threads.jsonl.bz2 (Webis-CMV-20)")
    ap.add_argument("--out", dest="out", required=True, help="Chemin sortie CSV thread_id,text")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    n, n_ok, n_text = 0, 0, 0
    with open_maybe_bz2(args.inp) as f_in, open(args.out, "w", encoding="utf-8", newline="") as f_out:
        f_out.write("thread_id,text\n")
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue

            tid = get_thread_id(obj)
            if not tid:
                continue

            texts: List[str] = []
            # Récupère tout le texte utile du thread
            collect_texts(obj, texts)
            txt = norm_text(" ".join(texts))
            if txt:
                n_text += 1
            f_out.write('"{0}","{1}"\n'.format(tid.replace('"', '""'), txt.replace('"', '""')))
            n_ok += 1

    print(f"WROTE {args.out}  rows={n_ok}  (input lines={n}; with_non_empty_text={n_text})")

if __name__ == "__main__":
    main()
