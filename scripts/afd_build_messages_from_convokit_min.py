# -*- coding: utf-8 -*-
import argparse, os, sys
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Chemin de sortie du CSV (afd_messages.csv)")
    ap.add_argument("--chunk", type=int, default=100_000, help="taille de flush CSV")
    args = ap.parse_args()

    try:
        from convokit import Corpus, download
    except Exception as e:
        raise SystemExit("Installez convokit: pip install convokit") from e

    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    # (1) Charger / télécharger le corpus
    print("[dl] Using ConvoKit dataset 'wiki-articles-for-deletion-corpus' ...")
    corpus = Corpus(filename=download("wiki-articles-for-deletion-corpus"))

    # (2) Reprise : si un CSV partiel existe, on reprend après
    written = 0
    if os.path.exists(out):
        try:
            prev = pd.read_csv(out, usecols=["thread_id","text"])
            written = len(prev)
            print(f"[resume] Reprend après {written} lignes déjà écrites.")
        except Exception:
            written = 0

    # (3) Parcours + écriture par paquets
    rows = []
    total = 0
    for i, utt in enumerate(corpus.iter_utterances()):
        if i < written:
            continue  # saute ce qui est déjà dans le CSV
        txt = (utt.text or "").strip()
        if not txt:
            continue
        rows.append({"thread_id": utt.conversation_id, "text": txt})
        total += 1
        if total % args.chunk == 0:
            mode = "a" if os.path.exists(out) else "w"
            header = not os.path.exists(out)
            pd.DataFrame(rows).to_csv(out, index=False, mode=mode, header=header)
            print(f"[flush] +{len(rows)} lignes (total écrit ~{written + total})")
            rows = []

    # flush final
    if rows:
        mode = "a" if os.path.exists(out) else "w"
        header = not os.path.exists(out)
        pd.DataFrame(rows).to_csv(out, index=False, mode=mode, header=header)
        print(f"[flush] +{len(rows)} lignes (total écrit ~{written + total})")

    print(f"[OK] WROTE {out}")

if __name__ == "__main__":
    main()
