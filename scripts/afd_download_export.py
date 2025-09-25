# scripts/afd_download_export.py
import os, csv
from pathlib import Path
from convokit import Corpus, download
from tqdm import tqdm

OUT_DIR = Path("data/wikipedia/afd")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "afd.csv"

def first_nonempty(*vals):
    for v in vals:
        if v is None: 
            continue
        s = str(v).strip()
        if s:
            return s
    return ""

def get_timestamp(utt, seq_idx):
    # Cherche timestamp dans meta, sinon fallback ordinal
    if hasattr(utt, "meta") and isinstance(utt.meta, dict):
        for key in ("timestamp", "time", "created_utc", "date"):
            if utt.meta.get(key) is not None:
                return utt.meta[key]
    if getattr(utt, "timestamp", None) is not None:
        return utt.timestamp
    return seq_idx

def main():
    print("Téléchargement du corpus ConvoKit: wiki-articles-for-deletion-corpus ...")
    corpus = Corpus(filename=download("wiki-articles-for-deletion-corpus"))
    print("OK. Export CSV minimal ->", OUT_CSV)

    rows = []
    conv_ids = list(corpus.get_conversation_ids())
    for cid in tqdm(conv_ids, desc="Conversations"):
        conv = corpus.get_conversation(cid)
        # outcome au niveau conversation.meta
        outcome = ""
        if hasattr(conv, "meta") and isinstance(conv.meta, dict):
            for k in ("outcome", "decision", "result", "final_decision"):
                if conv.meta.get(k):
                    outcome = str(conv.meta[k])
                    break

        utt_ids = list(conv.get_utterance_ids())
        for i, uid in enumerate(utt_ids):
            utt = corpus.get_utterance(uid)
            text = first_nonempty(
                getattr(utt, "text", None),
                getattr(utt, "raw_text", None),
                (utt.meta.get("text") if hasattr(utt, "meta") and isinstance(utt.meta, dict) else None)
            )
            text = " ".join(text.split())
            if not text:
                continue
            ts = get_timestamp(utt, i)
            rows.append({"thread_id": cid, "text": text, "timestamp": ts, "outcome": outcome})

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["thread_id", "text", "timestamp", "outcome"])
        w.writeheader()
        w.writerows(rows)

    print(f"Export terminé: {OUT_CSV} | lignes={len(rows)}")

if __name__ == "__main__":
    main()
