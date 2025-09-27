# -*- coding: utf-8 -*-
r"""
Rebâtit un jeu AfD...
Entrées :
  C:\Users\<toi>\.convokit\saved-corpora\...

Rebâtit un jeu AfD minimal depuis le corpus ConvoKit local, en visant
la même numérotation séquentielle des thread_id (100000001, 100000002, ...).

Entrées (ConvoKit, local):
  C:\Users\<toi>\.convokit\saved-corpora\wiki-articles-for-deletion-corpus\
    - utterances.jsonl   (texte, ids d'utterance et conversation)
    - conversations.json (liste des conversations)

Sorties:
  artifacts_xfer\afd_messages.csv  (thread_id,text)  -- texte concaténé par thread
  artifacts_xfer\afd_idmap.csv     (thread_id,conversation_id) -- pour audit

Commande:
  python scripts/afd_rebuild_from_convokit.py ^
    --convokit-root "C:\Users\ojahi\.convokit\saved-corpora\wiki-articles-for-deletion-corpus" ^
    --out-msg "C:\AxioPoC\artifacts_xfer\afd_messages.csv" ^
    --out-map "C:\AxioPoC\artifacts_xfer\afd_idmap.csv"
"""
import argparse, json
from pathlib import Path
import pandas as pd

def load_conversations(conv_path: str) -> pd.DataFrame:
    """
    Supporte:
      - JSON dict: { "<id>": {...}, ... }
      - JSON list: [ {"id": ...}, {"id": ...}, ... ]
      - JSONL: une conv par ligne (avec champ "id")
    """
    import json

    # 1) On tente JSON standard
    with open(conv_path, "r", encoding="utf-8") as f:
        txt = f.read().strip()

    data = None
    try:
        data = json.loads(txt)
    except Exception:
        data = None

    rows = []
    if isinstance(data, dict):
        # cas dict: clés = ids
        for idx, (cid, obj) in enumerate(data.items(), start=1):
            # si l'objet a aussi un "id", on le garde, sinon on prend la clé
            cid2 = obj.get("id", cid) if isinstance(obj, dict) else cid
            rows.append({"_order": idx, "conversation_id": cid2})
        return pd.DataFrame(rows)

    if isinstance(data, list):
        # cas liste d'objets
        for idx, obj in enumerate(data, start=1):
            if isinstance(obj, dict):
                cid = obj.get("id")
            else:
                cid = None
            if cid is None:
                continue
            rows.append({"_order": idx, "conversation_id": cid})
        return pd.DataFrame(rows)

    # 2) Fallback: JSONL (une conv par ligne)
    rows = []
    with open(conv_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                cid = obj.get("id")
                if cid is None:
                    continue
                rows.append({"_order": idx, "conversation_id": cid})
            except Exception:
                continue
    return pd.DataFrame(rows)


def load_utterances(utt_path: str) -> pd.DataFrame:
    # utterances.jsonl : une ligne JSON par utterance, avec "conversation_id" et "text"
    import json as _json
    texts = {}
    with open(utt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u = _json.loads(line)
            cid = u.get("conversation_id")
            tx  = u.get("text") or ""
            if cid not in texts:
                texts[cid] = []
            texts[cid].append(tx)
    # concat par conversation
    rows = [{"conversation_id": cid, "text": " ".join(lst)} for cid, lst in texts.items()]
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convokit-root", required=True)
    ap.add_argument("--out-msg", required=True)
    ap.add_argument("--out-map", required=True)
    args = ap.parse_args()

    root = Path(args.convokit_root)
    conv_path = root / "conversations.json"
    utt_path  = root / "utterances.jsonl"
    if not conv_path.exists() or not utt_path.exists():
        raise SystemExit("Fichiers ConvoKit introuvables (conversations.json / utterances.jsonl).")

    df_conv = load_conversations(str(conv_path))
    df_utt  = load_utterances(str(utt_path))

    # Jointure et ordre d'origine pour numéroter de façon déterministe
    df = df_conv.merge(df_utt, on="conversation_id", how="left")
    df["text"] = df["text"].fillna("")
    df = df.sort_values("_order", kind="mergesort")

    # Assigner thread_id séquentiels (100000001, 100000002, ...)
    base = 100000000
    df["thread_id"] = [base + i for i in range(1, len(df) + 1)]

    # Sauvegardes
    Path(args.out_msg).parent.mkdir(parents=True, exist_ok=True)
    df[["thread_id", "text"]].to_csv(args.out_msg, index=False)
    df[["thread_id", "conversation_id"]].to_csv(args.out_map, index=False)
    print(f"Wrote messages: {args.out_msg}  (n={len(df)})")
    print(f"Wrote idmap:    {args.out_map}")

if __name__ == "__main__":
    main()
