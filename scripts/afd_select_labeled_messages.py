# -*- coding: utf-8 -*-
"""
Sélectionne les messages AfD qui ont un label (0/1) et écrit afd_messages_labeled.csv.
Entrées:
  --msgs  : artifacts_xfer/afd_messages.csv     (thread_id,text)
  --labels: data/wikipedia/afd/afd_eval.csv     (thread_id,label)
Sortie:
  --out   : artifacts_xfer/afd_messages_labeled.csv
"""
import argparse
import pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msgs",   required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out",    required=True)
    args = ap.parse_args()

    dfm = pd.read_csv(args.msgs, dtype={"thread_id": int})
    dfl = pd.read_csv(args.labels, dtype={"thread_id": int, "label": int})
    m_ids = set(dfm["thread_id"])
    l_ids = set(dfl["thread_id"])
    inter = m_ids & l_ids
    print(f"messages: {len(dfm)} | labels: {len(dfl)} | communs: {len(inter)}")

    df_out = dfm[dfm["thread_id"].isin(inter)].merge(
        dfl, on="thread_id", how="inner"
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.out, index=False)
    print(f"Wrote labeled messages: {len(df_out)} -> {args.out}")

if __name__ == "__main__":
    main()
