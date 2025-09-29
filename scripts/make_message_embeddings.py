#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sys, math, glob, tempfile
import numpy as np
import pandas as pd

# ---------------- Auto-détection de colonnes ----------------
TEXT_CANDS   = ["text", "body", "content", "message", "clean_text"]
THREAD_CANDS = ["thread_id", "conversation_id", "thread", "conv_id", "t3_id"]
TIME_CANDS   = ["created_utc", "created", "timestamp", "time", "date"]
ID_CANDS     = ["message_id", "id", "post_id", "comment_id"]

def pick_first_present(df: pd.DataFrame, candidates, required=False, what=""):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise SystemExit(f"Impossible de trouver une colonne pour {what}. Cherché: {candidates}")
    return ""
# ------------------------------------------------------------

def write_parquet(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.lower().endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)

def concat_shards(shard_paths: list[str], final_out: str):
    # Concaténation finale (mémoire OK pour ~80k x 384 float32)
    if not shard_paths:
        return
    parts = [pd.read_parquet(p) for p in shard_paths]
    full = pd.concat(parts, ignore_index=True)
    write_parquet(full, final_out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--messages", required=True, help="CSV des messages")
    ap.add_argument("--text_col", default="", help="Nom de la colonne texte (auto si vide)")
    ap.add_argument("--thread_col", default="", help="Nom de la colonne thread_id (auto si vide)")
    ap.add_argument("--time_col", default="", help="Nom de la colonne temps (auto si vide)")
    ap.add_argument("--id_col", default="", help="Nom de la colonne id message (auto si vide)")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", required=True, help="Sortie finale (.parquet recommandé)")
    ap.add_argument("--out_prefix", default="", help="Préfixe pour shards (par défaut dérivé de --out)")
    ap.add_argument("--shard-size", type=int, default=10000, help="Nb de lignes par shard parquet")
    args = ap.parse_args()

    # ----- Lock anti-doublon -----
    lock_path = args.out + ".lock"
    if os.path.exists(lock_path):
        print(f"[LOCK] Un run est déjà actif ou s’est terminé anormalement. Supprime {lock_path} si tu es sûr.", file=sys.stderr)
        sys.exit(1)
    open(lock_path, "w").close()

    try:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise SystemExit("Installez sentence-transformers : pip install sentence-transformers") from e

        df = pd.read_csv(args.messages)

        text_col   = args.text_col   or pick_first_present(df, TEXT_CANDS,   required=True,  what="le texte")
        thread_col = args.thread_col or pick_first_present(df, THREAD_CANDS, required=True,  what="l'identifiant de thread")
        time_col   = args.time_col   or pick_first_present(df, TIME_CANDS,   required=False, what="le temps")
        id_col     = args.id_col     or pick_first_present(df, ID_CANDS,     required=False, what="l'identifiant message")

        if not id_col or id_col not in df.columns:
            df["__msg_id__"] = np.arange(len(df), dtype=np.int64)
            id_col = "__msg_id__"

        # Tri temporel (ou ordre implicite par cumcount)
        if time_col and time_col in df.columns:
            df = df.sort_values([thread_col, time_col]).copy()
        else:
            df = df.sort_values([thread_col]).copy()
            df["__t__"] = df.groupby(thread_col).cumcount().astype(float)
            time_col = "__t__"

        texts = df[text_col].fillna("").astype(str).tolist()
        n = len(df)

        model = SentenceTransformer(args.model)
        dim = model.get_sentence_embedding_dimension()

        # Préfixe shards
        out_prefix = args.out_prefix or (os.path.splitext(args.out)[0] + "_shard")
        shard_dir = os.path.dirname(args.out) or "."
        shard_paths: list[str] = []

        # Buffer courant
        rows_buf = []
        ecols = [f"e{j}" for j in range(dim)]

        for i in range(0, n, args.batch):
            batch_texts = texts[i:i+args.batch]
            emb = model.encode(batch_texts, convert_to_numpy=True, normalize_embeddings=False)  # (b, dim)

            # slice du df aligné
            part = df.iloc[i:i+args.batch][[id_col, thread_col, time_col]].reset_index(drop=True)
            edf = pd.DataFrame(emb.astype(np.float32), columns=ecols)
            out_part = pd.concat([part, edf], axis=1)
            rows_buf.append(out_part)

            done = min(i + args.batch, n)
            print(f"[emb] {done}/{n}", flush=True)

            # Flush en shard si on dépasse shard-size
            total_in_buf = sum(len(x) for x in rows_buf)
            if total_in_buf >= args.shard_size:
                shard_idx = len(shard_paths) + 1
                shard_path = f"{out_prefix}_{shard_idx:03d}.parquet"
                write_parquet(pd.concat(rows_buf, ignore_index=True), shard_path)
                shard_paths.append(shard_path)
                rows_buf = []  # reset

        # Dernier flush du buffer
        if rows_buf:
            shard_idx = len(shard_paths) + 1
            shard_path = f"{out_prefix}_{shard_idx:03d}.parquet"
            write_parquet(pd.concat(rows_buf, ignore_index=True), shard_path)
            shard_paths.append(shard_path)
            rows_buf = []

        # Concaténation finale -> args.out
        print(f"[merge] Concatène {len(shard_paths)} shard(s) -> {args.out}")
        concat_shards(shard_paths, args.out)

        # Nettoyage shards (optionnel)
        for p in shard_paths:
            try: os.remove(p)
            except: pass

        print(f"[OK] Embeddings -> {args.out} | dim={dim} | n={n}")
        print(f"     cols: id={id_col}, thread={thread_col}, time={time_col}")

    finally:
        # Libérer le lock
        try: os.remove(lock_path)
        except: pass

if __name__ == "__main__":
    main()
