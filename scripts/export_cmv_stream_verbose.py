# -*- coding: utf-8 -*-
"""
export_cmv_stream_verbose.py
--------------------------------
Exporte le corpus ConvoKit "Winning Arguments (ChangeMyView)" vers un CSV
compatible Benchmark B : colonnes [thread_id, text, timestamp, outcome].

Définition du label (niveau conversation) : MAJORITÉ des utterances annotées
- outcome = 1 si (#success==1) > (#success==0)
- outcome = 0 si (#success==0) > (#success==1)
- outcome = None si aucune utterance annotée OU égalité (fil ignoré par défaut)

Options :
  --only-paired (défaut)   : ne garder que les fils avec majorité claire (0/1)
  --all-conv               : garder aussi les fils outcome=None (peu utile pour l'apprentissage)
  --limit-conv N           : limiter le nombre de fils gardés (après filtre)
  --verbose-every N        : logs progressifs toutes les N conversations parcourues
"""

import os
import csv
import time
import argparse
from typing import Optional

try:
    from convokit import Corpus
except Exception as e:
    raise SystemExit(
        "Erreur: impossible d'importer ConvoKit. "
        "Installez-le dans votre venv : pip install convokit\n"
        f"Détails: {e}"
    )


# ---------- Détermination de l'issue par majorité ----------

def conv_outcome_majority(conv) -> Optional[int]:
    """
    Retourne 1 / 0 / None selon la majorité des utterances annotées.
    None si aucune utterance n'a meta['success'] ∈ {0,1} OU égalité.
    """
    pos = 0
    neg = 0
    for utt in conv.iter_utterances():
        meta = utt.meta or {}
        s = meta.get("success", None)
        if s == 1:
            pos += 1
        elif s == 0:
            neg += 1
    if pos == 0 and neg == 0:
        return None
    if pos == neg:
        return None
    return 1 if pos > neg else 0


# ---------- Export principal ----------

def export_cmv(corpus_dir: str, out_csv: str, limit_conv: Optional[int],
               only_paired: bool, verbose_every: int = 100) -> None:
    t0 = time.time()
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    print(f"[INFO] Chargement du corpus : {corpus_dir}")
    cmv = Corpus(filename=corpus_dir)
    print("[INFO] Corpus chargé.")

    n_conv_scanned = 0     # fils parcourus
    n_conv_kept = 0        # fils gardés (après filtre)
    n_rows = 0
    outcome_pos = 0
    outcome_neg = 0
    outcome_none = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["thread_id", "text", "timestamp", "outcome"])
        w.writeheader()

        for conv in cmv.iter_conversations():
            n_conv_scanned += 1

            outc = conv_outcome_majority(conv)
            if outc is None:
                outcome_none += 1
                if only_paired:
                    # On saute les fils sans majorité claire si demandé
                    if (n_conv_scanned % verbose_every) == 0:
                        size_mb = os.path.getsize(out_csv) / (1024 * 1024)
                        print(f"[{n_conv_scanned}] (skip None) {n_conv_kept} fils gardés, {n_rows} lignes, ~{size_mb:.1f} Mo")
                    # Limite s'applique sur les fils GARDÉS ; on continue donc
                    continue

            # Stats issues
            if outc == 1:
                outcome_pos += 1
            elif outc == 0:
                outcome_neg += 1

            n_conv_kept += 1

            # Écriture de toutes les utterances du fil, labellisées par outcome du fil
            for utt in conv.iter_utterances():
                meta = utt.meta or {}
                ts = meta.get("timestamp") or meta.get("created_utc")
                text = utt.text
                if not isinstance(text, str):
                    text = "" if text is None else str(text)

                w.writerow({
                    "thread_id": conv.id,
                    "text": text,
                    "timestamp": ts,
                    "outcome": outc
                })
                n_rows += 1

            # Logs réguliers
            if (n_conv_scanned % verbose_every) == 0:
                size_mb = os.path.getsize(out_csv) / (1024 * 1024)
                print(f"[{n_conv_scanned}] {n_conv_kept} fils gardés, {n_rows} lignes, ~{size_mb:.1f} Mo")

            # Limite de fils gardés (après filtre)
            if limit_conv and n_conv_kept >= limit_conv:
                break

    elapsed = time.time() - t0
    print("\n[FIN]")
    print(f"- Fils parcourus      : {n_conv_scanned}")
    print(f"- Fils gardés         : {n_conv_kept} (only_paired={only_paired})")
    print(f"- Lignes écrites      : {n_rows}")
    print(f"- outcome=1           : {outcome_pos}")
    print(f"- outcome=0           : {outcome_neg}")
    print(f"- outcome=None (skip) : {outcome_none}")
    print(f"- Fichier             : {out_csv}")
    print(f"- Temps total         : {elapsed:.1f}s")


# ---------- CLI ----------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Export ConvoKit CMV -> CSV (label par MAJORITÉ des utterances annotées)"
    )
    ap.add_argument(
        "--corpus-dir",
        default=r"data/convokit/cmv/winning-args-corpus/winning-args-corpus",
        help="Dossier contenant conversations.json, utterances.jsonl, etc."
    )
    ap.add_argument(
        "--out",
        default=r"data/convokit/cmv/cmv.csv",
        help="Chemin du CSV de sortie"
    )
    ap.add_argument(
        "--limit-conv",
        type=int,
        default=None,
        help="Limiter le nombre de fils gardés (après filtre). None = tous."
    )
    ap.add_argument(
        "--only-paired",
        action="store_true",
        default=True,
        help="Garder uniquement les fils avec majorité claire (success 1/0)."
    )
    ap.add_argument(
        "--all-conv",
        action="store_true",
        help="Garder aussi les fils non annotés ou à égalité (outcome=None)."
    )
    ap.add_argument(
        "--verbose-every",
        type=int,
        default=100,
        help="Afficher un log toutes les N conversations parcourues."
    )
    return ap.parse_args()


def main():
    args = parse_args()
    only_paired = args.only_paired and not args.all_conv

    corpus_dir = os.path.normpath(args.corpus_dir)
    out_csv = os.path.normpath(args.out)

    export_cmv(
        corpus_dir=corpus_dir,
        out_csv=out_csv,
        limit_conv=args.limit_conv,
        only_paired=only_paired,
        verbose_every=args.verbose_every
    )


if __name__ == "__main__":
    main()
