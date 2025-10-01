#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Telos-temporal ablation runner (annule & remplace)

Enchaîne automatiquement :
  - Signatures (src & tgt) pour 5 variantes:
      base4 = telos_wd_l0.02,telos_wd_l0.05,telos_klast5,telos_delta5
      noDelta = sans telos_delta5
      noKlast = sans telos_klast5
      noL002  = sans telos_wd_l0.02
      noL005  = sans telos_wd_l0.05
  - Évaluations de transfert (metric=diag)
  - (Optionnel) Permutation SAFE pour chaque variante (--perm-iters > 0)
  - Consolidation & figure barres dédiées aux ablations

Usage minimal (exemple PARTIAL):
  python scripts/run_telostemporal_ablation.py ^
    --feat-src H:\AxioPoC\artifacts_xfer\features_afd_telos_temporal_PARTIAL.csv ^
    --raw-src  C:\AxioPoC\data\wikipedia\afd\afd_eval.csv ^
    --feat-tgt H:\AxioPoC\artifacts_xfer\features_cmv_telos_temporal_from_AFDpartial.csv ^
    --raw-tgt  H:\AxioPoC\data\convokit\cmv\cmv_balanced_int.csv ^
    --out-dir  H:\AxioPoC\artifacts_xfer ^
    --dataset  CMV ^
    --direction "afd->cmv" ^
    --title "AUC — Telos-temporal (PARTIAL) ablations" ^
    --prefix TelosPARTIAL ^
    --perm-iters 0

Note : Le script utilise le Python courant (sys.executable) et gère correctement
les chemins Windows avec espaces (subprocess avec liste d’arguments).
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

VARIANTS = {
    "base4":  ["telos_wd_l0.02","telos_wd_l0.05","telos_klast5","telos_delta5"],
    "noDelta":["telos_wd_l0.02","telos_wd_l0.05","telos_klast5"],
    "noKlast":["telos_wd_l0.02","telos_wd_l0.05","telos_delta5"],
    "noL002": ["telos_wd_l0.05","telos_klast5","telos_delta5"],
    "noL005": ["telos_wd_l0.02","telos_klast5","telos_delta5"],
}

def run(cmd, cwd=None):
    print("[CMD]", " ".join(cmd))
    res = subprocess.run(cmd, cwd=cwd)
    if res.returncode != 0:
        raise SystemExit(f"Underlying command failed with code {res.returncode}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-src", required=True, help="CSV features source (AfD telos-temporal)")
    ap.add_argument("--raw-src",  required=True, help="CSV labels source")
    ap.add_argument("--feat-tgt", required=True, help="CSV features target (CMV telos-temporal)")
    ap.add_argument("--raw-tgt",  required=True, help="CSV labels target (CMV; ici int normalisés)")
    ap.add_argument("--out-dir",  required=True, help="Dossier de sortie pour signatures/metrics/perm")
    ap.add_argument("--dataset",  default="CMV",  help="Nom dataset cible (pour les JSON metrics)")
    ap.add_argument("--direction", default="afd->cmv", help="Direction report (ex: afd->cmv)")
    ap.add_argument("--metric",   default="diag", choices=["diag","sigma_inv"], help="Métrique")
    ap.add_argument("--prefix",   default="Telos", help="Préfixe pour nommer les fichiers générés")
    ap.add_argument("--title",    default="AUC — Telos-temporal ablations", help="Titre figure barres")
    ap.add_argument("--perm-iters", type=int, default=0, help="Itérations SAFE permutation (0 = skip)")
    ap.add_argument("--report-dir", default="REPORTS", help="Dossier REPORTS (pour consolidated.csv & figures)")
    args = ap.parse_args()

    py = sys.executable
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Boucle variantes: signatures src/tgt + éval transfert
    metric_json_paths = []
    for vname, cols in VARIANTS.items():
        cols_arg = ",".join(cols)
        sig_src = out_dir / f"signature_afd_{args.prefix}_{vname}.pkl"
        sig_tgt = out_dir / f"signature_cmv_{args.prefix}_{vname}.pkl"
        # Signatures
        run([py, "scripts/make_signature.py",
             "--feat", args.feat_src, "--out", str(sig_src), "--cols", cols_arg])
        run([py, "scripts/make_signature.py",
             "--feat", args.feat_tgt, "--out", str(sig_tgt), "--cols", cols_arg])

        # Évaluation transfert
        metrics_out = out_dir / f"metrics_afd2cmv_{args.prefix}_{vname}_{args.metric}.json"
        run([py, "scripts/run_and_report.py",
             "--mode", "transfer",
             "--feat_src", args.feat_src, "--raw_src", args.raw_src, "--signature_src", str(sig_src),
             "--feat_tgt", args.feat_tgt, "--raw_tgt", args.raw_tgt, "--signature_tgt", str(sig_tgt),
             "--metric", args.metric,
             "--dataset", args.dataset, "--direction", args.direction,
             "--tag_signature", f"{args.prefix}_{vname}_{args.metric}",
             "--out", str(metrics_out)])
        metric_json_paths.append(str(metrics_out))

        # (Option) Permutation SAFE
        if args.perm_iters and args.perm_iters > 0:
            perm_out = out_dir / f"permSAFE_afd2cmv_{args.prefix}_{vname}_{args.metric}_{args.perm_iters}.json"
            run([py, "scripts/perm_test_transfer_safe.py",
                 "--feat-src", args.feat_src, "--raw-src", args.raw_src, "--sig-src", str(sig_src),
                 "--feat-tgt", args.feat_tgt, "--raw-tgt", args.raw_tgt, "--sig-tgt", str(sig_tgt),
                 "--iters", str(args.perm_iters),
                 "--outjson", str(perm_out)])

    # 2) Consolidation — n’agrège que les runs de ce script (prefix filtrant)
    reports_dir = Path(args.report_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    consolidated_csv = reports_dir / f"metrics_consolidated_{args.prefix}_ablations.csv"

    # utilise un pattern qui matche ce que l’on vient d’écrire
    inputs_pattern = str(out_dir / f"metrics_afd2cmv_{args.prefix}_*_{args.metric}.json")
    run([py, "scripts/collect_metrics.py",
         "--inputs", inputs_pattern,
         "--outcsv", str(consolidated_csv)])

    # 3) Figure barres (ablations seulement)
    bars_png = reports_dir / f"img/auc_bars_{args.prefix}_ablations_{args.metric}.png"
    bars_png.parent.mkdir(parents=True, exist_ok=True)
    run([py, "scripts/plot_auc_bars.py",
         "--csv", str(consolidated_csv),
         "--filter_metric", args.metric,
         "--title", args.title,
         "--out", str(bars_png)])

    print(f"[OK] Ablations terminées.\n - Consolidated -> {consolidated_csv}\n - Figure -> {bars_png}")

if __name__ == "__main__":
    main()
