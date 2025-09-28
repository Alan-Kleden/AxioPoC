#!/usr/bin/env python3
import argparse, subprocess, json, re, os, datetime as dt, sys

# Accept lines like:
# "AUC mean±sd: 0.534 ± 0.044" or "AUC mean+/-sd: 0.534 +/- 0.044"
PAT_AUC = re.compile(r"AUC\s+mean(?:±|\+/-)sd:\s*([0-9.]+)\s*(?:±|\+/-)\s*([0-9.]+)", re.I)
PAT_ACC = re.compile(r"ACC\s+mean(?:±|\+/-)sd:\s*([0-9.]+)\s*(?:±|\+/-)\s*([0-9.]+)", re.I)
# Optional thread count line e.g. "Threads: 640 | Positives: ..."
PAT_N   = re.compile(r"Threads:\s*([0-9]+)", re.I)

def run_cmd(cmd_list):
    # cmd_list is a list like [sys.executable, "./scripts/eval_da.py", ...]
    p = subprocess.run(cmd_list, capture_output=True, text=True, shell=False)
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    return p.returncode, out

def parse_metrics(stdout_text):
    # 1) Format INTRA attendu : "AUC mean±sd: 0.534 ± 0.044" / "ACC mean±sd: ..."
    auc = PAT_AUC.search(stdout_text)
    acc = PAT_ACC.search(stdout_text)
    n   = PAT_N.search(stdout_text)

    if auc and acc:
        return {
            "auc_mean": float(auc.group(1)),
            "auc_sd":   float(auc.group(2)),
            "acc_mean": float(acc.group(1)),
            "acc_sd":   float(acc.group(2)),
            "n":        int(n.group(1)) if n else None,
        }

    # 2) Format TRANSFER observé :
    # "Transfer diag Eval: n_src=279636, n_tgt=640, AUC 0.525475, ACC 0.487500"
    m = re.search(r"n_src\s*=\s*(\d+).*?n_tgt\s*=\s*(\d+).*?AUC\s*([0-9.]+)\s*,\s*ACC\s*([0-9.]+)", stdout_text, re.I|re.S)
    if m:
        n_src = int(m.group(1))
        n_tgt = int(m.group(2))
        auc_mean = float(m.group(3))
        acc_mean = float(m.group(4))
        # pas d'écart-type dispo en transfert -> None
        return {
            "auc_mean": auc_mean,
            "auc_sd":   None,
            "acc_mean": acc_mean,
            "acc_sd":   None,
            # on met n = n_tgt par cohérence "évalué sur la cible"
            "n":        n_tgt
        }

    # 3) Dernière chance : format très simple "AUC 0.xxx, ACC 0.yyy"
    m2 = re.search(r"AUC\s*([0-9.]+)\s*,\s*ACC\s*([0-9.]+)", stdout_text, re.I)
    if m2:
        return {
            "auc_mean": float(m2.group(1)),
            "auc_sd":   None,
            "acc_mean": float(m2.group(2)),
            "acc_sd":   None,
            "n":        None
        }

    # Échec -> aide au debug
    tail = stdout_text[-1200:]
    raise SystemExit("Could not parse AUC/ACC from output:\n---8<---\n" + tail + "\n--->8---")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["intra","transfer"], required=True)
    # intra
    ap.add_argument("--feat", help="features CSV (intra)")
    ap.add_argument("--raw", help="labels CSV (intra)")
    ap.add_argument("--signature", help="signature pickle (intra)")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--reps", type=int, default=3)
    # transfer
    ap.add_argument("--feat_src", help="features CSV (transfer source)")
    ap.add_argument("--raw_src", help="labels CSV (transfer source)")
    ap.add_argument("--signature_src", help="signature pickle (transfer source)")
    ap.add_argument("--feat_tgt", help="features CSV (transfer target)")
    ap.add_argument("--raw_tgt", help="labels CSV (transfer target)")
    ap.add_argument("--signature_tgt", help="signature pickle (transfer target)")
    # common
    ap.add_argument("--metric", choices=["diag","sigma_inv"], default="diag")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--tag_signature", default="Da13_diag", help="string label for 'signature' field in JSON")
    ap.add_argument("--dataset", default="", help="string label for dataset field (e.g., CMV or AfD)")
    ap.add_argument("--direction", default="", help="afd->cmv, cmv->afd or intra")
    ap.add_argument("--model", default="logreg", help="string label for model field")
    args = ap.parse_args()

    if args.mode == "intra":
        if not (args.feat and args.raw and args.signature):
            raise SystemExit("--feat/--raw/--signature are required for --mode intra")
        cmd = [
            sys.executable, "./scripts/eval_da.py",
            "--feat", args.feat, "--raw", args.raw,
            "--signature", args.signature, "--metric", args.metric,
            "--cv", str(args.cv), "--reps", str(args.reps)
        ]
    else:
        # transfer
        reqs = [args.feat_src, args.raw_src, args.signature_src,
                args.feat_tgt, args.raw_tgt, args.signature_tgt]
        if not all(reqs):
            raise SystemExit("All --feat_src/--raw_src/--signature_src and --feat_tgt/--raw_tgt/--signature_tgt are required for --mode transfer")
        cmd = [
            sys.executable, "./scripts/eval_da_transfer.py",
            "--feat-src", args.feat_src, "--raw-src", args.raw_src, "--signature-src", args.signature_src,
            "--feat-tgt", args.feat_tgt, "--raw-tgt", args.raw_tgt, "--signature-tgt", args.signature_tgt,
            "--metric", args.metric
        ]

    # run
    code, out = run_cmd(cmd)
    print(out)  # keep full stdout/stderr visible in the console
    if code != 0:
        raise SystemExit(f"Underlying command failed with code {code}")

    # parse
    m = parse_metrics(out)

    # build JSON
    payload = {
        "dataset": args.dataset,
        "setting": args.mode,
        "direction": args.direction if args.direction else ("intra" if args.mode == "intra" else ""),
        "signature": args.tag_signature,
        "model": args.model,
        "cv": args.cv if args.mode == "intra" else None,
        "reps": args.reps if args.mode == "intra" else None,
        "n": m["n"],
        "metrics": {
            "auc_mean": m["auc_mean"],
            "auc_sd":   m["auc_sd"],
            "acc_mean": m["acc_mean"],
            "acc_sd":   m["acc_sd"]
        },
        "timestamp": dt.datetime.now().isoformat()
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[OK] JSON -> {args.out}")

if __name__ == "__main__":
    main()
