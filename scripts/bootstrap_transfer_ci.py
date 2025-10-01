#!/usr/bin/env python3
import argparse, csv, random, tempfile, os, subprocess, sys, json, statistics as st

def run_auc(feat_src, raw_src, sig_src, feat_tgt, raw_tgt, sig_tgt):
    cmd = [sys.executable, "./scripts/eval_da_transfer.py",
           "--feat-src", feat_src, "--raw-src", raw_src, "--signature-src", sig_src,
           "--feat-tgt", feat_tgt, "--raw-tgt", raw_tgt, "--signature-tgt", sig_tgt,
           "--metric", "diag"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + ("\n"+p.stderr if p.stderr else "")
    if p.returncode != 0:
        raise SystemExit(out)
    for line in out.splitlines():
        if "AUC" in line and "ACC" in line:
            try:
                parts = line.replace(",", " ").split()
                auc_i = parts[parts.index("AUC")+1]
                return float(auc_i)
            except Exception:
                pass
    raise SystemExit("AUC non parsée:\n"+out)

def resample_rows(in_csv, out_csv, n, rng):
    with open(in_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    sample = [rng.choice(rows) for _ in range(n)]
    with open(out_csv, "w", newline="", encoding="utf-8") as g:
        w = csv.DictWriter(g, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(sample)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-src", required=True)
    ap.add_argument("--raw-src", required=True)
    ap.add_argument("--sig-src", required=True)
    ap.add_argument("--feat-tgt", required=True)
    ap.add_argument("--raw-tgt", required=True)
    ap.add_argument("--sig-tgt", required=True)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--outjson", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    # récupère n_tgt
    import pandas as pd
    n_tgt = len(pd.read_csv(args.raw_tgt))

    aucs = []
    tmp = tempfile.mkdtemp(prefix="boot_")
    try:
        for i in range(args.iters):
            raw_b = os.path.join(tmp, f"raw_boot_{i}.csv")
            resample_rows(args.raw_tgt, raw_b, n_tgt, rng)
            auc_i = run_auc(args.feat_src, args.raw_src, args.sig_src,
                            args.feat_tgt, raw_b, args.sig_tgt)
            aucs.append(auc_i)

        aucs_sorted = sorted(aucs)
        def pct(v, p): 
            k = int(p*(len(v)-1))
            return v[k]
        ci = [pct(aucs_sorted, 0.025), pct(aucs_sorted, 0.975)]
        payload = {
            "setting": "transfer",
            "direction": "afd->cmv",
            "signature": "Da13_diag",
            "bootstrap_mean": st.mean(aucs),
            "bootstrap_sd": st.pstdev(aucs),
            "ci_95": ci,
            "iters": args.iters
        }
        with open(args.outjson, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[OK] Bootstrap CI -> {args.outjson}")
        print(f"Mean={payload['bootstrap_mean']:.3f} ± {payload['bootstrap_sd']:.3f} | 95% CI={ci[0]:.3f}..{ci[1]:.3f}")
    finally:
        import shutil; shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    main()
