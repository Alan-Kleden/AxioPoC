#!/usr/bin/env python3
import argparse, csv, random, tempfile, shutil, os, subprocess, sys, json, statistics as st

def run_auc(feat_src, raw_src, sig_src, feat_tgt, raw_tgt, sig_tgt):
    cmd = [sys.executable, "./scripts/eval_da_transfer.py",
           "--feat-src", feat_src, "--raw-src", raw_src, "--signature-src", sig_src,
           "--feat-tgt", feat_tgt, "--raw-tgt", raw_tgt, "--signature-tgt", sig_tgt,
           "--metric", "diag"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + ("\n"+p.stderr if p.stderr else "")
    if p.returncode != 0:
        raise SystemExit(out)
    # parse "AUC 0.525475, ACC 0.487500"
    for line in out.splitlines():
        if "AUC" in line and "ACC" in line:
            # try robust parse
            try:
                parts = line.replace(",", " ").split()
                auc_i = parts.index("AUC") + 1
                return float(parts[auc_i])
            except Exception:
                pass
    raise SystemExit("Impossible d'extraire l'AUC depuis:\n"+out)

def shuffle_labels_csv(in_csv, out_csv, seed=None):
    rng = random.Random(seed)
    with open(in_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    labels = [r["label"] for r in rows]
    rng.shuffle(labels)
    for r, lab in zip(rows, labels):
        r["label"] = lab
    with open(out_csv, "w", newline="", encoding="utf-8") as g:
        w = csv.DictWriter(g, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-src", required=True)
    ap.add_argument("--raw-src", required=True)
    ap.add_argument("--sig-src", required=True)
    ap.add_argument("--feat-tgt", required=True)
    ap.add_argument("--raw-tgt", required=True)
    ap.add_argument("--sig-tgt", required=True)
    ap.add_argument("--iters", type=int, default=200, help="nb permutations (200-1000)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--outjson", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # AUC réel (labels réels)
    auc_real = run_auc(args.feat_src, args.raw_src, args.sig_src,
                       args.feat_tgt, args.raw_tgt, args.sig_tgt)

    # Permutations
    tmpdir = tempfile.mkdtemp(prefix="perm_")
    try:
        null_aucs = []
        for i in range(args.iters):
            tmp_raw = os.path.join(tmpdir, f"raw_perm_{i}.csv")
            shuffle_labels_csv(args.raw_tgt, tmp_raw, seed=rng.randint(0, 10**9))
            auc_i = run_auc(args.feat_src, args.raw_src, args.sig_src,
                            args.feat_tgt, tmp_raw, args.sig_tgt)
            null_aucs.append(auc_i)

        # p-value unilatérale P(AUC_null >= AUC_real)
        p = sum(1 for x in null_aucs if x >= auc_real) / len(null_aucs)

        # CI bootstrap approx (percentiles)
        null_sorted = sorted(null_aucs)
        def pct(v, p):  # simple percentile
            k = int(p*(len(v)-1))
            return v[k]
        null_ci = [pct(null_sorted, 0.025), pct(null_sorted, 0.975)]

        payload = {
            "setting": "transfer",
            "direction": "afd->cmv",
            "signature": "Da13_diag",
            "auc_real": auc_real,
            "null_mean": st.mean(null_aucs),
            "null_sd": st.pstdev(null_aucs),
            "p_value_one_sided_auc>=real": p,
            "null_ci_95": null_ci,
            "iters": args.iters,
        }
        with open(args.outjson, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[OK] Permutation test -> {args.outjson}")
        print(f"AUC_real={auc_real:.6f} | p(one-sided)={p:.4f} | null_mean={payload['null_mean']:.3f} ± {payload['null_sd']:.3f}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    main()
