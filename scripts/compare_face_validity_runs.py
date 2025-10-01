# C:\AxioPoC\scripts\compare_face_validity_runs.py
import argparse, json
from pathlib import Path
import math
import pandas as pd

def load_summary_from_json(run_dir: Path):
    js = run_dir / "summary.json"
    if js.exists():
        j = json.loads(js.read_text(encoding="utf-8"))
        # Normalise quelques champs
        return {
            "run_dir": str(run_dir),
            "n_total_msgs": j.get("n_total_msgs"),
            "topk": j.get("topk"),
            "top_pos": (j.get("top_counts_[label1,label0]") or [None, None])[0],
            "top_neg": (j.get("top_counts_[label1,label0]") or [None, None])[1],
            "bot_pos": (j.get("bot_counts_[label1,label0]") or [None, None])[0],
            "bot_neg": (j.get("bot_counts_[label1,label0]") or [None, None])[1],
            "p_value": j.get("p_value"),
            "chi2": j.get("chi2"),
            "odds_ratio": j.get("odds_ratio"),
            "delta_pp_label1": j.get("delta_pp_label1_top_minus_bottom"),
            "test": j.get("test", "chi2"),
        }
    return None

def compute_from_csvs(run_dir: Path):
    from scipy.stats import chi2_contingency  # suppose dispo dans ton venv
    top = pd.read_csv(run_dir / "top_threads_messages.csv")
    bot = pd.read_csv(run_dir / "bottom_threads_messages.csv")
    # On tolère label float/str
    def is_pos(v):
        try:
            return int(v) == 1
        except Exception:
            return str(v).strip() == "1"
    top_pos = int(sum(is_pos(x) for x in top["label"]))
    top_neg = int(len(top) - top_pos)
    bot_pos = int(sum(is_pos(x) for x in bot["label"]))
    bot_neg = int(len(bot) - bot_pos)

    table = [[top_pos, top_neg],
             [bot_pos, bot_neg]]
    chi2, p, dof, exp = chi2_contingency(table, correction=False)

    def safe_or(tp, tn, bp, bn):
        # OR = (tp * bn) / (tn * bp)
        num = tp * bn
        den = tn * bp
        if den == 0: 
            return None
        return num / den

    def delta_pp(tp, tn, bp, bn):
        top_pp = 100.0 * tp / (tp+tn) if (tp+tn)>0 else float("nan")
        bot_pp = 100.0 * bp / (bp+bn) if (bp+bn)>0 else float("nan")
        return top_pp - bot_pp

    return {
        "run_dir": str(run_dir),
        "n_total_msgs": None,
        "topk": len(top),
        "top_pos": top_pos,
        "top_neg": top_neg,
        "bot_pos": bot_pos,
        "bot_neg": bot_neg,
        "p_value": p,
        "chi2": chi2,
        "odds_ratio": safe_or(top_pos, top_neg, bot_pos, bot_neg),
        "delta_pp_label1": delta_pp(top_pos, top_neg, bot_pos, bot_neg),
        "test": "chi2"
    }

def infer_run_name(run_dir: Path):
    # essaie de déduire mean/max/dot depuis le nom du dossier
    n = run_dir.name.lower()
    score = "cos" if "cos" in n or "dot" not in n else "dot"
    agg = "max" if "max" in n else ("mean" if "mean" in n else "mean")
    return f"{score}_{agg}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="Liste de dossiers de runs (chacun contenant top_threads_messages.csv, bottom_threads_messages.csv, et idéalement summary.json)")
    ap.add_argument("--outcsv", required=True, help="Chemin de sortie CSV récapitulatif")
    ap.add_argument("--outmd", default=None, help="(Optionnel) Chemin .md pour un tableau Markdown")
    args = ap.parse_args()

    rows = []
    for r in args.runs:
        rdir = Path(r)
        if not rdir.exists():
            print(f"[WARN] run dir not found: {r}")
            continue
        # 1) tente summary.json
        summ = load_summary_from_json(rdir)
        # 2) sinon calcule depuis CSV
        if summ is None:
            summ = compute_from_csvs(rdir)
        # run label
        summ["run"] = infer_run_name(rdir)
        # arrondis lisibles
        for k in ["p_value","chi2","odds_ratio","delta_pp_label1"]:
            if summ.get(k) is not None and not (isinstance(summ[k], float) and (math.isnan(summ[k]) or math.isinf(summ[k]))):
                summ[k] = float(summ[k])
        rows.append(summ)

    if not rows:
        print("[ERROR] Aucun run lisible.")
        return

    df = pd.DataFrame(rows, columns=[
        "run","run_dir","topk","top_pos","top_neg","bot_pos","bot_neg",
        "test","chi2","p_value","odds_ratio","delta_pp_label1"
    ])
    # tri esthétique: dot_mean, cos_mean, cos_max…
    df["sort_key"] = df["run"].map(lambda x: {"dot_mean":0,"cos_mean":1,"cos_max":2}.get(x, 99))
    df = df.sort_values(["sort_key","run"]).drop(columns=["sort_key"])

    outcsv = Path(args.outcsv)
    outcsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outcsv, index=False)
    print(f"[OK] Summary -> {outcsv}")

    if args.outmd:
        outmd = Path(args.outmd)
        outmd.parent.mkdir(parents=True, exist_ok=True)
        # format markdown simple
        md = ["| run | topk | top_pos | top_neg | bot_pos | bot_neg | test | chi2 | p_value | OR | Δpp(label=1) |",
              "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
        for _,row in df.iterrows():
            md.append(f"| {row['run']} | {row['topk']} | {row['top_pos']} | {row['top_neg']} | "
                      f"{row['bot_pos']} | {row['bot_neg']} | {row['test']} | "
                      f"{row['chi2']:.4f} | {row['p_value']:.6f} | "
                      f"{('' if pd.isna(row['odds_ratio']) else f'{row['odds_ratio']:.3f}')} | "
                      f"{row['delta_pp_label1']:.1f} |")
        outmd.write_text("\n".join(md), encoding="utf-8")
        print(f"[OK] Markdown -> {outmd}")

if __name__ == "__main__":
    main()
