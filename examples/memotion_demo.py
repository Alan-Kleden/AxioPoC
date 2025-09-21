# examples/memotion_demo.py
import argparse, math
from pathlib import Path
from memotion_decay.decay import memotion_decay, DecayParams

def parse_args():
    p = argparse.ArgumentParser("Tracer v(t) pour mé-émotions ± avec asymétrie")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--lambda-pos", type=float, default=0.06, dest="lambda_pos")
    p.add_argument("--lambda-neg", type=float, default=0.03, dest="lambda_neg")
    p.add_argument("--vpos0", type=float, default=1.0)
    p.add_argument("--vneg0", type=float, default=-1.0)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--save-fig", type=str, default="out/memotion_decay.png")
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()

def main():
    a = parse_args()
    P = DecayParams(lambda_pos=a.lambda_pos, lambda_neg=a.lambda_neg)
    t = list(range(a.steps + 1))
    vpos = [memotion_decay(a.vpos0, k, P) for k in t]
    vneg = [memotion_decay(a.vneg0, k, P) for k in t]

    if a.plot or a.save_fig:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise SystemExit("Installe matplotlib : python -m pip install matplotlib")
        fig = plt.figure(figsize=(6,4))
        plt.plot(t, vpos, marker="o", label=f"v0={a.vpos0} (λ+= {a.lambda_pos})")
        plt.plot(t, vneg, marker="o", label=f"v0={a.vneg0} (λ−= {a.lambda_neg})")
        plt.axhline(0, linewidth=1)
        plt.xlabel("t"); plt.ylabel("v(t)"); plt.title("Décroissance asymétrique des mémotions")
        plt.grid(True); plt.legend()
        if a.save_fig:
            out = Path(a.save_fig); out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=a.dpi, bbox_inches="tight")
        if a.plot and not a.save_fig: plt.show()
        plt.close()
    else:
        for k in t:
            print(f"{k}\t{vpos[k]:+.3f}\t{vneg[k]:+.3f}")

if __name__ == "__main__":
    main()
