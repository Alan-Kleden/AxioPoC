# examples/ntel_examples.py
import math, argparse
from pathlib import Path
import matplotlib.pyplot as plt
from negentropy_telotopic import ntel_from_degrees

def polar_plot(angles_deg, R, title, out_path):
    th = [math.radians(a) for a in angles_deg]
    r = [1.0]*len(th)
    fig = plt.figure(figsize=(5,5))
    ax = plt.subplot(111, projection="polar")
    ax.scatter(th, r)
    ax.set_rticks([]); ax.set_title(f"{title}\nR = {R:.3f}", va="bottom")
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180, bbox_inches="tight"); plt.close()

def main():
    p = argparse.ArgumentParser("Génère trois images : haute/moyenne/faible cohérence")
    p.add_argument("--out-dir", type=str, default="out")
    args = p.parse_args()

    # 1) élevée : angles serrés autour de 0°
    high = [0, 2, -1, 3, -2, 1, 4, -3, 2, 0]
    R = ntel_from_degrees(high)
    polar_plot(high, R, "Cohérence élevée", f"{args.out_dir}/ntel_high.png")

    # 2) moyenne : deux grappes ±25°
    mid = [-25,-23,-20,-18, 0, 0, 18,20,23,25]
    R = ntel_from_degrees(mid)
    polar_plot(mid, R, "Cohérence moyenne", f"{args.out_dir}/ntel_mid.png")

    # 3) faible : moitié 0°, moitié 180°
    low = [0,0,0,0, 180,180,180,180]
    R = ntel_from_degrees(low)
    polar_plot(low, R, "Cohérence faible", f"{args.out_dir}/ntel_low.png")

if __name__ == "__main__":
    main()
