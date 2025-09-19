# negentropy_telotopic/__main__.py
import argparse, math
from .ntel import ntel_from_degrees, ntel_from_radians, classify_coherence, CoherenceThresholds

def parse_floats(xs): return [float(x) for x in xs]

def main():
    p = argparse.ArgumentParser(description="Compute telotopic negentropy R and coherence label.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--deg", nargs="+", help="angles in degrees", metavar="A")
    g.add_argument("--rad", nargs="+", help="angles in radians", metavar="A")
    p.add_argument("--weights", nargs="+", help="optional weights", metavar="W")
    p.add_argument("--high", type=float, default=0.70, help="high coherence threshold (default: 0.70)")
    p.add_argument("--medium", type=float, default=0.40, help="medium coherence threshold (default: 0.40)")
    args = p.parse_args()

    th = CoherenceThresholds(high=args.high, medium=args.medium)

    weights = parse_floats(args.weights) if args.weights else None
    if args.deg:
        angles = parse_floats(args.deg)
        R = ntel_from_degrees(angles, weights=weights)
    else:
        angles = parse_floats(args.rad)
        R = ntel_from_radians(angles, weights=weights)

    label = classify_coherence(R, th)
    print(f"R = {R:.6f}  |  coherence = {label}")

if __name__ == "__main__":
    main()
