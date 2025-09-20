# examples/ntel_chain.py — v2 (seuils, weights, export CSV)
import argparse
import math
import csv
from pathlib import Path
from typing import List, Dict, Optional

from memotion_decay.decay import memotion_decay, DecayParams
from negentropy_telotopic import (
    ntel_from_radians,
    classify_coherence,
    CoherenceThresholds,
)

# 4 acteurs par défaut : angle initial (degrés) + valence initiale v0
DEFAULT_AGENTS = [
    {"theta0_deg": 0.0,   "v0":  1.0},
    {"theta0_deg": 5.0,   "v0":  0.8},
    {"theta0_deg": -3.0,  "v0":  0.6},
    {"theta0_deg": 180.0, "v0": -1.0},  # opposant pour créer de l'ambivalence
]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Chaînage memotion_decay -> angles(t) -> R(t) + label"
    )
    p.add_argument("--steps", type=int, default=20, help="Nombre d'étapes t=0..steps")
    # Biais de négativité par défaut : négatif plus persistant (lambda_neg < lambda_pos)
    p.add_argument("--lambda-pos", type=float, default=0.06, dest="lambda_pos",
                   help="Taux de décroissance (valences positives)")
    p.add_argument("--lambda-neg", type=float, default=0.03, dest="lambda_neg",
                   help="Taux de décroissance (valences négatives)")
    # Encodage initial asymétrique (alpha_neg > alpha_pos)
    p.add_argument("--alpha-pos", type=float, default=1.0, help="Facteur d'encodage positif")
    p.add_argument("--alpha-neg", type=float, default=1.6, help="Facteur d'encodage négatif")
    # Drift angulaire proportionnel à v(t)
    p.add_argument("--drift-gain", type=float, default=0.15,
                   help="Gain (rad) appliqué à la valence pour le drift angulaire")
    # Nouveaux paramètres v2
    p.add_argument("--high", "--thr-high", dest="high", type=float, default=0.70,
                   help="Seuil de cohérence élevée (par défaut 0.70)")
    p.add_argument("--medium", "--thr-medium", dest="medium", type=float, default=0.40,
                   help="Seuil de cohérence moyenne (par défaut 0.40)")
    p.add_argument("--weights", type=str, default=None,
                   help='Poids des acteurs, ex: "1,1,1,2" ou "1 1 1 2" (longueur = nb d’acteurs)')
    p.add_argument("--save-csv", type=str, default=None,
                   help="Chemin d'export CSV de la série (t,R,label,angles)")
    p.add_argument("--no-header", action="store_true", help="N'affiche pas l'entête")
    p.add_argument("--plot", action="store_true",
                   help="Trace R(t) (nécessite matplotlib, non installé par défaut)")
    return p.parse_args()

def build_agents(base: List[Dict]) -> List[Dict]:
    agents = []
    for a in base:
        agents.append({
            "theta0": math.radians(float(a["theta0_deg"])),
            "v0": float(a["v0"]),
        })
    return agents

def encode_v0(v0: float, alpha_pos: float, alpha_neg: float) -> float:
    """Encodage initial asymétrique : négatif amplifié si alpha_neg > alpha_pos."""
    return alpha_pos * v0 if v0 >= 0 else alpha_neg * v0

def parse_weights_arg(s: Optional[str], n: int) -> Optional[List[float]]:
    if not s:
        return None
    # accepte "1,2,3" ou "1 2 3"
    items = []
    for chunk in s.split(","):
        items.extend(chunk.split())
    w = [float(x) for x in items if x != ""]
    if len(w) != n:
        raise SystemExit(f"--weights: attendu {n} valeurs, obtenu {len(w)}")
    return w

def step_angles(agents: List[Dict], t: int, p: DecayParams, drift_gain: float,
                alpha_pos: float, alpha_neg: float):
    """Liste des angles (radians) au temps t, après encodage & décroissance."""
    thetas = []
    for a in agents:
        v0_eff = encode_v0(a["v0"], alpha_pos, alpha_neg)
        v_t = memotion_decay(v0_eff, t, p)   # valence résiduelle
        drift = drift_gain * v_t             # petit déplacement angulaire
        thetas.append(a["theta0"] + drift)
    return thetas

def main():
    args = parse_args()
    p = DecayParams(lambda_pos=args.lambda_pos, lambda_neg=args.lambda_neg)
    agents = build_agents(DEFAULT_AGENTS)
    n = len(agents)
    weights = parse_weights_arg(args.weights, n)
    th = CoherenceThresholds(high=args.high, medium=args.medium)

    if not args.no_header:
        print("t\tR(t)\tlabel\t\tangles_deg")

    t_values, r_values = [], []

    # Prépare CSV si demandé
    writer = None
    csv_file = None
    if args.save_csv:
        out_path = Path(args.save_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)  # ✅ crée le dossier si absent
        fieldnames = ["t", "R", "label"] + [f"angle{i}_deg" for i in range(n)]
        if weights is not None:
            fieldnames += [f"w{i}" for i in range(n)]
        csv_file = out_path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    try:
        for t in range(0, args.steps + 1):
            thetas_t = step_angles(agents, t, p, args.drift_gain, args.alpha_pos, args.alpha_neg)
            R_t = ntel_from_radians(thetas_t, weights=weights)
            label = classify_coherence(R_t, th)
            degs = [round(math.degrees(x), 1) for x in thetas_t]

            if not args.no_header:
                print(f"{t}\t{R_t:.3f}\t{label:8s}\t{degs}")

            t_values.append(t)
            r_values.append(R_t)

            if writer:
                row = {"t": t, "R": R_t, "label": label}
                for i, d in enumerate(degs):
                    row[f"angle{i}_deg"] = d
                if weights is not None:
                    for i, w in enumerate(weights):
                        row[f"w{i}"] = w
                writer.writerow(row)
    finally:
        if csv_file:
            csv_file.close()

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise SystemExit("Matplotlib n'est pas installé. Fais: python -m pip install matplotlib")
        plt.figure()
        plt.plot(t_values, r_values, marker="o")
        plt.xlabel("t")
        plt.ylabel("R(t)")
        plt.title("Cohérence télotopique R(t)")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()
