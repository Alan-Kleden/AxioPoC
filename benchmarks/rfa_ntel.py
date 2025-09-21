# benchmarks/rfa_ntel.py
"""
Benchmark A — Consensus de groupe via N_tel (R / cos² / entropie)

Entrée CSV minimale :
- thread_id : identifiant de fil
- stance    : -1 / 0 / +1 (ou texte 'support/neutral/oppose' -> mappé)
- outcome   : 1/0 (ou texte 'accepted/rejected' -> mappé)
- timestamp : (optionnel) parsable par pandas.to_datetime
- index_in_thread : (optionnel) index entier; sera créé si --msg-window et absent

Deux modes d’agrégation par fenêtre :
- --time-window 24H   (utilise 'timestamp')
- --msg-window 20     (utilise 'index_in_thread')

Métrique de cohérence (sélectionnable) :
- --ntel-mode R        -> mean resultant length (par défaut)
- --ntel-mode cos2     -> moyenne pondérée de cos² autour d’un télos
- --ntel-mode entropy  -> N_tel = 1 - H(p_i), p_i ∝ w_i * cos²(Δ_i)

Options télos :
- --thetaT-deg <deg>   -> télos imposé (degrés). Sinon: angle du résultant local (endogène).
- --signed             -> version "signée" pour cos2/entropy (antipodal ignoré via max(cos,0)^2)
- --log-base <b>       -> base du log pour l’entropie (2.0 par défaut)
- --no-normalize       -> entropie non normalisée (sinon normalisée par log(n_actif))

NOUVEAU :
- --early-frac f       -> n’utiliser que la fraction initiale f ∈ (0,1] de chaque fil pour les fenêtres/features
- --support-band a b   -> évaluer uniquement les fils avec pct_support ∈ [a,b] (sauvegarde des features complète inchangée)

Sorties :
- AUC/ACC en CV (Features_R vs Baselines vs Combine)
- --save-features path.csv : enregistre les features par fil (toujours “complètes”, avant banding)
- --plot-mean-rt : figure du "metric(t)" moyen par classe (sur l’échantillon évalué)
"""

import argparse
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Iterable
from numbers import Number

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

# Import des métriques depuis le package
from negentropy_telotopic import (
    ntel_from_radians,           # mode R (MRL)
    ntel_cos2_from_radians,      # mode cos²
    ntel_entropy_from_radians,   # mode entropie
    telos_angle_from_resultant,  # utilitaire télos endogène (si besoin ailleurs)
)

warnings.filterwarnings("ignore", category=FutureWarning)


# -----------------------------
# Utilitaires mapping & calcul
# -----------------------------

def parse_stance(x):
    """Map text or numeric stance to {-1,0,+1}. Fallback: try float sign."""
    if pd.isna(x):
        return 0
    if isinstance(x, (int, float)):
        v = float(x)
        if v > 0:
            return 1
        if v < 0:
            return -1
        return 0
    s = str(x).strip().lower()
    if any(k in s for k in ["support", "pour", "upvote", "+"]):
        return 1
    if any(k in s for k in ["oppose", "contre", "downvote", "-"]):
        return -1
    if any(k in s for k in ["neutral", "neutre", "abstain", "abstention"]):
        return 0
    # fallback: numeric
    try:
        v = float(s)
        if v > 0:
            return 1
        if v < 0:
            return -1
        return 0
    except Exception:
        return 0


def stance_to_angle_default(s: int) -> float:
    """θ = π*(1-s)/2 with s in {-1,0,+1} -> {π, π/2, 0}."""
    return math.pi * (1 - s) / 2


def stance_to_angle_alt(s: int) -> float:
    """Alternative robustness: θ = arccos(s) with s in {-1,0,+1}."""
    return math.acos(float(s))


def entropy_stance(values: List[int]) -> float:
    """Shannon entropy (nats) over {-1,0,1} distribution."""
    if len(values) == 0:
        return 0.0
    arr = np.asarray(values)
    counts = np.array([
        np.sum(arr == -1),
        np.sum(arr == 0),
        np.sum(arr == 1)
    ], dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def windows_by_time(df_thread: pd.DataFrame, freq: str, min_n: int = 3):
    """Yield (t_center, stances_in_window) for time windows (e.g., '24H')."""
    df = df_thread.sort_values("timestamp").copy()
    df = df.set_index("timestamp")
    for key, g in df.groupby(pd.Grouper(freq=freq)):
        if g.shape[0] < min_n:
            continue
        center = key
        yield center, g["stance"].tolist()


def windows_by_count(df_thread: pd.DataFrame, k: int, min_n: int = 3):
    """Yield (idx, stances_in_block) for contiguous non-overlapping blocks of size k."""
    st = df_thread.sort_values("index_in_thread").reset_index(drop=True)
    n = st.shape[0]
    for start in range(0, n, k):
        end = min(start + k, n)
        block = st.iloc[start:end]
        if block.shape[0] < min_n:
            continue
        yield start, block["stance"].tolist()


# --- Sélection de la métrique de cohérence ---

def coherence_score(angles_rad: Iterable[Number],
                    mode: str,
                    weights=None,
                    thetaT_deg: Optional[float] = None,
                    signed: bool = False,
                    log_base: float = 2.0,
                    normalize: bool = True) -> float:
    """
    Calcule la cohérence selon le mode choisi.
    - R        : mean resultant length
    - cos2     : moyenne pondérée de cos² autour d’un télos
    - entropy  : N_tel = 1 - H(p_i), p_i ∝ w_i * cos²(Δ_i)
    """
    if mode == "R":
        return ntel_from_radians(angles_rad, weights=weights)

    theta_T = None if thetaT_deg is None else math.radians(thetaT_deg)

    if mode == "cos2":
        return ntel_cos2_from_radians(
            angles_rad,
            weights=weights,
            theta_T=theta_T,
            signed=signed,
        )
    if mode == "entropy":
        return ntel_entropy_from_radians(
            angles_rad,
            weights=weights,
            theta_T=theta_T,
            signed=signed,
            log_base=log_base,
            normalize=normalize,
        )
    raise ValueError(f"Unknown mode: {mode}")


# --- Nouveau : fraction initiale ---

def take_early_fraction(df_thread: pd.DataFrame,
                        early_frac: Optional[float],
                        use_time: bool) -> pd.DataFrame:
    """
    Retourne un sous-ensemble "début de fil" selon early_frac.
    - si use_time=True : tranche par timestamp entre [t_min, t_min + f*(t_max - t_min)]
    - sinon            : tranche par index_in_thread (si absent, crée un rang temporaire)
    """
    if early_frac is None or early_frac >= 1.0:
        return df_thread
    if early_frac <= 0.0:
        return df_thread.iloc[0:0]

    df = df_thread.copy()

    if use_time and "timestamp" in df.columns:
        tmin = df["timestamp"].min()
        tmax = df["timestamp"].max()
        if pd.isna(tmin) or pd.isna(tmax) or tmin == tmax:
            # pas de dispersion temporelle exploitable → fallback par index
            use_time = False
        else:
            cutoff = tmin + (tmax - tmin) * early_frac
            return df[df["timestamp"] <= cutoff]

    # par index
    if "index_in_thread" not in df.columns:
        df = df.sort_values("timestamp" if "timestamp" in df.columns else "stance").copy()
        df["__tmp_idx__"] = np.arange(df.shape[0])
        max_idx = df["__tmp_idx__"].max() if df.shape[0] > 0 else 0
        cutoff = int(math.floor(max_idx * early_frac))
        out = df[df["__tmp_idx__"] <= cutoff].drop(columns=["__tmp_idx__"])
        return out

    max_idx = df["index_in_thread"].max() if df.shape[0] > 0 else 0
    cutoff = int(math.floor(max_idx * early_frac))
    return df[df["index_in_thread"] <= cutoff]


def R_series_for_thread(df_thread: pd.DataFrame,
                        time_window: Optional[str],
                        msg_window: Optional[int],
                        mapping: str = "default",
                        min_n: int = 3,
                        ARGS: Optional[dict] = None,
                        early_frac: Optional[float] = None) -> List[Tuple[float, float]]:
    """
    Retourne la liste de (t_idx_numeric, metric) pour un fil donné,
    où 'metric' est la cohérence choisie (R / cos² / entropy),
    calculée éventuellement sur la **première fraction** du fil.
    """
    if ARGS is None:
        ARGS = {
            "mode": "R",
            "thetaT_deg": None,
            "signed": False,
            "log_base": 2.0,
            "normalize": True,
        }

    f_map = stance_to_angle_default if mapping == "default" else stance_to_angle_alt
    series: List[Tuple[float, float]] = []

    # Sous-échantillonnage "début de fil"
    use_time = (time_window is not None and "timestamp" in df_thread.columns)
    df_local = take_early_fraction(df_thread, early_frac, use_time=use_time)

    # Pas de données après découpe
    if df_local.shape[0] == 0:
        return []

    if time_window is not None and "timestamp" in df_local.columns:
        for t, stances in windows_by_time(df_local, time_window, min_n=min_n):
            thetas = [f_map(parse_stance(s)) for s in stances]
            metric = coherence_score(
                thetas,
                mode=ARGS["mode"],
                weights=None,
                thetaT_deg=ARGS["thetaT_deg"],
                signed=ARGS["signed"],
                log_base=ARGS["log_base"],
                normalize=ARGS["normalize"],
            )
            series.append((float(pd.Timestamp(t).value), metric))  # temps numérique (ns)
    elif msg_window is not None and "index_in_thread" in df_local.columns:
        for idx, stances in windows_by_count(df_local, msg_window, min_n=min_n):
            thetas = [f_map(parse_stance(s)) for s in stances]
            metric = coherence_score(
                thetas,
                mode=ARGS["mode"],
                weights=None,
                thetaT_deg=ARGS["thetaT_deg"],
                signed=ARGS["signed"],
                log_base=ARGS["log_base"],
                normalize=ARGS["normalize"],
            )
            series.append((float(idx), metric))
    else:
        # fallback: un seul score sur le sous-ensemble early
        stances = [parse_stance(s) for s in df_local["stance"].tolist()]
        thetas = [f_map(s) for s in stances]
        metric = coherence_score(
            thetas,
            mode=ARGS["mode"],
            weights=None,
            thetaT_deg=ARGS["thetaT_deg"],
            signed=ARGS["signed"],
            log_base=ARGS["log_base"],
            normalize=ARGS["normalize"],
        )
        series.append((0.0, metric))

    return series


def features_from_R_series(series: List[Tuple[float, float]]) -> Tuple[float, float, float]:
    """Given list of (t, metric), return (R_mean, R_last, R_slope)."""
    if len(series) == 0:
        return (0.0, 0.0, 0.0)
    series = sorted(series, key=lambda x: x[0])
    t = np.array([x[0] for x in series], dtype=float)
    r = np.array([x[1] for x in series], dtype=float)
    R_mean = float(r.mean())
    R_last = float(r[-1])
    slope = float(np.polyfit(t, r, 1)[0]) if len(r) >= 2 else 0.0
    return (R_mean, R_last, slope)


def outcome_to_label(x) -> int:
    """Map outcome text/num to {0,1} with 1=accepted/success."""
    if isinstance(x, (int, float)):
        return 1 if float(x) > 0 else 0
    s = str(x).strip().lower()
    if s in {"accept", "accepted", "success", "successful", "yes", "true", "1"}:
        return 1
    return 0


@dataclass
class EvalResult:
    name: str
    auc: float
    acc: float


# -----------------------------
# Évaluation
# -----------------------------

def eval_cv(X: np.ndarray, y: np.ndarray, name: str, n_splits: int = 5, seed: int = 42) -> EvalResult:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs, accs = [], []
    for tr, te in skf.split(X, y):
        model = LogisticRegression(max_iter=1000)
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        yhat = (p >= 0.5).astype(int)
        aucs.append(roc_auc_score(y[te], p))
        accs.append(accuracy_score(y[te], yhat))
    return EvalResult(name, float(np.mean(aucs)), float(np.mean(accs)))


def plot_mean_R_curves(per_thread_series, per_thread_label, out_png: Optional[str] = None, metric_name: str = "R"):
    """Plot mean metric(t) over normalized steps for class 0 vs 1."""
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit("Matplotlib n'est pas installé. Fais: python -m pip install matplotlib") from e

    target_steps = 20

    def interp_to(Rs):
        xs = np.linspace(0, 1, num=len(Rs))
        xt = np.linspace(0, 1, num=target_steps)
        return np.interp(xt, xs, Rs)

    pos, neg = [], []
    for tid, series in per_thread_series.items():
        r = [val for _, val in sorted(series, key=lambda x: x[0])]
        if len(r) < 2:
            continue
        arr = interp_to(r)
        if per_thread_label[tid] == 1:
            pos.append(arr)
        else:
            neg.append(arr)

    import matplotlib.pyplot as plt  # noqa
    plt.figure(figsize=(6, 4))
    if len(pos) > 0:
        plt.plot(np.linspace(0, 1, target_steps), np.mean(pos, axis=0), label="accepted")
    if len(neg) > 0:
        plt.plot(np.linspace(0, 1, target_steps), np.mean(neg, axis=0), label="rejected")
    plt.xlabel("normalized progress")
    plt.ylabel(metric_name)
    plt.title(f"Mean {metric_name}(t) by class")
    plt.grid(True)
    plt.legend()
    if out_png:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=180, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


# -----------------------------
# Main pipeline
# -----------------------------

def main():
    ap = argparse.ArgumentParser("Benchmark A — Consensus via N_tel (R / cos² / entropie)")
    ap.add_argument("--input", required=True, help="CSV with columns: thread_id, stance, outcome, and timestamp or index_in_thread")
    ap.add_argument("--time-window", type=str, default=None, help="e.g., 24H or 12H (uses timestamp)")
    ap.add_argument("--msg-window", type=int, default=None, help="e.g., 20 (uses index_in_thread)")
    ap.add_argument("--min-n", type=int, default=3, help="Min messages per window")
    ap.add_argument("--mapping", choices=["default", "alt"], default="default", help="s→θ mapping")

    # Métrique de cohérence
    ap.add_argument("--ntel-mode", choices=["R", "cos2", "entropy"], default="R",
                    help="Choix de la métrique de cohérence (R par défaut).")
    ap.add_argument("--thetaT-deg", type=float, default=None,
                    help="Télos en degrés (si omis, télos = angle du résultant local).")
    ap.add_argument("--signed", action="store_true",
                    help="Version signée (ignore l'antipodal) pour cos2/entropy.")
    ap.add_argument("--log-base", type=float, default=2.0,
                    help="Base logarithmique pour l'entropie (2 par défaut).")
    ap.add_argument("--no-normalize", action="store_true",
                    help="Ne pas normaliser l'entropie (par log(n)).")

    # NOUVEAU
    ap.add_argument("--early-frac", type=float, default=None,
                    help="Fraction initiale de chaque fil à utiliser (0<f<=1). Ex: 0.5 = première moitié.")
    ap.add_argument("--support-band", nargs=2, type=float, default=None, metavar=("LOW", "HIGH"),
                    help="Évaluer uniquement les fils avec pct_support ∈ [LOW, HIGH].")

    ap.add_argument("--save-features", type=str, default=None, help="Path to save features CSV (features complètes)")
    ap.add_argument("--plot-mean-rt", action="store_true", help="Plot mean metric(t) per class")
    ap.add_argument("--plot-out", type=str, default="out/mean_rt.png", help="Where to save plot if plotting")
    args = ap.parse_args()

    # Validation des nouveaux flags
    if args.early_frac is not None and not (0.0 < args.early_frac <= 1.0):
        print("ERROR: --early-frac doit être dans (0,1].", file=sys.stderr)
        sys.exit(1)
    if args.support_band is not None:
        a, b = args.support_band
        if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0 and a <= b):
            print("ERROR: --support-band LOW HIGH doivent vérifier 0<=LOW<=HIGH<=1.", file=sys.stderr)
            sys.exit(1)

    df = pd.read_csv(args.input)

    # Vérification colonnes minimales
    missing = [c for c in ["thread_id", "stance", "outcome"] if c not in df.columns]
    if missing:
        print(f"ERROR: CSV must contain columns: thread_id, stance, outcome. Missing: {missing}", file=sys.stderr)
        sys.exit(1)

    # Parse stance & outcome
    df["stance"] = df["stance"].apply(parse_stance)
    df["outcome_bin"] = df["outcome"].apply(outcome_to_label)

    # Prépare timestamp / index si requis
    if args.time_window is not None:
        if "timestamp" not in df.columns:
            print("ERROR: --time-window requires a 'timestamp' column.", file=sys.stderr)
            sys.exit(1)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
    if args.msg_window is not None:
        if "index_in_thread" not in df.columns:
            df = df.sort_values(["thread_id"]).copy()
            df["index_in_thread"] = df.groupby("thread_id").cumcount()

    # Par fil : série metric(t) + features + baselines
    rows = []
    per_thread_series = {}
    per_thread_label = {}

    ARGS = {
        "mode": args.ntel_mode,
        "thetaT_deg": args.thetaT_deg,
        "signed": args.signed,
        "log_base": args.log_base,
        "normalize": (not args.no_normalize),
    }

    for tid, g in df.groupby("thread_id"):
        series = R_series_for_thread(
            g,
            args.time_window,
            args.msg_window,
            mapping=args.mapping,
            min_n=args.min_n,
            ARGS=ARGS,
            early_frac=args.early_frac,
        )
        per_thread_series[tid] = series

        R_mean, R_last, R_slope = features_from_R_series(series)
        stances_all = g["stance"].tolist()
        # baselines
        pct_support = float(np.mean(np.array(stances_all) == 1)) if len(stances_all) > 0 else 0.0
        ent = entropy_stance(stances_all)

        y = int(g["outcome_bin"].iloc[0])
        per_thread_label[tid] = y
        rows.append({
            "thread_id": tid,
            "R_mean": R_mean,
            "R_last": R_last,
            "R_slope": R_slope,
            "pct_support": pct_support,
            "stance_entropy": ent,
            "y": y,
            "n_msgs": int(g.shape[0]),
            "n_windows": int(len(series)),
        })

    feat = pd.DataFrame(rows).dropna()

    # Sauvegarde des features complètes (avant filtrage band)
    if args.save_features:
        outp = Path(args.save_features)
        outp.parent.mkdir(parents=True, exist_ok=True)
        feat.to_csv(outp, index=False)
        print(f"Saved features to: {outp}")

    # Filtrage éventuel par bande de support pour l'ÉVALUATION/TRACE
    feat_eval = feat
    if args.support_band is not None:
        a, b = args.support_band
        feat_eval = feat[(feat["pct_support"] >= a) & (feat["pct_support"] <= b)].copy()
        if feat_eval.shape[0] < 10:
            print(f"WARNING: very few threads in support band [{a},{b}] — CV may be unstable.", file=sys.stderr)

        # Restreindre les séries/labels pour le tracé
        tids = set(feat_eval["thread_id"].tolist())
        per_thread_series = {k: v for k, v in per_thread_series.items() if k in tids}
        per_thread_label  = {k: v for k, v in per_thread_label.items()  if k in tids}

    # Matrices (sur l'échantillon évalué)
    X_R = feat_eval[["R_mean", "R_last", "R_slope"]].values
    X_B = feat_eval[["pct_support", "stance_entropy"]].values
    X_C = feat_eval[["R_mean", "R_last", "R_slope", "pct_support", "stance_entropy"]].values
    y = feat_eval["y"].values.astype(int)

    print(f"\nDataset: {feat_eval.shape[0]} threads | mean n_msgs/thread={feat_eval['n_msgs'].mean():.1f} | mean n_windows/thread={feat_eval['n_windows'].mean():.1f}")
    window_desc = f"time {args.time_window}" if args.time_window else f"count {args.msg_window}" if args.msg_window else "whole-thread"
    mode_label = args.ntel_mode
    extra = []
    if args.ntel_mode in {"cos2", "entropy"} and args.signed:
        extra.append("signed")
    if args.thetaT_deg is not None:
        extra.append(f"thetaT={args.thetaT_deg}°")
    if args.ntel_mode == "entropy":
        extra.append(f"logBase={args.log_base}")
        extra.append("normalized" if not args.no_normalize else "non-normalized")
    if args.early_frac is not None:
        extra.append(f"early={args.early_frac}")
    if args.support_band is not None:
        a, b = args.support_band
        extra.append(f"band=[{a},{b}]")
    extra_str = (" | " + " ; ".join(extra)) if extra else ""
    print(f"Mapping: {args.mapping} | Window: {window_desc} | N_tel mode: {mode_label}{extra_str}")

    # Évaluation CV
    if feat_eval.shape[0] >= 10 and len(np.unique(y)) > 1:
        r1 = eval_cv(X_R, y, "Features_R")
        r2 = eval_cv(X_B, y, "Baselines")
        r3 = eval_cv(X_C, y, "Combine")

        print("\nCV Results (mean over folds):")
        for r in [r1, r2, r3]:
            print(f"- {r.name:10s}  AUC={r.auc:.3f}  ACC={r.acc:.3f}")
    else:
        print("\n[INFO] Not enough data or only one class after filtering — skipping CV.")

    # Plot optionnel (sur l'échantillon évalué)
    if args.plot_mean_rt:
        ylabel = {"R": "R", "cos2": "cos2", "entropy": "N_tel(entropy)"}[args.ntel_mode]
        plot_mean_R_curves(per_thread_series, per_thread_label, args.plot_out, metric_name=ylabel)
        print(f"Mean {ylabel}(t) plot saved to: {args.plot_out}")


if __name__ == "__main__":
    main()
