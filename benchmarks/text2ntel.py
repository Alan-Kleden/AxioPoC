# -*- coding: utf-8 -*-
"""
text2ntel.py — Benchmark B (Texte -> Affect -> θ -> Cohérence télotopique)

Entrée  : CSV avec colonnes: thread_id, text, timestamp, outcome
Sorties : outdir/
  - features.csv              (par thread: R_mean, R_last, R_slope)
  - rt_series.csv             (par fenêtre: thread_id, window_idx, R)
  - mean_rt_by_outcome.png    (boxplot de R_mean par outcome si demandé)
  - eval.txt                  (corrélation jouet R_mean ~ outcome si demandé)
  - run.log                   (trace courte)

Exemples :
  python benchmarks/text2ntel.py ^
    --input data\\convokit\\cmv\\cmv.csv ^
    --emoji-mode interpret ^
    --msg-window 20 ^
    --save-features --plot-mean-rt ^
    --outdir artifacts_benchB --evaluate
"""
from __future__ import annotations

import argparse
import os
import sys
import math
import time
import warnings
from typing import Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sklearn uniquement pour la pente (slope) via régression linéaire simple
from sklearn.linear_model import LinearRegression

# --- VADER (optionnel) -------------------------------------------------------
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER_OK = True
except Exception:
    SentimentIntensityAnalyzer = None
    _VADER_OK = False

# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Texte -> Affect -> θ -> Cohérence télotopique")
    p.add_argument("--input", required=True, help="CSV d'entrée: thread_id,text,timestamp,outcome")
    grp_w = p.add_mutually_exclusive_group(required=True)
    grp_w.add_argument("--msg-window", type=int, help="Taille de fenêtre en N messages (recommandé)")
    grp_w.add_argument("--time-window", type=int, help="Taille de fenêtre en minutes (requiert des timestamps)")

    p.add_argument("--emoji-mode", choices=["raw", "interpret"], default="interpret",
                   help="Traitement des émojis: raw (inchangé) ou interpret (remappage simple)")
    p.add_argument("--sentiment", choices=["vader", "toy"], default="vader",
                   help="Méthode de valence: vader (si dispo) sinon fallback -> toy")
    p.add_argument("--outdir", default="artifacts_benchB", help="Dossier de sortie")
    p.add_argument("--save-features", action="store_true", help="Écrit features.csv et rt_series.csv")
    p.add_argument("--plot-mean-rt", action="store_true", help="Trace mean_rt_by_outcome.png")
    p.add_argument("--evaluate", action="store_true", help="Écrit eval.txt (corrélation jouet)")
    return p.parse_args()


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# --- Mini interprétation d'émojis (sans dépendance externe) ------------------
_EMOJI_MAP = {
    "😀": " happy ", "😃": " happy ", "😄": " happy ", "😁": " happy ",
    "😂": " laugh ", "🤣": " laugh ",
    "🙂": " positive ", "😊": " positive ",
    "😍": " love ", "❤️": " love ", "❤": " love ",
    "😢": " sad ", "😭": " cry ",
    "😡": " angry ", "🤬": " angry ",
    "😱": " fear ", "😨": " fear ",
    "👍": " thumbs up ", "👎": " thumbs down ",
    "😐": " neutral ", "🤔": " thinking ",
}

def interpret_emojis(text: str) -> str:
    if not text:
        return ""
    out = text
    for k, v in _EMOJI_MAP.items():
        out = out.replace(k, v)
    return out


# --- Sentiment -> θ ----------------------------------------------------------
class ValenceModel:
    def __init__(self, mode: str):
        self.mode = mode
        self.analyzer = None
        if mode == "vader" and _VADER_OK:
            self.analyzer = SentimentIntensityAnalyzer()
        elif mode == "vader" and not _VADER_OK:
            print("[WARN] VADER indisponible — repli sur 'toy'", file=sys.stderr)
            self.mode = "toy"

    def valence(self, text: str) -> float:
        """
        Retourne une valence v dans [-1, 1].
        - VADER: score 'compound'
        - toy  : 0.0 (aucun signal)
        """
        t = text or ""
        if self.mode == "vader" and self.analyzer is not None:
            try:
                return float(self.analyzer.polarity_scores(t)["compound"])
            except Exception:
                return 0.0
        return 0.0  # baseline neutre (evite pseudo-signal)

def valence_to_theta(v: float) -> float:
    """
    Map v in [-1,1] -> θ in [0, π], v=+1 -> θ=0 ; v=-1 -> θ=π
    """
    return math.pi * (1.0 - v) / 2.0

def resultant_length_thetas(thetas: np.ndarray) -> float:
    """
    Mean Resultant Length sur des angles θ (radians).
    """
    if len(thetas) == 0:
        return np.nan
    c = np.cos(thetas).mean()
    s = np.sin(thetas).mean()
    R = math.sqrt(c*c + s*s)
    return float(R)


# --- Fenêtrage ---------------------------------------------------------------
def windows_by_msg(df_thread: pd.DataFrame, msg_window: int) -> Iterable[Tuple[int, pd.DataFrame]]:
    """
    Coupe en blocs consécutifs de `msg_window` messages (ordre d'apparition).
    """
    n = len(df_thread)
    start = 0
    idx = 0
    while start < n:
        end = min(start + msg_window, n)
        yield idx, df_thread.iloc[start:end]
        start = end
        idx += 1

def windows_by_time(df_thread: pd.DataFrame, minutes: int) -> Iterable[Tuple[int, pd.DataFrame]]:
    """
    Coupe en fenêtres temporelles de `minutes`.
    Requiert une colonne 'timestamp' numérique (secondes). Trie avant.
    """
    df = df_thread.copy()
    if "timestamp" not in df.columns or df["timestamp"].isna().all():
        # Pas de timestamps => aucune fenêtre
        return
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        return
    df = df.sort_values("timestamp")
    start_ts = None
    bucket = []
    idx = 0
    for _, row in df.iterrows():
        ts = float(row["timestamp"])
        if start_ts is None:
            start_ts = ts
        # taille en secondes
        if ts - start_ts < minutes * 60:
            bucket.append(row)
        else:
            if bucket:
                yield idx, pd.DataFrame(bucket)
                idx += 1
            start_ts = ts
            bucket = [row]
    if bucket:
        yield idx, pd.DataFrame(bucket)


# --- Features par thread -----------------------------------------------------
def features_from_rt(rt: List[float]) -> Tuple[float, float, float]:
    """
    R_mean, R_last, R_slope (régression R ~ window_idx)
    """
    if len(rt) == 0:
        return (np.nan, np.nan, np.nan)
    R_mean = float(np.nanmean(rt))
    R_last = float(rt[-1])

    # slope via régression linéaire
    X = np.arange(len(rt)).reshape(-1, 1)
    y = np.array(rt, dtype=float)
    if np.all(np.isnan(y)):
        slope = np.nan
    else:
        # remplace NaN par moyenne (très rare si fenêtres vides)
        if np.isnan(y).any():
            m = np.nanmean(y)
            y = np.where(np.isnan(y), m, y)
        model = LinearRegression()
        try:
            model.fit(X, y)
            slope = float(model.coef_[0])
        except Exception:
            slope = np.nan

    return (R_mean, R_last, slope)


# --- Pipeline principal ------------------------------------------------------
def main():
    t0 = time.time()
    args = parse_args()
    ensure_outdir(args.outdir)

    # Log
    log_path = os.path.join(args.outdir, "run.log")
    def log(msg: str):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg.rstrip() + "\n")

    # Lecture CSV
    usecols = ["thread_id", "text", "timestamp", "outcome"]
    df = pd.read_csv(args.input, usecols=usecols, encoding="utf-8")
    # Nettoyages de base
    if "outcome" in df.columns:
        # autorise outcome str/bool/num -> int {0,1}, NaN si autre
        df["outcome"] = df["outcome"].map(lambda x: 1 if str(x).strip() == "1" else (0 if str(x).strip() == "0" else np.nan))
    else:
        df["outcome"] = np.nan

    # timestamps -> numériques (optionnels)
    if "timestamp" in df.columns:
        def _to_num(x):
            try:
                if pd.isna(x): return np.nan
                return float(x)
            except Exception:
                return np.nan
        df["timestamp"] = df["timestamp"].map(_to_num)
    else:
        df["timestamp"] = np.nan

    # Traitement texte + émojis
    if args.emoji_mode == "interpret":
        df["text_proc"] = df["text"].astype(str).map(interpret_emojis)
    else:
        df["text_proc"] = df["text"].astype(str)

    # Sentiment
    vm = ValenceModel(args.sentiment)
    df["valence"] = df["text_proc"].map(vm.valence)
    df["theta"] = df["valence"].map(valence_to_theta)

    # Groupes par thread
    threads = df["thread_id"].unique().tolist()
    threads_outcomes = df.drop_duplicates("thread_id")[["thread_id", "outcome"]].set_index("thread_id")["outcome"].to_dict()

    rt_rows = []     # pour rt_series.csv
    feat_rows = []   # pour features.csv
    n_threads = 0
    skipped_time = 0

    for tid, g in df.groupby("thread_id", sort=False):
        g = g.reset_index(drop=True)

        # Choix du fenêtrage
        if args.msg_window:
            win_iter = windows_by_msg(g, args.msg_window)
        else:
            # time-window
            _wins = list(windows_by_time(g, args.time_window))
            if not _wins:
                skipped_time += 1
                continue
            win_iter = _wins

        rt = []
        for w_idx, g_w in win_iter:
            thetas = g_w["theta"].to_numpy(dtype=float)
            thetas = thetas[~np.isnan(thetas)]
            R = resultant_length_thetas(thetas)
            rt_rows.append({"thread_id": tid, "window_idx": int(w_idx), "R": R})
            rt.append(R)

        R_mean, R_last, R_slope = features_from_rt(rt)
        feat_rows.append({
            "thread_id": tid,
            "R_mean": R_mean,
            "R_last": R_last,
            "R_slope": R_slope
        })

        n_threads += 1
        if n_threads % 200 == 0:
            log(f"[{n_threads}] threads traités...")

    # Écritures
    if args.save_features:
        pd.DataFrame(rt_rows).to_csv(os.path.join(args.outdir, "rt_series.csv"), index=False, encoding="utf-8")
        pd.DataFrame(feat_rows).to_csv(os.path.join(args.outdir, "features.csv"), index=False, encoding="utf-8")

    # Plot
    if args.plot_mean_rt:
        feat_df = pd.DataFrame(feat_rows)
        # attache outcome par thread (si dispo)
        feat_df["outcome"] = feat_df["thread_id"].map(threads_outcomes)
        fig = plt.figure(figsize=(6, 4))
        groups = []
        labels = []
        for lab in [0, 1]:
            vals = feat_df.loc[feat_df["outcome"] == lab, "R_mean"].dropna().values
            if len(vals) > 0:
                groups.append(vals)
                labels.append(str(lab))
        if groups:
            plt.boxplot(groups, labels=labels, showfliers=False)
            plt.ylabel("R_mean")
            plt.xlabel("outcome")
            plt.title("R_mean par outcome")
            plt.tight_layout()
            fig_path = os.path.join(args.outdir, "mean_rt_by_outcome.png")
            plt.savefig(fig_path, dpi=150)
        plt.close(fig)

    # Évaluation jouet (corrélation)
    if args.evaluate:
        feat_df = pd.DataFrame(feat_rows)
        feat_df["outcome"] = feat_df["thread_id"].map(threads_outcomes)
        # corrélation de Pearson entre R_mean et outcome
        corr = np.nan
        try:
            sub = feat_df.dropna(subset=["R_mean", "outcome"])
            if len(sub) >= 3 and sub["outcome"].nunique() > 1:
                corr = float(np.corrcoef(sub["R_mean"].values, sub["outcome"].values)[0, 1])
        except Exception:
            pass
        with open(os.path.join(args.outdir, "eval.txt"), "w", encoding="utf-8") as f:
            f.write(f"Corr(R_mean, outcome) = {corr if not np.isnan(corr) else 'nan'}\n")
            f.write(f"Threads = {len(feat_rows)}\n")

    elapsed = time.time() - t0
    if args.time_window and skipped_time:
        log(f"[INFO] Threads ignorés faute de timestamps exploitables (time-window): {skipped_time}")
    log(f"OK -> {args.outdir} (threads={n_threads}) en {elapsed:.1f}s")


if __name__ == "__main__":
    # Évite les warnings verbeux inutiles
    warnings.filterwarnings("ignore")
    main()
