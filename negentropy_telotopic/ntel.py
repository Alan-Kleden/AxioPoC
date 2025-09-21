from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union
import math

# NumPy est optionnel mais recommandé
try:
    import numpy as np
except ImportError:  # fallback minimal
    np = None

Number = Union[int, float]
def _validate_angles(seq: Iterable[Number]) -> None:
    it = list(seq)
    if len(it) == 0:
        raise ValueError("angles must be a non-empty iterable")
    # pas de validation stricte des NaN ici; on peut l'ajouter si besoin


def ntel_from_radians(angles_rad: Iterable[Number], weights: Optional[Iterable[Number]] = None) -> float:
    r"""
    Telotopic negentropy (mean resultant length R \in [0,1]) from angles in radians.
    R = || (Σ w_k e^{i θ_k}) || / (Σ w_k).  Si weights=None, w_k = 1/n implicitement.
    """
    # Fallback pur Python si NumPy indisponible
    if np is None:
        angs = list(map(float, angles_rad))
        if not angs:
            return 0.0
        if weights is None:
            cx = sum(math.cos(a) for a in angs) / len(angs)
            sx = sum(math.sin(a) for a in angs) / len(angs)
        else:
            ws = list(map(float, weights))
            if len(ws) != len(angs):
                raise ValueError("weights must have same length as angles")
            wsum = sum(ws)
            if wsum <= 0 or not math.isfinite(wsum):
                raise ValueError("sum of weights must be > 0 and finite")
            cx = sum(w * math.cos(a) for a, w in zip(angs, ws)) / wsum
            sx = sum(w * math.sin(a) for a, w in zip(angs, ws)) / wsum
        R = math.hypot(cx, sx)
        return float(max(0.0, min(1.0, R)))

    # Chemin NumPy (recommandé)
    angs = np.asarray(list(angles_rad), dtype=np.float64)
    if angs.size == 0:
        return 0.0
    if weights is None:
        z = np.exp(1j * angs).mean()
    else:
        w = np.asarray(list(weights), dtype=np.float64)
        if w.shape != angs.shape:
            raise ValueError("weights must have same shape/length as angles")
        wsum = w.sum()
        if not np.isfinite(wsum) or wsum <= 0:
            raise ValueError("sum of weights must be > 0 and finite")
        z = np.sum(w * np.exp(1j * angs)) / wsum
    R = float(np.abs(z))
    return float(np.clip(R, 0.0, 1.0))



def ntel_from_degrees(angles_deg: Iterable[Number], weights: Optional[Iterable[Number]] = None) -> float:
    """
    Wrapper degrés -> radians.
    """
    if np is None:
        # conversion manuelle si NumPy absent
        angs_rad = [math.radians(float(d)) for d in angles_deg]
    else:
        angs_rad = np.deg2rad(np.asarray(list(angles_deg), dtype=np.float64))
    return ntel_from_radians(angs_rad, weights=weights)


def ntel_vectorized(
    angles_rad: "np.ndarray",
    weights: Optional["np.ndarray"] = None,
    axis: int = -1
) -> "np.ndarray":
    """
    Version NumPy vectorisée : retourne R le long de `axis`.
    shapes compatibles: angles (..., N), weights idem ou None.
    """
    if np is None:
        raise RuntimeError("NumPy is required for ntel_vectorized")
    angs = np.asarray(angles_rad, dtype=np.float64)
    if weights is None:
        z = np.exp(1j * angs).mean(axis=axis)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != angs.shape:
            raise ValueError("weights must have same shape as angles")
        wsum = np.sum(w, axis=axis, keepdims=True)
        if not np.all(np.isfinite(wsum)) or np.any(wsum <= 0):
            raise ValueError("sum of weights along axis must be > 0")
        z = np.sum(w * np.exp(1j * angs), axis=axis) / np.squeeze(wsum, axis=axis)
    R = np.abs(z)
    return np.clip(R, 0.0, 1.0)

@dataclass(frozen=True)
class CoherenceThresholds:
    high: float = 0.70
    medium: float = 0.40

    def validate(self) -> None:
        if not (0.0 <= self.medium <= self.high <= 1.0):
            raise ValueError("Expected 0.0 <= medium <= high <= 1.0")


def classify_coherence(ntel: float, thresholds: CoherenceThresholds = CoherenceThresholds()) -> str:
    """
    Classe la cohérence télotopique à partir de R:
      - >= high     -> "élevée"
      - >= medium   -> "moyenne"
      - sinon       -> "faible"
    """
    thresholds.validate()
    x = float(ntel)
    if x >= thresholds.high:
        return "élevée"
    if x >= thresholds.medium:
        return "moyenne"
    return "faible"

def circular_mean(angles: Iterable[float]) -> float:
    xs = [math.cos(a) for a in angles]
    ys = [math.sin(a) for a in angles]
    return math.atan2(sum(ys), sum(xs))

# --- Telos-based variants: cos² and entropy ---

def telos_angle_from_resultant(angles_rad: Iterable[Number],
                               weights: Optional[Iterable[Number]] = None) -> float:
    """
    θ_T := angle du résultant pondéré (atan2(S, C)).
    Sert de télos "endogène" si l'utilisateur n'en fournit pas.
    """
    if np is None:
        angs = [float(a) for a in angles_rad]
        if not angs:
            return 0.0
        if weights is None:
            ws = [1.0] * len(angs)
        else:
            ws = [float(w) for w in weights]
        c = sum(w * math.cos(a) for a, w in zip(angs, ws)) / sum(ws)
        s = sum(w * math.sin(a) for a, w in zip(angs, ws)) / sum(ws)
        return math.atan2(s, c)
    else:
        angs = np.asarray(list(angles_rad), dtype=np.float64)
        if angs.size == 0:
            return 0.0
        ws = np.asarray(list(weights), dtype=np.float64) if weights is not None else np.ones_like(angs)
        C = float(np.sum(ws * np.cos(angs)) / np.sum(ws))
        S = float(np.sum(ws * np.sin(angs)) / np.sum(ws))
        return math.atan2(S, C)


def ntel_cos2_from_radians(angles_rad: Iterable[Number],
                           weights: Optional[Iterable[Number]] = None,
                           theta_T: Optional[Number] = None,
                           signed: bool = False) -> float:
    """
    Cohérence autour du télos via cos².
    - signed=False : cos²(Δ) (antipodal compte comme "aligné")
    - signed=True  : max(cos(Δ), 0)^2 (antipodal ignoré)
    Retourne un score ∈ [0,1].
    """
    if np is None:
        angs = [float(a) for a in angles_rad]
        if not angs:
            return 0.0
        ws = [float(w) for w in weights] if weights is not None else [1.0] * len(angs)
        if theta_T is None:
            theta_T = telos_angle_from_resultant(angs, ws)
        num = 0.0; den = 0.0
        for a, w in zip(angs, ws):
            d = a - theta_T
            c = math.cos(d)
            if signed:
                c = max(c, 0.0)
            num += w * (c * c)
            den += w
        return 0.0 if den <= 0 else num / den
    else:
        angs = np.asarray(list(angles_rad), dtype=np.float64)
        if angs.size == 0:
            return 0.0
        ws = np.asarray(list(weights), dtype=np.float64) if weights is not None else np.ones_like(angs)
        th = float(theta_T) if theta_T is not None else telos_angle_from_resultant(angs, ws)
        d = angs - th
        c = np.cos(d)
        if signed:
            c = np.clip(c, 0.0, None)
        num = np.sum(ws * (c ** 2))
        den = np.sum(ws)
        return float(0.0 if den <= 0 else num / den)


def ntel_entropy_from_radians(angles_rad: Iterable[Number],
                              weights: Optional[Iterable[Number]] = None,
                              theta_T: Optional[Number] = None,
                              signed: bool = False,
                              log_base: float = 2.0,
                              normalize: bool = True) -> float:
    """
    N_tel = 1 - H ; H = -Σ p_i log(p_i), avec p_i ∝ w_i * f(Δ_i)
    f(Δ) = cos²(Δ) (par défaut); en version signée: f(Δ) = max(cos(Δ),0)^2.
    Normalisation par log(n_actif) pour borner N_tel ∈ [0,1].
    """
    if np is None:
        angs = [float(a) for a in angles_rad]
        if not angs:
            return 0.0
        ws = [float(w) for w in weights] if weights is not None else [1.0] * len(angs)
        if theta_T is None:
            theta_T = telos_angle_from_resultant(angs, ws)
        num = []
        for a, w in zip(angs, ws):
            d = a - theta_T
            c = math.cos(d)
            if signed:
                c = max(c, 0.0)
            num.append(max(w * (c * c), 0.0))
        S = sum(num)
        if S <= 0:
            return 0.0
        p = [x / S for x in num if x > 0]
        # Entropie
        H = -sum(q * (math.log(q) / math.log(log_base)) for q in p)
        if normalize:
            n = len(p)
            if n <= 1:
                return 1.0
            Hmax = math.log(n) / math.log(log_base)
            return max(0.0, 1.0 - H / Hmax)
        else:
            # Non normalisée : on peut la rabattre dans [0,1] par convention
            return max(0.0, 1.0 - H)
    else:
        angs = np.asarray(list(angles_rad), dtype=np.float64)
        if angs.size == 0:
            return 0.0
        ws = np.asarray(list(weights), dtype=np.float64) if weights is not None else np.ones_like(angs)
        th = float(theta_T) if theta_T is not None else telos_angle_from_resultant(angs, ws)
        d = angs - th
        c = np.cos(d)
        if signed:
            c = np.clip(c, 0.0, None)
        num = ws * (c ** 2)
        mask = num > 0
        num = num[mask]
        S = float(np.sum(num))
        if S <= 0:
            return 0.0
        p = num / S
        # Entropie
        H = float(-np.sum(p * (np.log(p) / np.log(log_base))))
        if normalize:
            n = int(p.size)
            if n <= 1:
                return 1.0
            Hmax = math.log(n) / math.log(log_base)
            return float(max(0.0, 1.0 - H / Hmax))
        else:
            return float(max(0.0, 1.0 - H))

