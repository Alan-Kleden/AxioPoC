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


