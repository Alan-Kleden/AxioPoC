# tests/test_negentropy_telotopic_deg_vec.py
import math
import numpy as np
import pytest

from negentropy_telotopic.ntel import (
    ntel_from_radians,
    ntel_from_degrees,
    ntel_vectorized,
    classify_coherence,
    CoherenceThresholds,
)

def test_degrees_equals_radians_simple():
    deg = [0, 0, 90, 180]     # en degrés
    rad = [math.radians(d) for d in deg]
    assert abs(ntel_from_degrees(deg) - ntel_from_radians(rad)) < 1e-12

def test_opposed_split_zero():
    # moitié 0, moitié pi -> R ~ 0
    ang = np.array([0.0, 0.0, math.pi, math.pi])
    assert ntel_from_radians(ang) == pytest.approx(0.0, abs=1e-12)

def test_vectorized_matches_scalar():
    rng = np.random.default_rng(42)
    ang = rng.uniform(-math.pi, math.pi, size=(5, 100))  # 5 séries de 100 angles
    r_vec = ntel_vectorized(ang, axis=1)
    r_sca = np.array([ntel_from_radians(row) for row in ang])
    np.testing.assert_allclose(r_vec, r_sca, rtol=1e-12, atol=1e-12)

def test_weights_vectorized():
    ang = np.array([[0.0, 0.0, math.pi]])
    w   = np.array([[1.0, 2.0, 3.0]])
    # calcule et vérifie que ça reste dans [0,1] (sanity)
    r = ntel_vectorized(ang, weights=w, axis=1)
    assert r.shape == (1,)
    assert 0.0 <= float(r[0]) <= 1.0

def test_classify_defaults():
    th = CoherenceThresholds()  # 0.70 / 0.40
    assert classify_coherence(0.85, th) == "élevée"
    assert classify_coherence(0.55, th) == "moyenne"
    assert classify_coherence(0.10, th) == "faible"

def test_custom_thresholds():
    th = CoherenceThresholds(high=0.8, medium=0.6)
    assert classify_coherence(0.79, th) == "moyenne"
    assert classify_coherence(0.81, th) == "élevée"
