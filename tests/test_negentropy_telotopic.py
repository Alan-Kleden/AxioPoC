import math
from negentropy_telotopic.ntel import ntel_from_radians

def test_perfect_alignment():
    # tous alignés (0 rad)
    assert ntel_from_radians([0.0, 0.0, 0.0]) == 1.0

def test_opposed_split():
    # moitié 0, moitié π : dispersion max ≈ 0 (selon la métrique choisie)
    v = ntel_from_radians([0.0, 0.0, math.pi, math.pi])
    assert 0.0 <= v <= 0.1

def test_small_spread_high_ntel():
    # angles proches (faible dispersion) → N_tel élevé
    eps = math.radians(10)
    v = ntel_from_radians([0.0, eps, -eps, eps/2])
    assert v > 0.8
