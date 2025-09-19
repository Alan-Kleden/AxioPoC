import math
from dataclasses import dataclass

@dataclass(frozen=True)
class DecayParams:
    """Décroissance exponentielle avec asymétrie valence-positive/valence-négative."""
    lambda_pos: float = 0.01  # >0, mémotions positives
    lambda_neg: float = 0.02  # >0, mémotions négatives (ex: décroissent plus lentement/rapidement)

def memotion_decay(v0: float, t: float, p: DecayParams = DecayParams()) -> float:
    """
    v0: valence*intensité initiale (VI), t: temps (en pas discrétisés).
    Règle simple : si v0>=0 → lambda_pos, sinon → lambda_neg (asymétrie).
    """
    lam = p.lambda_pos if v0 >= 0 else p.lambda_neg
    return v0 * math.exp(-lam * max(0.0, t))
