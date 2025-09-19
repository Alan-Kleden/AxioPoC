__all__ = ["ntel"]
from .ntel import (
    ntel_from_radians,
    ntel_from_degrees,
    ntel_vectorized,
    classify_coherence,
    CoherenceThresholds,
)

__all__ = [
    "ntel_from_radians",
    "ntel_from_degrees",
    "ntel_vectorized",
    "classify_coherence",
    "CoherenceThresholds",
]
from negentropy_telotopic import ntel_from_degrees, classify_coherence
