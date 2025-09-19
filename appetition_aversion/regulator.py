def appetition_aversion_score(fc: float, fi: float) -> float:
    """
    Calcule un score très simple :
    - fc = force conative (appétition)
    - fi = force inhibitrice (aversion)
    Le score est simplement la différence.
    """
    return fc - fi
