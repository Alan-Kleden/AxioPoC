def clamp(value: float, min_value: float, max_value: float) -> float:
    """Force une valeur à rester entre min_value et max_value."""
    return max(min_value, min(value, max_value))
