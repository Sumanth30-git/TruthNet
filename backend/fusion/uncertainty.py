from backend.schemas import Uncertainty


def default_uncertainty() -> Uncertainty:
    return Uncertainty.HIGH
