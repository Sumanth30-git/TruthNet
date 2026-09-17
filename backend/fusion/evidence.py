from backend.schemas import EvidenceItem, SignalResult


def collect_evidence(signals: list[SignalResult]) -> list[EvidenceItem]:
    """Phase 0 intentionally does not derive evidence from model scores."""
    return []
