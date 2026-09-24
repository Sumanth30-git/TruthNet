"""Manual smoke test for TruthNet's optional NLI evidence adapter."""

import os

from backend.config import news_nli_model_id
from backend.models.nli_model import NLIModelAdapter


EXAMPLES = [
    ("New Delhi is the capital of India.", "India's capital is New Delhi."),
    ("New Delhi is the capital of India.", "India's capital is Mumbai."),
    ("India has a capital city.", "The country has several major cities."),
]


def main() -> None:
    if os.getenv("MODEL_INFERENCE_ENABLED", "false").strip().lower() not in {"1", "true", "yes", "on"}:
        raise SystemExit("Set MODEL_INFERENCE_ENABLED=true before running this manual NLI smoke test.")

    adapter = NLIModelAdapter()
    print("Model:", news_nli_model_id())
    print("Label mapping: 0 = contradiction, 1 = entailment, 2 = neutral")
    for premise, hypothesis in EXAMPLES:
        result = adapter.predict(premise, hypothesis)
        print("\n---")
        print("Premise (source evidence):", premise)
        print("Hypothesis (claim):", hypothesis)
        print("Device:", adapter.device)
        print("Prediction:", result.prediction)
        print("Status:", result.status.value)
        if result.probabilities:
            print("Probabilities:")
            print(f"  contradiction: {result.probabilities.contradiction:.4f}")
            print(f"  entailment: {result.probabilities.entailment:.4f}")
            print(f"  neutral: {result.probabilities.neutral:.4f}")
            print(f"Selected confidence: {result.selected_confidence:.4f}")
        timing = f"{result.inference_time_ms:.2f} ms" if result.inference_time_ms is not None else "unavailable"
        print("Inference time:", timing)


if __name__ == "__main__":
    main()
