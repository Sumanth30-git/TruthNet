import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

examples = [
    ("India's capital is New Delhi.", "New Delhi is the capital of India."),
    ("India's capital is Mumbai.", "New Delhi is the capital of India."),
    ("India has a capital city.", "The country has several major cities."),
]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Model:", MODEL_NAME)
    print("Device:", device)
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    print(f"Model load time: {time.perf_counter() - start:.2f}s")
    print("Labels:", model.config.id2label)

    for premise, hypothesis in examples:
        inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
        inference_time = time.perf_counter() - start
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]
        predicted_id = int(torch.argmax(probabilities).item())
        predicted_label = model.config.id2label[predicted_id]
        print("\n---")
        print("Premise:", premise)
        print("Hypothesis:", hypothesis)
        print("Prediction:", predicted_label)
        print("Scores:")
        for index, probability in enumerate(probabilities):
            print(f"  {model.config.id2label[index]}: {probability.item():.4f}")
        print(f"Inference time: {inference_time * 1000:.2f} ms")


if __name__ == "__main__":
    main()
