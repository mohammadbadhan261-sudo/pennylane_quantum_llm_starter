"""Verify that predictions vary on clearly positive/negative sentences."""
from pathlib import Path

import torch

from hybrid_model import QuantumTextHead, encode_texts, load_language_model

ROOT = Path(__file__).resolve().parent
checkpoint = ROOT / "models" / "quantum_head.pt"
if not checkpoint.exists():
    raise SystemExit("No checkpoint found. Run 02_train_hybrid_model.py")
info = torch.load(checkpoint, map_location="cpu", weights_only=True)
print("Dataset used:", info.get("dataset_path"))
print("Training rows:", info.get("train_examples"))
print("Architecture:", info.get("architecture", "original"))
model = QuantumTextHead(info["embedding_dim"])
try:
    model.load_state_dict(info["state_dict"])
except RuntimeError as e:
    raise SystemExit("Checkpoint uses an older architecture. Retrain with the updated scripts.") from e
model.eval()
tokenizer, encoder = load_language_model(info["model_name"])
sentences = [
    "I absolutely loved this movie.",
    "This movie was fantastic and enjoyable.",
    "I am happy with the excellent service.",
    "I hated this movie. It was awful.",
    "The service was terrible and disappointing.",
    "This is the worst experience I have ever had.",
]
with torch.no_grad():
    X = encode_texts(sentences, tokenizer, encoder)
    probs = torch.softmax(model(X), dim=1)[:, 1].tolist()
print("\nPredictions:")
for sentence, p in zip(sentences, probs):
    print(f"  Positive {p:6.1%} | {'Positive' if p >= 0.5 else 'Negative':8} | {sentence}")
print(f"\nPositive-score spread: {min(probs):.1%}–{max(probs):.1%}")
if max(probs) - min(probs) < .05:
    print("Warning: scores hardly change across sentences. Check training data and evaluation.")
