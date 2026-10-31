"""Quickly test the quantum layer's batch shape and gradient flow (no GPT-2 download)."""
import torch
from hybrid_model import QuantumTextHead

model = QuantumTextHead(embedding_dim=768)
features = torch.randn(6, 768)
logits = model(features)
print("Classifier output shape:", tuple(logits.shape))
assert logits.shape == (6, 2), f"Expected (6, 2), got {tuple(logits.shape)}"
loss = logits.sum()
loss.backward()
assert model.projection.weight.grad is not None, "Missing projection gradient"
assert model.quantum.weights.grad is not None, "Missing quantum gradient"
print("PASS: output shape and gradients are correct.")
