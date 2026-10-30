"""Frozen DistilGPT2 embeddings with a trainable hybrid PennyLane sentiment head.

This is a sentiment classifier using simulated quantum circuits, not an LLM
fine-tuning system. A trainable classical residual helps avoid near-constant
outputs; it also means good accuracy would not demonstrate quantum advantage.
"""
import math

import pennylane as qml
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

LANGUAGE_MODEL_ID = "distilbert/distilgpt2"
NUM_QUBITS = 4


def load_language_model(model_name=LANGUAGE_MODEL_ID):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoder = AutoModel.from_pretrained(model_name)
    encoder.eval()
    encoder.requires_grad_(False)
    return tokenizer, encoder


@torch.no_grad()
def encode_texts(texts, tokenizer, language_model, batch_size=8):
    """Masked average of contextual token features (up to 64 tokens)."""
    if not texts:
        raise ValueError("No input texts to encode")
    outputs = []
    for i in range(0, len(texts), batch_size):
        tokens = tokenizer(texts[i:i+batch_size], padding=True,
                           truncation=True, max_length=64,
                           return_tensors="pt")
        hidden = language_model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        outputs.append(pooled.float())
    return torch.cat(outputs, dim=0)


class QuantumTextHead(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.input_norm = nn.LayerNorm(embedding_dim)
        self.projection = nn.Linear(embedding_dim, NUM_QUBITS)

        dev = qml.device("default.qubit", wires=NUM_QUBITS)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(NUM_QUBITS), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(NUM_QUBITS))
            return qml.probs(wires=range(NUM_QUBITS))

        self.quantum = qml.qnn.TorchLayer(
            circuit, weight_shapes={"weights": (2, NUM_QUBITS, 3)}
        )
        self.classifier = nn.Linear(2 ** NUM_QUBITS, 2)
        # Provides a direct trainable path from text embeddings to the logits.
        # Its weights start at 0, so the model initially relies on the quantum head.
        self.classical_skip = nn.Linear(embedding_dim, 2, bias=False)
        nn.init.zeros_(self.classical_skip.weight)

    def forward(self, text_embeddings):
        features = self.input_norm(text_embeddings)
        angles = torch.tanh(self.projection(features)) * (math.pi / 2)
        quantum_features = self.quantum(angles)
        if quantum_features.ndim != 2 or quantum_features.shape[1] != 2 ** NUM_QUBITS:
            raise RuntimeError(f"Unexpected quantum output shape: {tuple(quantum_features.shape)}")
        return self.classifier(quantum_features) + self.classical_skip(features)
