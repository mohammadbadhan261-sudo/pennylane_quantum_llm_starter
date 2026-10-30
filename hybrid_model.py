"""Frozen GPT-2 feature extractor plus a trainable PennyLane quantum head."""
import math

import pennylane as qml
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

LANGUAGE_MODEL_ID = "distilbert/distilgpt2"
NUM_QUBITS = 4


def load_language_model(model_name=LANGUAGE_MODEL_ID):
    """Download (first time) and load a pretrained model on CPU."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    language_model = AutoModel.from_pretrained(model_name)
    language_model.eval()
    language_model.requires_grad_(False)  # We train only the small hybrid head.
    return tokenizer, language_model


@torch.no_grad()
def encode_texts(texts, tokenizer, language_model, batch_size=8):
    """Turn input sentences into fixed-size, masked-mean GPT-2 vectors."""
    all_embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        outputs = language_model(**tokens)
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        mean_embeddings = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        all_embeddings.append(mean_embeddings.float())
    return torch.cat(all_embeddings, dim=0)


class QuantumTextHead(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, NUM_QUBITS)
        device = qml.device("default.qubit", wires=NUM_QUBITS)

        @qml.qnode(device, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(NUM_QUBITS), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(NUM_QUBITS))
            return qml.probs(wires=range(NUM_QUBITS))

        self.quantum = qml.qnn.TorchLayer(
            circuit, weight_shapes={"weights": (2, NUM_QUBITS, 3)}
        )
        self.classifier = nn.Linear(2**NUM_QUBITS, 2)

    def forward(self, text_embeddings):
        angles = torch.tanh(self.projection(text_embeddings)) * (math.pi / 2)
        quantum_features = self.quantum(angles)
        return self.classifier(quantum_features)
