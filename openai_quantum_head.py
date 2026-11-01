"""Quantum sentiment head for OpenAI embeddings. GPT model weights are not trained."""
import math
import pennylane as qml
import torch
from torch import nn

NUM_QUBITS = 4

class QuantumTextHead(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.input_norm = nn.LayerNorm(embedding_dim)
        self.projection = nn.Linear(embedding_dim, NUM_QUBITS)
        device = qml.device("default.qubit", wires=NUM_QUBITS)

        @qml.qnode(device, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(NUM_QUBITS), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(NUM_QUBITS))
            return qml.probs(wires=range(NUM_QUBITS))

        self.quantum = qml.qnn.TorchLayer(circuit, weight_shapes={"weights": (2, NUM_QUBITS, 3)})
        self.classifier = nn.Linear(2 ** NUM_QUBITS, 2)
        # A classical skip pathway is included for robustness. This makes it
        # essential to compare against a classical-only ablation in research.
        self.classical_skip = nn.Linear(embedding_dim, 2, bias=False)
        nn.init.zeros_(self.classical_skip.weight)

    def forward(self, x):
        features = self.input_norm(x)
        angles = torch.tanh(self.projection(features)) * (math.pi / 2)
        quantum = self.quantum(angles)
        if quantum.ndim != 2 or quantum.shape[1] != 16:
            raise RuntimeError(f"Expected quantum output (batch_size, 16), got {tuple(quantum.shape)}")
        return self.classifier(quantum) + self.classical_skip(features)
