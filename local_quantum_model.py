"""Four-qubit simulated quantum classifier; no Transformers or OpenAI SDK needed."""
import math
import pennylane as qml
import torch
from torch import nn

NUM_QUBITS = 4


class LocalQuantumHead(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.normalization = nn.LayerNorm(embedding_dim)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = torch.tanh(self.projection(self.normalization(x))) * (math.pi / 2)
        probabilities = self.quantum(angles)
        if probabilities.ndim != 2 or probabilities.shape[1] != 2 ** NUM_QUBITS:
            raise RuntimeError(f"Expected [batch,16] quantum probabilities, got {tuple(probabilities.shape)}")
        return self.classifier(probabilities)
