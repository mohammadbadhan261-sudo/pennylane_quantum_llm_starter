"""Your first simulated quantum circuit in PennyLane."""
import pennylane as qml

# default.qubit runs on your classical computer, not on real quantum hardware.
device = qml.device("default.qubit", wires=2)

@qml.qnode(device)
def bell_circuit():
    qml.Hadamard(wires=0)   # Put qubit 0 in a superposition.
    qml.CNOT(wires=[0, 1])  # Entangle qubit 0 with qubit 1.
    return qml.probs(wires=[0, 1])

print("PennyLane version:", qml.__version__)
print("Quantum circuit:\n")
print(qml.draw(bell_circuit)())
print("\nProbabilities for [00, 01, 10, 11]:")
print(bell_circuit())
print("Expected approximately: [0.5, 0.0, 0.0, 0.5]")
