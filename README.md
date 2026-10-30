# PennyLane + GPT-2 Hybrid Model: Beginner Starter

This is a learning example, not a production-ready LLM and not a benchmark demonstrating quantum advantage.

## Setup (Windows 10/11 PowerShell in VS Code)

1. Install Python 3.12 (64-bit) and the Microsoft VS Code Python extension.
2. Open this folder via **File > Open Folder** in VS Code.
3. Select **Terminal > New Terminal** and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prohibits activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Alternatively, without activation, run `.\.venv\Scripts\python.exe` instead of `python` in subsequent commands.

Use **Ctrl+Shift+P > Python: Select Interpreter** and choose `.venv\Scripts\python.exe`.

## Run the three stages

```powershell
python 01_first_quantum_circuit.py
python 02_train_hybrid_model.py
python -m streamlit run 03_app.py
```

The first training run downloads `distilbert/distilgpt2` from Hugging Face (several hundred MB). An internet connection is required for initial download. This is a relatively small pretrained generative language model; its weights remain frozen.

The training script trains only a classical projection, a quantum circuit, and a classifier. The quantum circuit is simulated by PennyLane's `default.qubit` on your CPU. No real quantum hardware is used.

## Structure

- `01_first_quantum_circuit.py`: Hadamard + CNOT Bell state demonstration.
- `hybrid_model.py`: frozen GPT-2 embedding extraction, quantum PyTorch head.
- `02_train_hybrid_model.py`: toy dataset, training, evaluation, model checkpoint.
- `03_app.py`: Streamlit local sentiment prediction app.
- `models/`: saved weights from training.

## Limitations

- 30 hand-written sample sentences, only 6 test examples. Results are highly unstable and not scientifically meaningful.
- The language model is not fine-tuned. No new text-generation capabilities or speedups are being claimed.
- The hybrid classifier may be worse than a normal classical linear classifier.
- To research actual quantum benefits, compare against classical baselines using larger datasets and independent train/validation/test splits.
