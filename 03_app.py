"""Launch: python -m streamlit run 03_app.py"""
from pathlib import Path

import streamlit as st
import torch

from hybrid_model import QuantumTextHead, encode_texts, load_language_model

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "models" / "quantum_head.pt"

st.set_page_config(page_title="Hybrid PennyLane Sentiment", page_icon="⚛️")
st.title("⚛️ Quantum + GPT-2 Sentiment")
st.caption("Frozen DistilGPT2 embeddings + PennyLane simulated quantum circuit + trainable classical skip path")

if not CHECKPOINT.exists():
    st.error("No trained model found. Run python 02_train_hybrid_model.py first.")
    st.stop()


@st.cache_resource(show_spinner="Loading GPT-2 and trained head...")
def load_components(checkpoint_mtime_ns):
    # mtime argument invalidates cache when you retrain the checkpoint.
    # Only load checkpoints you created or trust.
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    tokenizer, encoder = load_language_model(state["model_name"])
    head = QuantumTextHead(state["embedding_dim"])
    head.load_state_dict(state["state_dict"])
    head.eval()
    return tokenizer, encoder, head, state


try:
    tokenizer, encoder, head, details = load_components(CHECKPOINT.stat().st_mtime_ns)
except RuntimeError as exc:
    st.error("Model architecture and checkpoint do not match. Retrain using the updated 02_train_hybrid_model.py.")
    st.exception(exc)
    st.stop()

st.caption(f"Training examples: {details.get('train_examples', 'unknown')} | Source: {details.get('dataset_path', 'unknown')}")
metrics = details.get("test_metrics")
if metrics:
    st.caption(f"Held-out accuracy: {metrics['accuracy']:.1%} | Positive F1: {metrics['f1_positive']:.3f} (small test sets are noisy)")

text = st.text_area("Enter a sentence:", value="I absolutely loved this movie.")
if st.button("Predict sentiment", type="primary"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        vector = encode_texts([text], tokenizer, encoder)
        with torch.no_grad():
            prob_pos = torch.softmax(head(vector), dim=1)[0, 1].item()
        label = "Positive" if prob_pos >= 0.5 else "Negative"
        st.subheader(f"Prediction: {label}")
        st.progress(prob_pos, text=f"Positive score: {prob_pos:.1%}")
        if 0.4 <= prob_pos <= 0.6:
            st.warning("Score near 50%: the classifier is uncertain. This is not a calibrated probability.")
        st.caption("This is a small experimental sentiment classifier, not a fully trained LLM or quantum-advantage benchmark.")
