"""Run with: python -m streamlit run 03_app.py"""
from pathlib import Path

import streamlit as st
import torch

from hybrid_model import QuantumTextHead, encode_texts, load_language_model

st.set_page_config(page_title="Quantum + GPT-2 Demo", page_icon="⚛️")
st.title("⚛️ PennyLane + GPT-2 Sentiment Demo")
st.caption("A pretrained language model with a simulated, trained quantum classification head.")

CHECKPOINT = Path(__file__).resolve().parent / "models" / "quantum_head.pt"
if not CHECKPOINT.exists():
    st.error("No trained model found. First run: python 02_train_hybrid_model.py")
    st.stop()


@st.cache_resource(show_spinner="Loading pretrained GPT-2 and trained quantum head...")
def load_components():
    # Only load checkpoints that you created or trust.
    data = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    tokenizer, encoder = load_language_model(data["model_name"])
    head = QuantumTextHead(data["embedding_dim"])
    head.load_state_dict(data["state_dict"])
    head.eval()
    return tokenizer, encoder, head, data.get("train_examples")


tokenizer, encoder, head, training_count = load_components()
user_text = st.text_area("Type a sentence in English:", value="This is a fantastic experiment.")

if st.button("Predict sentiment", type="primary"):
    if not user_text.strip():
        st.warning("Please type a sentence first.")
    else:
        features = encode_texts([user_text], tokenizer, encoder)
        with torch.no_grad():
            scores = torch.softmax(head(features), dim=1)[0]
        positive_probability = scores[1].item()
        prediction = "Positive" if positive_probability >= 0.5 else "Negative"
        st.subheader(f"Prediction: {prediction}")
        st.progress(positive_probability, text=f"Model positive score: {positive_probability:.1%}")
        st.info("Experimental sentiment classifier. Scores are not calibrated confidence estimates. Training on more examples does not guarantee accuracy.")

st.markdown("**Quantum configuration:** 4 simulated qubits, 2 trainable entangling layers, PennyLane `default.qubit`.")
st.markdown("**Training:** Frozen GPT-2 embeddings. Only the hybrid head is trained.")

if training_count is not None:
    st.caption(f"Checkpoint trained on {training_count:,} labeled sentences.")
