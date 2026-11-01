"""Streamlit: OpenAI embeddings -> trained PennyLane classifier; optional GPT comparison."""
from pathlib import Path
import streamlit as st
import torch
from openai_quantum_head import QuantumTextHead
from openai_text_api import EMBED_MODEL, embed_texts, classify_with_gpt

ROOT=Path(__file__).resolve().parent
CHECKPOINT=ROOT/"models"/"openai_quantum_head.pt"
st.set_page_config(page_title="OpenAI + PennyLane Sentiment",page_icon="⚛️")
st.title("⚛️ OpenAI Embeddings + PennyLane")
st.caption("OpenAI supplies embeddings; the four-qubit PennyLane head is trained locally. GPT is optional for comparison.")
if not CHECKPOINT.exists():
    st.error("Checkpoint missing. First run 12_prepare_openai_embeddings.py, then 13_train_openai_quantum.py")
    st.stop()

@st.cache_resource

def load_model(file_mtime):
    checkpoint=torch.load(CHECKPOINT,map_location="cpu",weights_only=True)
    if checkpoint.get("architecture")!="openai-q4-classical-skip-v1":
        raise ValueError("Unexpected checkpoint architecture. Retrain with 13_train_openai_quantum.py")
    head=QuantumTextHead(checkpoint["embedding_dim"])
    head.load_state_dict(checkpoint["state_dict"]);head.eval()
    return checkpoint,head

try:
    checkpoint,head=load_model(CHECKPOINT.stat().st_mtime_ns)
except Exception as exc:
    st.error(f"Could not load checkpoint: {exc}");st.stop()

st.write(f"Training records: {checkpoint['train_examples']} | Encoder: {checkpoint['embedding_model']}")
metrics=checkpoint.get("test_metrics")
if metrics:
    st.caption(f"Held-out test accuracy: {metrics['accuracy']:.1%} | positive F1: {metrics['positive_f1']:.3f}")
text=st.text_area("Sentence to analyze", "I really enjoyed this experiment.")
if st.button("Predict with quantum classifier",type="primary"):
    if not text.strip():st.warning("Enter some text first.")
    else:
        try:
            vector=embed_texts([text], model=checkpoint["embedding_model"])
            if vector.shape[1]!=checkpoint["embedding_dim"]:
                raise ValueError("Embedding dimensions differ from checkpoint")
            with torch.no_grad():
                positive=torch.softmax(head(vector),dim=1)[0,1].item()
            st.subheader("Positive" if positive>=0.5 else "Negative")
            st.progress(positive,text=f"Positive score: {positive:.1%}")
            st.caption("Score is not a calibrated probability.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

st.divider()
st.subheader("Optional: compare with a hosted GPT model")
gpt=st.selectbox("GPT model",["gpt-6-luna","gpt-4o"])
st.caption("This performs a separate API classification call. It is not quantum training and will incur API usage.")
if st.button("Compare with GPT"):
    if not text.strip():st.warning("Enter some text first.")
    else:
        try:
            st.info(f"{gpt} says: {classify_with_gpt(text,model=gpt)}")
        except Exception as exc:
            st.error(f"GPT API call failed: {exc}")
