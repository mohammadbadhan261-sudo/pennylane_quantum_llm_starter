"""Streamlit demo. No remote API or paid keys. Run with streamlit run 03_app_local.py."""
from pathlib import Path
import streamlit as st
import torch
from local_ai import CHAT_MODEL, ask_local_model, embed_texts
from local_quantum_model import LocalQuantumHead

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "models" / "local_quantum_head.pt"
st.set_page_config(page_title="Free Local Quantum AI", page_icon="⚛️")
st.title("⚛️ Local AI + PennyLane")
st.caption("No API key. Ollama runs language and embedding models on your own computer.")

if not CHECKPOINT.is_file():
    st.error("Model not trained. Run 01_prepare_embeddings.py and 02_train_local_quantum.py first.")
    st.stop()


@st.cache_resource(show_spinner="Loading saved PennyLane model...")
def load_head(path: str, modified: int):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    head = LocalQuantumHead(embedding_dim=checkpoint["embedding_dim"])
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()
    return head, checkpoint


head, metadata = load_head(str(CHECKPOINT), CHECKPOINT.stat().st_mtime_ns)
st.write("Embedding model:", metadata["embedding_model"])
st.write("Training examples:", metadata["train_examples"])
st.write("Held-out evaluation:", metadata["metrics"])
user_text = st.text_area("Enter an English sentence", value="I really enjoyed this experiment!")

if st.button("Predict using PennyLane"):
    if user_text.strip():
        try:
            vector = embed_texts([user_text.strip()], model=metadata["embedding_model"])
            with torch.no_grad():
                score = torch.softmax(head(vector), dim=1)[0,1].item()
            st.subheader("Positive" if score >= 0.5 else "Negative")
            st.progress(score, text=f"Positive score: {score:.1%}")
            st.caption("The score is not calibrated confidence or held-out accuracy.")
        except Exception as exc:
            st.error(str(exc))
    else:
        st.warning("Enter a sentence.")

st.divider()
st.subheader("Optional: compare with your local LLM")
st.caption("This is a separate AI prediction, not the trained quantum classifier.")
chat_model = st.text_input("Local Ollama chat model", value=CHAT_MODEL)
if st.button("Ask local AI"):
    try:
        response = ask_local_model(
            user_text,
            "Classify the user's text as Positive, Negative, or Neutral. Return the label and one short explanation.",
            model=chat_model,
        )
        st.write(response)
    except Exception as exc:
        st.error(str(exc))
