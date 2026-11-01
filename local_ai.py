"""Use Ollama entirely on localhost. No paid API or authentication key."""
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

OLLAMA_URL = "http://127.0.0.1:11434"
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "qwen3:1.7b")


def request_ollama(endpoint: str, payload: dict) -> dict:
    request = Request(
        OLLAMA_URL + endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=300) as response:
            return json.load(response)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {body[:700]}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(
            "Cannot reach local Ollama at http://127.0.0.1:11434. "
            "Open the Ollama app (or run 'ollama serve') and try again."
        ) from exc


def embed_texts(texts: list[str], model: str = EMBED_MODEL, batch_size: int = 8):
    """Return a float32 tensor [number_of_texts, embedding_dimension]."""
    import torch
    embeddings = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        response = request_ollama("/api/embed", {"model": model, "input": chunk})
        vectors = response.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(chunk):
            raise RuntimeError(f"Unexpected Ollama embedding response for batch {start}: {response}")
        embeddings.extend(vectors)
        print(f"Embedded {min(start + len(chunk), len(texts))}/{len(texts)}")
    if not embeddings:
        raise ValueError("No text supplied to embed_texts")
    tensor = torch.tensor(embeddings, dtype=torch.float32)
    if tensor.ndim != 2 or not torch.isfinite(tensor).all():
        raise RuntimeError("Ollama returned invalid embedding vectors")
    return tensor


def ask_local_model(text: str, system: str, model: str = CHAT_MODEL, json_output: bool = False) -> str:
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        "options": {"temperature": 0},
    }
    # Qwen 3 can skip its reasoning trace; GPT-OSS requires a level instead.
    payload["think"] = "low" if model.startswith("gpt-oss") else False
    if json_output:
        payload["format"] = "json"
    response = request_ollama("/api/chat", payload)
    return str(response.get("message", {}).get("content", "")).strip()
