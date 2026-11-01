# Free local AI + PennyLane sentiment classifier

This is an **add-on** to your existing PennyLane project, not a GPT-4o installation. GPT-4o is hosted and its weights cannot be downloaded for your VS Code app. This add-on uses Ollama locally; no OpenAI API key or billing is required.

## Models

- `nomic-embed-text`: turns text into vectors; used for the PennyLane classifier.
- `qwen3:1.7b`: optional local sentiment labeling and comparisons. This is an Apache 2.0 licensed open-weight model. Increase to `qwen3:4b` for a larger model.
- Optional on a powerful PC: `gpt-oss:20b` (around 14 GB download, recommended >=16 GB memory). This is OpenAI's **open-weight** model, **not GPT-4o**. To choose it in the app, install with `ollama pull gpt-oss:20b` and enter `gpt-oss:20b` in the comparison model field.

## Setup on Windows VS Code

1. Install Ollama from https://ollama.com/download and open the app.
2. In PowerShell: `ollama pull nomic-embed-text`
3. In PowerShell: `ollama pull qwen3:1.7b`
4. Keep your original `.venv`. In your project terminal activate it with `.\.venv\Scripts\Activate.ps1`.
5. `python -m pip install -r requirements-local.txt`
6. Confirm Ollama is running: `ollama list`.

## First free local AI training

Run in the extracted add-on folder (or copy the scripts into your existing project):

```powershell
python 01_prepare_embeddings.py --input data/sentences.csv --output data/local_embeddings.pt
python 02_train_local_quantum.py --embeddings data/local_embeddings.pt --epochs 20 --batch-size 8 --lr 0.003
python -m streamlit run 03_app_local.py
```

The model saves `models/local_quantum_head.pt` so it does **not** overwrite your GPT-2 experiments.

**Note:** The sample dataset is only 40 sentences; it demonstrates the training pipeline, not reliable accuracy. With real data, use a larger, carefully labeled and balanced CSV and a separate untouched test set.

## Combine with your existing Common Crawl workflow

Use your earlier `07_fetch_commoncrawl.py` to download **real** web text to `data/commoncrawl_raw.csv`. The text file must contain a column named `text`. Then:

```powershell
python 04_label_commoncrawl_local.py --input data/commoncrawl_raw.csv --max-rows 50
```

Open `data/commoncrawl_labeled.csv`, correct all labels, and mark accepted rows `reviewed=yes`. Most web text is neutral; do not force sentiment labels. Then:

```powershell
python 05_merge_reviewed.py --base data/sentences.csv --review data/commoncrawl_labeled.csv
python 01_prepare_embeddings.py --input data/train_mixed.csv --output data/local_embeddings.pt
python 02_train_local_quantum.py --embeddings data/local_embeddings.pt --epochs 20 --batch-size 8 --lr 0.003
python -m streamlit run 03_app_local.py
```

If you already have IMDb labels at `data/imdb_sentences.csv`, use that file for `--base`.

For **unlabeled** Common Crawl embeddings (for future semantic retrieval rather than sentiment training):

```powershell
python 01_prepare_embeddings.py --input data/commoncrawl_raw.csv --output data/commoncrawl_unlabeled.pt
```

## Switch to a larger local LLM

The training embeddings should stay generated from the same model (`nomic-embed-text`). The LLM chat model is separate.

```powershell
ollama pull qwen3:4b
$env:OLLAMA_CHAT_MODEL = 'qwen3:4b'
python -m streamlit run 03_app_local.py
```

Or use `gpt-oss:20b` if the PC supports it. Neither choice changes the embedding dimensionality. If you **change the embedding model**, recreate the embedding file and retrain the PennyLane head.

## GitHub

The included `.gitignore` excludes downloaded Common Crawl text, embedding caches, keys, and trained weights. Review `git status` before pushing.

```powershell
git init
git add .gitignore README.md requirements-local.txt local_ai.py local_quantum_model.py 01_prepare_embeddings.py 02_train_local_quantum.py 03_app_local.py 04_label_commoncrawl_local.py 05_merge_reviewed.py data/sentences.csv
git commit -m 'Add free offline Ollama embeddings and PennyLane classifier'
```

Connect your GitHub remote and `git push` separately.

## What is actually trained?

- The local LLM (Qwen3 or gpt-oss) is **used for inference/labeling**, not fine-tuned here.
- The Nomic embedding model is **frozen**.
- Only the PennyLane circuit and the small classifier network are **trained**.
- Local weights can be downloaded without usage billing but use your computer's memory, storage, and electricity.

## Limitations

This is educational research code. No external model download, live Common Crawl retrieval or full quantum training was executed when packaging this add-on. The scripts were syntax checked and key data-path behavior checked separately. Evaluating only on a small random subset is not a reliable scientific benchmark; use official held-out test splits for any research claims.
