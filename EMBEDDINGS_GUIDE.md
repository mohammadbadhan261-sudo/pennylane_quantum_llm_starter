# Embed Common Crawl with AI and train PennyLane (VS Code / Windows)

This project converts **real Common Crawl web text** into fixed-length numerical **DistilGPT2 embeddings** and trains the **existing PennyLane + PyTorch sentiment head** on reviewed labeled examples. **It is not full GPT-2 fine-tuning.** The 4-qubit circuit runs on the `default.qubit` simulator, not quantum hardware.

## 1. Preparation

Open the project folder in VS Code. Use your existing working Python environment (`.venv`). If you already installed the earlier project dependencies, you only need the Common Crawl WET parser:

```powershell
python -m pip install warcio
```

If starting with a fresh Python 3.12 environment, use `python -m pip install -r requirements.txt` and `python -m pip install -r requirements-commoncrawl.txt`. Do **not** install the conflicting `datasets` package to use this pipeline. The first AI embedding run downloads the pretrained DistilGPT2 weights.

## 2. Download real web text (no sentiment labels yet)

```powershell
python 07_fetch_commoncrawl.py --target 50 --max-wet-files 3 --max-mb-per-file 30
```

Output: `data/commoncrawl_raw.csv` containing `text`, `source_url`, and `wet_path`. This is a random sample of plaintext excerpts from Common Crawl WET files, **not an already-curated sentiment dataset**. Downloading can take time and may return fewer than 50 useful sentences. Online access required; please review copyright, privacy, source material and unsuitable/PII text.

## 3. Convert raw Common Crawl text into AI embeddings

```powershell
python 10_embed_dataset.py --input data/commoncrawl_raw.csv --output data/commoncrawl_raw_embeddings.pt
```

Output: a `.pt` tensor dataset with `embeddings` shape `(N, 768)`, `labels` (all `-1` for unlabeled), text, source URLs, encoder metadata, and a SHA-256 record fingerprint. Each sentence is tokenized with DistilGPT2, truncated to 64 tokens and mask-mean-pooled into a numerical vector. This DOES embed the web text, but the embeddings alone CANNOT teach a sentiment classifier what is positive or negative.

Optionally test the embeddings using similarity search:

```powershell
python 11_search_embeddings.py --cache data/commoncrawl_raw_embeddings.pt --query "an excellent experience"
```

Similarity is illustrative, not a calibrated quality measure. DistilGPT2 is not specifically trained as a sentence-embedding model.

## 4. Label the sentences using a local AI model (plus human review)

Install Ollama (https://ollama.com/download), then run:

```powershell
ollama pull llama3.2:1b
python 08_label_commoncrawl.py --max-rows 50
```

Output: `data/commoncrawl_labeled.csv`. Open that file in Excel or VS Code. For each suitable sentiment statement, check/correct `label` (`1` positive, `0` negative) and change `reviewed` to `yes`. Keep neutral and irrelevant sentences unlabeled/unused. Do not trust unreviewed AI-generated labels as ground truth.

## 5. Mix the reviewed Common Crawl examples with your existing sentiment dataset

For the original sample:

```powershell
python 09_merge_training_data.py --base data/sentences.csv
```

Or, if you downloaded your larger IMDb examples:

```powershell
python 09_merge_training_data.py --base data/imdb_sentences.csv
```

This produces **`data/train_mixed.csv`**. Every row has a text plus a checked binary sentiment label. You will need at least five examples per class. Some crawled text will be filtered out.

## 6. Embed the labeled mixed dataset and train directly from embeddings

```powershell
python 10_embed_dataset.py --input data/train_mixed.csv --output data/train_mixed_embeddings.pt
python 02_train_hybrid_model.py --embeddings data/train_mixed_embeddings.pt --dataset data/train_mixed.csv --epochs 20 --batch-size 8 --lr 0.003
```

The training script **reuses saved vectors** and verifies the saved dataset matches the CSV. It still uses an 80/20 random stratified train/test split for a quick diagnostic. For a research paper, use a separately prepared and untouched real test set with near-duplicates removed across splits; don't claim "quantum advantage" without a matched classical-only baseline.

Test and launch:

```powershell
python 05_check_predictions.py
python -m streamlit run 03_app.py
```

At prediction time, Streamlit uses the **same DistilGPT2 encoder** to compute features for the typed sentence. Do not change to another embedding model without changing both the training and inference pipeline.

## 7. GitHub commits

`.gitignore` keeps `data/*.csv` (except the original sample), `*.pt`, `models/`, `.venv` out of your repo. These files can contain copyrighted text, private data or pretrained model parameters. Commit *scripts*, documentation and safe sample schemas instead:

```powershell
git status
git add .gitignore README.md EMBEDDINGS_GUIDE.md GITHUB_GUIDE.md requirements.txt requirements-commoncrawl.txt 02_train_hybrid_model.py 10_embed_dataset.py 11_search_embeddings.py 07_fetch_commoncrawl.py 08_label_commoncrawl.py 09_merge_training_data.py
git commit -m "Add Common Crawl AI embedding cache and hybrid training"
git push
```

If you have not created a repo, read `GITHUB_GUIDE.md` first. Never commit tokens, private text or unreviewed crawled data.

## What exactly is learned?

- DistilGPT2 embeddings = computed features from a frozen pretrained language model (no LLM weight updates).
- PennyLane + classical head = trainable part that learns binary sentiment labels.
- Embedding search = finds passages with similar vectors; it is not an LLM chatbot or RAG until a retrieval-to-generation stage is added.

## Sources

- Common Crawl archive: https://www.commoncrawl.org/blog/october-2026-crawl-archive-now-available
- Common Crawl access: https://commoncrawl.org/get-started
- PennyLane: https://docs.pennylane.ai/
