# Neural Search Engine (from scratch)

A small **neural semantic search engine** over the Jurafsky & Martin textbook
*Speech and Language Processing*. Given a free-text query it returns the top-k
most relevant 200–300-word passages.

Everything is built **from scratch with only `torch`** — no pretrained models, no
`transformers` / `sentence-transformers`:

```
query → BPE tokenizer → Transformer encoder → query embedding → cosine search → top-k chunks
```

## Components
| File | What it is |
|---|---|
| `src/tokenizer.py` | Byte-Pair-Encoding tokenizer learned from the corpus (~8k vocab) |
| `src/model.py` | Transformer encoder: token + sinusoidal positional embeddings, multi-head self-attention, FFN, pre-norm residuals, mean pooling, L2 norm (`BiEncoder`) |
| `src/loss.py` | Symmetric InfoNCE contrastive loss (in-batch negatives) |
| `src/dataset.py` | `(query, positive)` dataset with dynamic-padding collate |
| `src/train.py` | `Trainer` (AdamW + warmup, CSV logging, checkpoints) + `build_tokenizer_from_pairs` |
| `src/vector_store.py` | Encode-once / search-many dense index |
| `src/evaluate.py` | BM25 baseline + MRR@10 / Recall@k + per-query-type breakdown |

## Run
The notebooks are written for **Google Colab (T4 GPU)**:

1. `notebooks/01_data_v2_llm.ipynb` — chunk the PDF + generate training queries (data prep).
2. `notebooks/02_training.ipynb` — train the BPE tokenizer, then the encoder; save `checkpoints/`.
3. `notebooks/03_evaluation.ipynb` — BM25 vs untrained vs trained encoder; metrics + plots.
4. `notebooks/04_demo.ipynb` — interactive side-by-side search demo.

Local sanity checks (CPU):
```bash
uv sync
uv run python -m src.tokenizer    # BPE round-trip
uv run python -m src.model        # encoder shape/norm smoke test
uv run python -m src.loss         # InfoNCE ≈ log(B)
uv run python -m src.evaluate     # BM25 + (untrained) neural metrics
```

## Docs
- `docs/REPORT_GE.md` — full project report (Georgian).
- `docs/06_FROM_SCRATCH_MODEL.md` — architecture deep dive (English).
