# Project Overview — Neural Search Engine

> **⚠️ Historical note.** Parts of this document describe the *earlier* version
> that used a pretrained DistilBERT backbone. Pretrained models were later
> **prohibited**, so the encoder and tokenizer were rebuilt **from scratch**.
> For the current architecture see [`06_FROM_SCRATCH_MODEL.md`](06_FROM_SCRATCH_MODEL.md)
> and [`REPORT_GE.md`](REPORT_GE.md). The data, BM25 baseline, and evaluation
> framework below are unchanged.

> **Read this first.** Everything else makes more sense after this page.

---

## 1. The Big Picture

You are building a system that takes a free-text question and finds the most
relevant passages from *Speech and Language Processing* (Jurafsky & Martin):

```
user query → text encoder → 768-dim vector
                                ↓
                            cosine similarity
                                ↓
            corpus of 999 chunk vectors (pre-computed)
                                ↓
                top-k most similar chunks → shown to user
```

Two systems do the encoding:

| System | Type | Purpose |
|---|---|---|
| **BM25** | Classical (statistical) | Baseline — the model you must beat |
| **BiEncoder** | Neural (DistilBERT + mean-pool + L2-norm) | Main model you train and evaluate |

---

## 2. What is the Baseline?

**`BM25Okapi` from `rank_bm25`.** Already implemented in `src/search_engine.py`
(your original file). The logic:

1. Tokenize each document: lowercase + remove punctuation + split on whitespace.
2. Build inverted index: for every word, store which docs contain it and how often.
3. At query time: score each document by `BM25(doc, query)` — a TF-IDF-like
   formula with length normalization and term-frequency saturation.

**Why BM25 is a strong baseline:**
- It dominates information retrieval for 25+ years on most benchmarks.
- It works out-of-the-box with zero training data.
- It's particularly strong on technical / scientific text where domain-specific
  terms are repeated verbatim — exactly your situation.

**Where BM25 fails:**
- Paraphrased questions ("How does a machine learn to translate?" won't find the
  chunk on "encoder-decoder architectures" if those exact words don't appear).
- Synonyms (BM25 doesn't know "rapid" ≈ "fast").
- Conceptual questions that require understanding rather than keyword overlap.

This is precisely where your **BiEncoder** is supposed to win.

---

## 3. What is the Main Model?

A **Bi-Encoder** (also called a "dual encoder" or "siamese encoder") built from:

```
        ┌─────────────────────────────────────────────────────────┐
        │                  BiEncoder Architecture                 │
        ├─────────────────────────────────────────────────────────┤
        │                                                         │
        │  input_ids [B, 256] ──► DistilBERT ──► [B, 256, 768]   │
        │                                              │          │
        │                                              ▼          │
        │                                       mean_pooling       │
        │                                       (masks padding)   │
        │                                              │          │
        │                                              ▼          │
        │                                          [B, 768]       │
        │                                              │          │
        │                                              ▼          │
        │                                       F.normalize       │
        │                                       (L2-norm = 1)     │
        │                                              │          │
        │                                              ▼          │
        │                                   embedding [B, 768]    │
        └─────────────────────────────────────────────────────────┘
```

Why this design?

| Component | Choice | Reason |
|---|---|---|
| Backbone | DistilBERT | 66 M params, half of BERT-base. Trains in minutes on a T4. |
| Pooling | Mean pooling | Empirically beats `[CLS]` for sentence similarity (per SBERT paper). |
| Normalization | L2 | Makes cosine sim = dot product → fast retrieval via matmul. |
| Output dim | 768 | DistilBERT's native hidden size. No projection layer needed. |

See `BIENCODER_STUDY_GUIDE.md` for line-by-line math.

---

## 4. What Do We Train the Model On?

**The core problem:** Contrastive training needs (query, relevant_chunk, irrelevant_chunk)
triplets — but we only have 10 hand-crafted ones in `evaluation_set.csv`.

**The solution: synthetic query generation.** For every chunk, we extract its
first complete sentence as a synthetic query. The full chunk is the relevant
document. A random distant chunk is the irrelevant document.

```
Example Chunk:
    "A bigram language model assigns a probability to each word in a sequence
     based on the previous word. P(w_n | w_{n-1}) is estimated from corpus
     counts. The model is useful for spelling correction, autocompletion, and
     speech recognition because it captures local word-pair statistics …"

Generated training triplet:
    QUERY    : "A bigram language model assigns a probability to each word in
                a sequence based on the previous word."
    POSITIVE : (the full chunk above)
    NEGATIVE : (a random chunk about, say, parser evaluation, sampled from
                a chunk at least 20 positions away in the book)
```

**Why this works:**
- Textbook paragraphs are written with a topic sentence first → the sentence
  genuinely *summarizes* what the paragraph contains.
- This gives us **~870 training pairs from 999 chunks** (the 127 chunks whose
  first sentence is a figure caption or continuation are filtered out).
- The model never sees the 10 human-written test queries during training, so
  evaluation is honest.

**Splits:**
```
999 raw chunks
   ├── ~870 yield a usable training pair
   │      ├── 783 train pairs   (90 %)
   │      └──  87 validation pairs (10 %)  — used to detect overfitting
   └── 10 hand-crafted queries (held-out test) — used ONLY for final evaluation
```

See `01_TRAINING_DATA.md` for the full explanation of why this synthetic
approach is valid and what its limitations are.

---

## 5. What Loss Function Do We Use?

**InfoNCELoss** (a.k.a. NT-Xent, in-batch negatives loss).

Given a batch of 32 (query, positive) pairs, we build a 32×32 similarity matrix:

```
            positive_1    positive_2    positive_3   …  positive_32
query_1    [ HIGH  ]    [  low   ]    [  low   ]   …  [  low   ]    ← row 1: query 1's true match is position 1
query_2    [  low   ]    [ HIGH  ]    [  low   ]   …  [  low   ]
query_3    [  low   ]    [  low   ]    [ HIGH  ]   …  [  low   ]
   ⋮
query_32   [  low   ]    [  low   ]    [  low   ]   …  [ HIGH  ]
```

The loss is cross-entropy applied row-wise: each row should look like a
softmax that picks out the diagonal. With batch size 32, every query gets
**31 free negatives** per training step — much richer signal than a single
hand-picked negative would give.

**Why InfoNCE over plain TripletLoss?**
- More negatives per step → faster convergence.
- No need to do expensive "hard negative mining."
- Standard choice in modern contrastive learning (CLIP, SimCSE, DPR all use it).

See `src/loss.py` for the full mathematical breakdown.

---

## 6. How Do We Know It Works? — Evaluation

We use the 10 hand-crafted queries from `data/evaluation_set.csv` as a held-out
test set. For each query we have the ground-truth chunk ID.

**Metrics computed in `src/evaluate.py`:**

| Metric | Meaning |
|---|---|
| MRR@10 | Mean Reciprocal Rank — `1/rank_of_correct_answer`, averaged across queries |
| Recall@1 | Was the very first result correct? |
| Recall@5 | Was the correct answer somewhere in the top-5? |
| Recall@10 | Was the correct answer somewhere in the top-10? |

All four metrics are reported for **three systems**:
1. BM25 (baseline)
2. BiEncoder pre-trained (DistilBERT with no fine-tuning) — a sanity check that
   our fine-tuning actually helps
3. BiEncoder fine-tuned (our trained model)

The comparison happens in `notebooks/03_evaluation.ipynb` and produces
`checkpoints/evaluation_results.png` for your report.

---

## 7. The Files at a Glance

```
Neural_Search_Engine/
├── data/
│   ├── jurafsky_martin.pdf                  (raw textbook)
│   ├── evaluation_set.csv                   (10 test queries — DO NOT TOUCH for training)
│   └── processed/
│       ├── jurafsky_chunks.json             (999 chunks — already exists)
│       ├── train_pairs.json                 (generated by notebook 01)
│       └── val_pairs.json                   (generated by notebook 01)
│
├── src/
│   ├── utils.py                             (PDF extraction — already exists)
│   ├── pipeline.py                          (chunking — already exists)
│   ├── search_engine.py                     (BM25 — already exists)
│   ├── baseline.py                          (placeholder, can be ignored)
│   ├── model.py            ★ NEW            (BiEncoder architecture)
│   ├── dataset.py          ★ NEW            (training pair generation + PyTorch Datasets)
│   ├── loss.py             ★ NEW            (InfoNCE + Triplet loss implementations)
│   ├── train.py            ★ NEW            (full training loop)
│   ├── vector_store.py     ★ NEW            (encode-once / search-many)
│   └── evaluate.py         ★ NEW            (MRR@10, Recall@K, BM25 vs BiEnc comparison)
│
├── notebooks/
│   ├── biencoder_walkthrough.ipynb          (educational — math walkthrough)
│   ├── colab_gpu_setup.ipynb                (Colab setup helper)
│   ├── 01_data_preparation.ipynb  ★ NEW     (generate train/val pairs from chunks)
│   ├── 02_training.ipynb          ★ NEW     (fine-tune BiEncoder on GPU)
│   ├── 03_evaluation.ipynb        ★ NEW     (BM25 vs BiEncoder metrics)
│   └── 04_demo.ipynb              ★ NEW     (interactive search demo)
│
├── docs/
│   ├── BIENCODER_STUDY_GUIDE.md             (deep dive on the architecture)
│   ├── 00_PROJECT_OVERVIEW.md  ★ NEW        (this file)
│   ├── 01_TRAINING_DATA.md     ★ NEW        (why synthetic queries are valid)
│   ├── 02_EVALUATION.md        ★ NEW        (MRR vs Recall, how to read the numbers)
│   └── 03_REPORT_TEMPLATE.md   ★ NEW        (Georgian-friendly skeleton for the final report)
│
├── checkpoints/             (created during training)
│   ├── best.pt
│   ├── final.pt
│   ├── training_log.csv
│   ├── training_curves.png
│   └── evaluation_results.png
│
├── pyproject.toml
└── README.md
```

---

## 8. How to Run Everything End-to-End

### Local (Windows, CPU) — for code dev and small tests

```powershell
uv sync                                              # install all deps
uv run python src/model.py                           # smoke test the architecture
uv run python -m src.dataset                         # smoke test dataset generation
uv run python -m src.evaluate                        # baseline numbers (no training)
```

### Google Colab (T4 GPU) — for training and full pipeline

1. Upload the **whole** `Neural_Search_Engine` folder to Google Drive.
2. Open `notebooks/01_data_preparation.ipynb` in Colab → Run all cells.
3. Open `notebooks/02_training.ipynb` → Set runtime to T4 GPU → Run all cells.
   - ~5–8 min training time.
4. Open `notebooks/03_evaluation.ipynb` → Run all cells. Outputs the comparison
   table and bar chart you'll use in your report.
5. Open `notebooks/04_demo.ipynb` → Run all cells and try your own queries
   for screenshots in the report.

---

## 9. Mapping to Grading Criteria

| Criterion | Weight | Where it's covered |
|---|---|---|
| Data collection & pair generation | 20% | `notebooks/01_data_preparation.ipynb` + `docs/01_TRAINING_DATA.md` |
| Model training (code, logging, etc.) | 30% | `src/train.py` + `notebooks/02_training.ipynb` + `checkpoints/training_log.csv` |
| Evaluation set + metrics | 10% | `data/evaluation_set.csv` + `src/evaluate.py` + `notebooks/03_evaluation.ipynb` |
| Report (architecture, training graphs, eval analysis) | 30% | `docs/03_REPORT_TEMPLATE.md` |
| Clean, documented code | implicit | All `src/*.py` files have docstrings + type hints |

---

## 10. Next Steps After Reading This

1. Read `01_TRAINING_DATA.md` — understand *why* the synthetic query approach is valid.
2. Read `02_EVALUATION.md` — understand what MRR and Recall mean in plain language.
3. Run the four notebooks in order.
4. Read `03_REPORT_TEMPLATE.md` to start drafting your written report.
