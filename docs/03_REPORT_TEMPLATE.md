# Report Template — Neural Search Engine

> A drop-in skeleton for the written report (30 % of the grade).
> Replace `[...]` with your own text. Tables and graphs come from the notebooks.

---

## 1. Introduction (½ page)

The goal of this project is to build a **Neural Search Engine** for the
textbook *Speech and Language Processing* (Jurafsky & Martin). Given a
free-text query, the system retrieves the most relevant passages from a
pre-processed corpus of 999 chunks (~250 words each).

We compare a **BM25 baseline** against a **fine-tuned BiEncoder** based on
DistilBERT. The BiEncoder is trained with **InfoNCE contrastive loss** on
synthetic query-document pairs extracted automatically from the same corpus.

---

## 2. Data (1 page) — covers the 20 % grading slice

### 2.1 Source

`Speech and Language Processing` (3rd ed., draft), Jurafsky & Martin.
After PDF extraction and cleaning (see `src/utils.py`), the text was split
into 999 fixed-length chunks of ~250 words with 20-word overlap
(`src/pipeline.py`). Each chunk has the format:

```json
{ "id": "chunk_0123", "content": "…", "word_count": 247 }
```

### 2.2 What is a query? What is a document?

- **Document**: a single chunk (~250 words).
- **Query**: a natural-language question whose answer is contained in some
  document.

### 2.3 Training pair generation

We do not have natively-labeled (query, document) pairs. To generate them,
we use **first-sentence query extraction**:

- For each chunk we take its first complete sentence (≥ 8 words, not a
  cross-reference or figure caption) as a **synthetic query**.
- The chunk itself is the **positive document**.
- A random chunk at least 20 positions away is the **negative document**.

This is conceptually similar to the TSDAE / GPL techniques but simpler.
After filtering (some chunks start with figure captions or continuations),
**872 of 999 chunks** yield a usable training pair.

### 2.4 Train / Validation / Test split

| Split | Source | Size | Purpose |
|---|---|---|---|
| Train | synthetic pairs | 783 | gradient updates |
| Val | synthetic pairs | 87 | early-stopping / overfit detection |
| Test | hand-crafted queries (`evaluation_set.csv`) | 10 | final benchmark, never seen during training |

The test set is kept fully independent — these queries are written by a
human and phrased differently from the chunk content, ensuring an honest
evaluation.

### 2.5 Why this data is appropriate

- The corpus is genuine target domain (NLP textbook) so the model learns
  domain-relevant terminology.
- 783 pairs is enough to fine-tune a 66 M-parameter DistilBERT for 5 epochs
  in ~5 minutes on a T4 GPU.
- The synthetic-query trick scales: if we add Wikipedia or arXiv text later,
  no additional human labeling is required.

### 2.6 Quality analysis

[Insert query length histogram from notebook 01.]
[Insert 2–3 sample training pairs and explain whether the query genuinely
relates to the positive.]

Limitations (be honest about these):
- Synthetic queries are extracted from the target chunks → train task is
  easier than the real retrieval task.
- Random negatives don't push the model as hard as hard-negative mining
  would.
- Future work: replace synthetic queries with LLM-generated questions
  (T5 query generator, BeIR-style).

---

## 3. Baseline — BM25 (½ page)

We use the standard `rank_bm25` implementation of BM25-Okapi with default
parameters (k1=1.5, b=0.75). Tokenization is lowercase + strip punctuation +
whitespace split.

**Why BM25:** it is the strongest classical IR baseline, particularly
effective on technical text where domain terminology is repeated verbatim
between query and document.

---

## 4. Main Model — BiEncoder (1 page) — covers the 30 % grading slice

### 4.1 Architecture

```
input_ids [B, 256] → DistilBERT → [B, 256, 768]
                             ↓
                       mean_pooling (masks padding)
                             ↓
                          [B, 768]
                             ↓
                       F.normalize (L2)
                             ↓
                    embedding [B, 768], ‖·‖₂ = 1
```

- **Backbone**: `distilbert-base-uncased` (66 M params). Chosen over
  BERT-base for 2× speed at small accuracy loss.
- **Pooling**: mean over non-padding tokens. Empirically better than `[CLS]`
  for sentence similarity (per SBERT, Reimers & Gurevych 2019).
- **Normalization**: L2 to make cosine similarity = dot product, enabling
  fast retrieval via a single matmul.

### 4.2 Loss function — InfoNCE

InfoNCE (NT-Xent) was chosen over plain TripletLoss for three reasons:
1. With batch size 32 every query gets 31 free in-batch negatives → richer
   gradient signal.
2. No need for hard-negative mining (which is complex and brittle).
3. Used by CLIP, SimCSE, DPR — the de-facto standard for sentence encoders.

The loss for a batch is:
```
L = -1/B  Σ_i  log( exp(sim(qᵢ, pᵢ)/τ)  /  Σ_j  exp(sim(qᵢ, pⱼ)/τ) )
```

with temperature τ = 0.07 (SimCSE default).

### 4.3 Training setup

| Hyperparameter | Value | Rationale |
|---|---|---|
| Batch size | 32 | More in-batch negatives → better InfoNCE signal |
| Epochs | 5 | Converges quickly; val loss plateaus around epoch 4 |
| Learning rate | 2e-5 (peak) | Standard BERT fine-tuning LR |
| LR schedule | linear warmup (10%) + linear decay | Prevents large early updates |
| Optimizer | AdamW, weight decay 0.01 | No decay on bias/LayerNorm |
| Gradient clipping | max-norm 1.0 | Prevents exploding gradients |
| Max sequence length | 256 | Matches chunk length |
| Device | T4 GPU (Colab) | ~5 min total training time |

Training was logged to `checkpoints/training_log.csv` and the best
checkpoint (lowest val_loss) was saved to `checkpoints/best.pt`.

### 4.4 Training curves

[Insert `checkpoints/training_curves.png` from notebook 02.]

The training loss decreased monotonically from ~3.5 (random baseline ≈
log(32) = 3.47) to ~[FILL IN]. The validation loss tracked training loss
closely, indicating no overfitting.

---

## 5. Evaluation (½ page) — covers the 10 % grading slice

### 5.1 Evaluation set

The held-out test set consists of 10 hand-crafted queries with a single
ground-truth chunk each (`data/evaluation_set.csv`). These queries are
phrased in natural language and deliberately do NOT lift wording from the
target chunks — this prevents trivial keyword matching from inflating the
neural model's scores.

### 5.2 Metrics

- **MRR@10** — Mean Reciprocal Rank truncated at 10. Standard for ranked
  retrieval with one correct answer per query.
- **Recall@K** for K ∈ {1, 5, 10} — fraction of queries whose correct
  answer falls in the top-K.

### 5.3 Results

[Insert metrics table from notebook 03 — replace the example below with your real numbers.]

| Metric | BM25 | BiEnc (pretrained) | BiEnc (fine-tuned) |
|---|---|---|---|
| MRR@10 | 0.55 | 0.30 | **0.72** |
| Recall@1 | 0.40 | 0.20 | **0.60** |
| Recall@5 | 0.70 | 0.40 | **0.90** |
| Recall@10 | 0.80 | 0.60 | **1.00** |

[Insert `checkpoints/evaluation_results.png` bar chart.]

### 5.4 Qualitative analysis

[Pick 2–3 example queries from notebook 03 — show side-by-side results
from BM25 and BiEncoder, explain why the BiEncoder wins or loses.]

Example:
> Query: "How do machines understand the meaning of a sentence?"
>
> BM25's top-1 result is unrelated (chunk on `[…]`) because the query uses
> no domain-specific keywords.
>
> BiEncoder's top-1 result is `chunk_0XXX` on semantic role labeling —
> correctly identifying that "meaning understanding" relates to semantic
> parsing despite zero keyword overlap.

---

## 6. Discussion (½ page)

- **What worked well:** [...]
- **What didn't work:** [...]
- **Limitations:**
  - Small evaluation set (10 queries) → high variance in metrics.
  - Synthetic training data has limited diversity.
  - No hard-negative mining.
- **Future work:**
  - Generate queries with an LLM (T5 / Mistral) for better diversity.
  - Implement hard-negative mining using BM25 results as initial candidates.
  - Try larger encoders (bge-small, e5-small) for comparison.

---

## 7. Reproducibility

All code lives in the project repo. To reproduce:

```bash
uv sync                                            # install dependencies
jupyter notebook notebooks/01_data_preparation.ipynb  # generate train/val pairs
jupyter notebook notebooks/02_training.ipynb          # train (GPU required)
jupyter notebook notebooks/03_evaluation.ipynb        # compute metrics
jupyter notebook notebooks/04_demo.ipynb              # interactive search
```

Or on Google Colab: upload `Neural_Search_Engine/` to Drive and run the
notebooks in order.

---

## 8. References

- Jurafsky, D. & Martin, J. *Speech and Language Processing* (3rd ed., draft).
- Sanh et al., "DistilBERT, a distilled version of BERT" (2019).
- Reimers, N. & Gurevych, I. "Sentence-BERT" (EMNLP 2019).
- Gao et al., "SimCSE: Simple Contrastive Learning of Sentence Embeddings" (EMNLP 2021).
- van den Oord et al., "Representation Learning with Contrastive Predictive Coding" (2018) — the original InfoNCE paper.
- Robertson, S. & Zaragoza, H. "The Probabilistic Relevance Framework: BM25 and Beyond" (2009).
