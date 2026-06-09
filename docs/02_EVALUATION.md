# Evaluation Methodology

> What metrics we use, why, and how to interpret them.

---

## The Evaluation Set

`data/evaluation_set.csv` contains **10 hand-crafted (query, expected_chunk_id) pairs**:

```
query                                                              | expected_chunk_id
-------------------------------------------------------------------|-------------------
What is the standard notation for the activation function and …    | chunk_0223
How can we compute the probability of a full sentence using big…   | chunk_0046
How does an encoder-decoder model compute the probability of a …   | chunk_0554
...                                                                | ...
```

**Crucially:** these queries are written in natural language by a human,
phrased differently from the chunk content. This is what makes the evaluation
meaningful — a model that just memorized topic-sentence patterns from training
would score badly here.

---

## The Two Families of Metrics

### Mean Reciprocal Rank (MRR@10)

> "How early in the result list does the correct answer appear?"

For each query, look at where the correct chunk shows up in the top-10
results. Take `1/rank`. Average across all queries.

```
correct chunk at rank 1   →  1/1 = 1.00      (perfect)
correct chunk at rank 2   →  1/2 = 0.50
correct chunk at rank 3   →  1/3 = 0.33
correct chunk at rank 10  →  1/10 = 0.10
correct chunk not in top-10 →  0.00          (miss)
```

`MRR@10 = mean of these scores across all 10 queries.`

**Range:** 0 (always misses) to 1 (always rank 1).

**Why MRR?** It's the standard metric when there's exactly one correct answer
per query (which is our setup). It's also smooth — moving from rank 5 to
rank 3 is a noticeable improvement in the metric.

### Recall@K

> "In what fraction of queries does the correct answer appear in the top K?"

```
Recall@1  = "is the very first result correct?"  (also called accuracy)
Recall@5  = "is the correct result in the top 5?"
Recall@10 = "is the correct result in the top 10?"
```

`Recall@K = (number of queries where correct answer is in top-K) / total queries`

**Range:** 0 to 1.

**Why Recall?** It directly mirrors the user experience of a search engine.
"Did I see the right result on the first page?" is exactly Recall@10 if the
page shows 10 results.

---

## Reading the Numbers

Suppose our final table looks like:

| Metric | BM25 | BiEnc (pretrained) | BiEnc (fine-tuned) |
|---|---|---|---|
| MRR@10 | 0.55 | 0.30 | 0.72 |
| R@1 | 0.40 | 0.20 | 0.60 |
| R@5 | 0.70 | 0.40 | 0.90 |
| R@10 | 0.80 | 0.60 | 1.00 |

What this tells us:

- **BM25 beats untrained BiEncoder.** Out-of-the-box DistilBERT doesn't
  understand search — its embeddings cluster by topic but not by query-document
  relevance. This is the strongest argument for fine-tuning.
- **Fine-tuning helps a lot.** MRR went from 0.30 → 0.72; R@10 from 0.60 → 1.00.
  The contrastive training taught the model the specific relationship between
  query-shaped sentences and document-shaped paragraphs.
- **The fine-tuned model beats BM25.** It's not just "the neural one is fancier" —
  the fine-tuned BiEncoder's R@1 of 0.60 vs BM25's 0.40 means it finds the
  exact right answer first 50% more often.

If your real numbers come out **worse** than BM25, the most common causes are:

1. Training data too small (only 783 pairs — try increasing epochs).
2. Temperature too high — try `temperature=0.05`.
3. Batch size too small — InfoNCE benefits from large batches; try 64 if VRAM allows.
4. Model overfit — check training_log.csv: if val_loss starts going UP while
   train_loss keeps going DOWN, that's overfitting. Reduce epochs or add weight decay.

---

## Why Only 10 Test Queries?

10 is small. For a published paper you'd want 1,000+. For a course project
it's enough to:

- See a clear win/loss against BM25 on each individual query.
- Compute MRR and Recall with enough resolution to detect differences.
- Read every result manually and write a qualitative analysis.

If you want to be thorough in the report, mention that **with 10 queries the
metrics have ~30% variance** — i.e. a single query going from rank 1 to rank
2 changes MRR by 0.05. So a 0.05 difference between BM25 and BiEncoder isn't
meaningful; a 0.20 difference is.

---

## Per-Query Analysis (the qualitative part)

Numbers don't tell the full story. The evaluation notebook also produces a
**per-query rank table** like this:

```
Query                                                  BM25 rank   BiEnc rank
─────────────────────────────────────────────────────────────────────────────
What is the standard notation for the activation…       2          1   ← BiEnc wins
How can we compute the probability of a full sent…      1          1   ← tie
How does an encoder-decoder model compute…              5          2   ← BiEnc wins
What is the role of the embedding matrix…              >10         3   ← BiEnc wins
...
```

This is more useful than the aggregate metrics because you can SEE which kinds
of queries each system handles well. For your report, pick 2–3 examples that
show:
- A query where BM25 is competitive (usually exact-terminology queries).
- A query where BiEncoder dominates (usually paraphrased / conceptual queries).
- A query where both fail (this points to corpus or chunking issues, not model issues).

---

## Why Not Use Other Metrics?

You might see these in IR papers — here's why we don't need them:

| Metric | Why we skip it |
|---|---|
| nDCG@K | Requires graded relevance (1 / 2 / 3-star), not binary. We only have one correct answer per query. |
| Precision@K | With exactly one correct answer per query, P@K = Recall@K / K. Redundant. |
| MAP | Same reasoning — requires multiple relevant docs per query. |
| F1 / accuracy | Designed for classification, not ranking. |

MRR + Recall@{1,5,10} are sufficient and standard.

---

## Reproducibility

The evaluation is **deterministic** because:
- `evaluate_bm25` uses a fixed tokenization (lowercase + strip punctuation).
- `evaluate_biencoder` uses `model.eval()` + `torch.no_grad()` — no dropout,
  no batchnorm noise, identical outputs every run.

You can re-run notebook 03 multiple times and the numbers won't change. If
you re-train the model (notebook 02), the numbers WILL change slightly
because PyTorch's CUDA ops have small non-determinism — usually within ±0.01
on MRR.
