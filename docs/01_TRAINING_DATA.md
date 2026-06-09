# Training Data Strategy

> Why we generate synthetic queries, why that's valid, and what its limitations are.

---

## The Core Problem

Contrastive learning needs many `(query, positive, negative)` examples. But our
domain (the Jurafsky textbook) doesn't come with such labels. Our only labeled
data is `data/evaluation_set.csv` — and that's just **10 queries**, kept as
the test set.

We need on the order of **hundreds or thousands** of training pairs.

## The Solution: First-Sentence as Query

For every chunk in `data/processed/jurafsky_chunks.json`, we extract its
first complete sentence and use it as a synthetic query. The full chunk is
the positive document. A random distant chunk is the negative.

### A concrete example

**Chunk #50 (raw text):**
> "High probability of sentences beginning with the words I. And some might
> even be cultural rather than linguistic, like the higher probability that
> people are looking for Chinese versus English food. 3.1.3 Dealing with
> scale in large n-gram models In practice, language models can be very
> large, leading to memory problems on real devices …"

**Extracted query (first usable sentence):**
> "High probability of sentences beginning with the words I."

**Positive document:** the full chunk above.

**Negative document:** some random chunk like #700 about semantic role labeling.

## Why This is Valid

### 1. Topic sentences exist in textbooks

A well-written textbook paragraph opens with a **topic sentence** — a one-line
summary of what the paragraph claims. This is a writing convention specifically
designed so readers can skim. It means the first sentence is genuinely a
"compressed version" of the paragraph.

### 2. It mirrors how search queries are written

When users search, they often paraphrase a sentence from the document they're
looking for. "How does a bigram model work?" is structurally similar to the
opening sentence of a chunk on bigram models.

### 3. It's a known technique

This approach is documented in the **TSDAE** paper (Wang et al., 2021) and
the **GPL** paper (Wang et al., 2022) as "query generation" or "synthetic
query mining."  It's not a hack — it's how people train sentence encoders
when they don't have labeled query-document pairs.

## What We Filter Out

Not every chunk has a usable first sentence. The function
`src.dataset._extract_first_sentence` rejects sentences that:

1. Have **fewer than 8 words** — too short to be a meaningful query.
2. Start with phrases like `Figure`, `Table`, `As we`, `However`, `Note that`,
   `Recall that`, `This`, `That`, `In the`, `In this`, `The`, `Such`.
   These are continuation phrases that don't summarize anything.

**Result:** 127 out of 999 chunks are filtered out, leaving **~872 training pairs**.

```python
# Quick verification you can run locally:
from src.dataset import load_chunks, generate_pairs
chunks = load_chunks('data/processed/jurafsky_chunks.json')
pairs  = generate_pairs(chunks)
print(f"{len(pairs)} pairs from {len(chunks)} chunks")
```

## Negative Sampling Strategy

For each anchor chunk at index `i`, we sample a negative from any chunk `j`
where `|i - j| >= 20`. This **minimum distance** rule matters because:

- Chunks that come right after an anchor often continue the same topic (the
  textbook has ~6 chunks per section). If we picked a chunk 3 positions away
  as the "negative," the model would be punished for correctly recognizing
  related content.
- A distance of 20 corresponds to roughly 5,000 words — enough to be in a
  different section, almost certainly a different topic.

This is called **"in-corpus negative sampling with a separation rule."** It's
a simple form of negative mining and works well for InfoNCE loss because
the rest of the negatives come from the in-batch negatives anyway.

## The Train / Val / Test Split

```
                            999 chunks
                                │
              ┌─────────────────┴─────────────────┐
              │                                   │
         872 chunks                       127 chunks
   (yield training pairs)            (filtered — bad topic sentence)
              │
   ┌──────────┴──────────┐
   │                     │
 783 train             87 val
  pairs                 pairs

                          INDEPENDENTLY:
                          10 hand-crafted queries
                          (evaluation_set.csv)
                          ↳ test set
                          ↳ never seen during training
```

**Important:** The held-out test set is the 10 hand-written queries, NOT a
random slice of the synthetic queries. Why? Because synthetic queries are
literally first sentences of chunks — evaluating on them would be trivially
easy (the model just needs to match identical or near-identical text).

The 10 hand-crafted queries in `evaluation_set.csv` are written in natural
language by a human ("How can we compute the probability of a full sentence
using bigram probabilities?") and are deliberately phrased differently from
the matching chunk's content. This is what makes the test set honest.

## Limitations You Should Acknowledge in the Report

1. **Synthetic queries are too similar to their targets.**
   The training queries are literally extracted from the target chunks. The
   model learns to match "topic sentence → paragraph" which is easier than
   "user question → relevant paragraph." This is exactly why we evaluate on
   hand-crafted queries — the test set is intentionally harder than training.

2. **Random negatives are easy.**
   The negative chunk is a random distant chunk. Most random negatives are
   so obviously different that they don't push the model hard. Modern systems
   use "hard negative mining" (find documents that look similar but aren't
   the answer). We don't, for simplicity. This is a fair area to mention
   under "Future Work" in the report.

3. **Filtered chunks are silently dropped.**
   127 chunks contribute no training signal. The model still has to encode
   them at retrieval time, but it never learned a topic-sentence → paragraph
   mapping for them. In practice, those chunks are mostly figure captions and
   noisy boilerplate, so this is fine — but worth being honest about.

4. **Only English. Only Jurafsky.**
   The model is fine-tuned on NLP textbook prose. It will not generalize
   well to, say, biomedical literature or news articles — though DistilBERT's
   pretraining gives it a decent base.

## Pairs vs Triplets — Why We Generate Both

`src.dataset.generate_pairs` produces records with both a positive AND a
negative. We use this dual format because:

| Loss type | Uses positive? | Uses negative? |
|---|---|---|
| InfoNCELoss | Yes | No (negatives come from other positives in the batch) |
| TripletLoss | Yes | Yes (explicit negative per anchor) |

Our `InBatchDataset` only reads the `positive_text` field; our
`TripletDataset` reads all three. Since we produce them once and save to
disk, both losses can be tried later without re-running data prep.

## Quality Check — How Do We Know the Pairs Are Good?

Notebook `01_data_preparation.ipynb` does three quality checks:

1. **Length distribution.** Plot a histogram of query lengths and document
   lengths. If queries are mostly 4–5 words, the topic sentences aren't very
   informative — that would suggest our chunking is bad.
2. **Sanity check on negatives.** Verify that no pair has `positive_id == negative_id`
   and that `|pos_idx - neg_idx| >= 20` for all pairs.
3. **Manual inspection.** Print 5–10 random pairs and read them. Are the
   queries plausible? Does the positive chunk genuinely answer the query?

This is what you cite in the report when the rubric asks about "pair quality
analysis" (20% of the grade — see grading criteria).
