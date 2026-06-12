# From-Scratch Model — Architecture Deep Dive

This document explains the two from-scratch components that replaced the
(prohibited) pretrained DistilBERT backbone and WordPiece tokenizer:

1. a **BPE tokenizer** (`src/tokenizer.py`), and
2. a **Transformer text encoder** (`src/model.py`),

trained with the symmetric InfoNCE contrastive loss (`src/loss.py`). Only
`torch` is used — no `transformers`, `tokenizers`, or `sentence-transformers`.

---

## 1. BPE Tokenizer (`src/tokenizer.py`)

### Why BPE?
Word-level vocabularies explode in size on a small technical corpus and turn most
rare terms into a single `[UNK]` (no signal). Byte-Pair-Encoding keeps frequent
words whole (`language`, `model`) and splits rare ones into reusable sub-words
(`perplex` + `ity`), so almost nothing is `[UNK]`.

### Training (Sennrich et al., 2016)
1. **Pre-tokenize**: lowercase; split into alphanumeric runs + single punctuation
   chars; each word becomes a char sequence ending with `</w>`
   (`low → l o w </w>`). The `</w>` marker lets the model distinguish the word
   `in` (`in</w>`) from the prefix `in` inside `inside`.
2. **Merge loop**: repeatedly find the most frequent adjacent symbol pair across
   the corpus, merge it into a new symbol, and record the merge — until the vocab
   reaches `vocab_size` (8000) or no pair repeats `min_frequency` times.
3. **Efficiency**: an inverted index `pair → {word ids}` plus a running
   `pair_counts` Counter means each merge only touches the words that contain the
   merged pair (incremental update), not the whole corpus.

### Encoding
Replay learned merges greedily by rank (earliest-learned = highest priority)
until none apply, then map sub-words to ids (`[PAD]=0`, `[UNK]=1`). No `[CLS]`/
`[SEP]` — masked mean pooling needs only the raw sub-word ids.

`encode_batch` pads dynamically to the longest sequence in the batch (capped at
`max_length`) and returns `input_ids` + `attention_mask` tensors. Persisted as a
single `tokenizer.json` (`token2id` + ordered `merges`).

---

## 2. Transformer Encoder (`src/model.py`)

```
input_ids [B, S]
   → TokenEmbedding · sqrt(d_model)        # [B, S, d_model]
   + SinusoidalPositionalEncoding
   → dropout
   → N × EncoderLayer (pre-norm)           # self-attention + FFN, residuals
   → final LayerNorm
   → masked mean pooling                   # [B, d_model]
   → L2 normalize                          # ||v|| = 1
```

### Components (all hand-built)
- **Token embedding** — `nn.Embedding(vocab, d_model, padding_idx=0)`, scaled by
  `√d_model` (standard Transformer convention).
- **Sinusoidal positional encoding** — fixed (non-learned) buffer giving each
  position a unique signal so attention is order-aware:
  `PE(pos,2i)=sin(pos/10000^(2i/d))`, `PE(pos,2i+1)=cos(pos/10000^(2i/d))`.
- **Multi-head self-attention** — own `q/k/v/out` linears; reshape to heads;
  `softmax(QKᵀ/√d_head) V`; a **key-padding mask** sets padded key positions to
  `-inf` before softmax (no query attends to padding). Implemented directly (not
  `nn.MultiheadAttention`) so the mechanism is fully visible.
- **Feed-forward** — `Linear(d_model, d_ff) → GELU → Linear(d_ff, d_model)`.
- **Pre-norm residual block** —
  `x = x + Attn(LN(x), mask)`; `x = x + FFN(LN(x))`. Pre-norm trains more stably
  from random init than post-norm.
- **Pooling + norm** — masked mean over real tokens, then `F.normalize(p=2)` so
  the dot product equals cosine similarity.

### Default config (~5.2M params)
| vocab | d_model | layers | heads | d_ff | max_len | dropout |
|---|---|---|---|---|---|---|
| 8000 | 256 | 4 | 4 | 1024 | 256 | 0.1 |

`BiEncoder.save` / `BiEncoder.load` store the `EncoderConfig` together with the
weights, so a checkpoint rebuilds its own architecture on load.

---

## 3. Symmetric InfoNCE (`src/loss.py`)

For a batch of `B` L2-normalized `(query, positive)` pairs, build
`S[i,j] = qᵢ · pⱼ / τ` (τ = 0.05). The diagonal is the correct match; every
off-diagonal positive is an in-batch negative. Loss is the average of two
cross-entropies (CLIP-style):

```
L = ½ ( CE(S, diag) + CE(Sᵀ, diag) )       # query→positive  and  positive→query
```

Batch 64 ⇒ 63 free negatives per query, no hard-negative mining needed. An
untrained model can't separate the positive from the negatives, so its loss sits
near `log(B)`; watching the curve fall below that line is direct evidence of
learning.

---

## 4. Why from-scratch performance is modest (and that's expected)
A ~5M-param encoder trained from random init on ~8k (mostly Wikipedia) pairs and
evaluated on 25 Jurafsky-specific queries will not match a pretrained model or
beat BM25 in aggregate. The assignment explicitly prioritizes *understanding,
methodology, and baseline comparison* over raw performance. Success signals here:
(1) train/val InfoNCE loss falling well below `log(B)`; (2) clear improvement over
the untrained (0.0) neural baseline; (3) competitiveness on paraphrase queries,
where BM25's lexical overlap collapses.
