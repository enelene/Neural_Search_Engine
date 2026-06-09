# Bi-Encoder Study Guide
## Your Module, Explained from First Principles

> **You are Elene.** Your job in the pipeline is to turn raw tokenized text into
> dense vectors that a loss function can train against. This document explains
> every line of `src/model.py`, the math behind it, how it connects to your
> teammates, and the hard questions you should be able to answer cold.

---

## Table of Contents
1. [The Big Picture — What Does a Bi-Encoder Actually Do?](#1-the-big-picture)
2. [Key Vocabulary — A Glossary to Internalize](#2-key-vocabulary)
3. [The Math of Mean Pooling, Step by Step](#3-the-math-of-mean-pooling)
4. [Line-by-Line Code Walkthrough](#4-line-by-line-code-walkthrough)
5. [Why L2 Normalization? The Geometry of Similarity](#5-why-l2-normalization)
6. [Your Place in the Pipeline — The Three Contracts](#6-your-place-in-the-pipeline)
7. [Self-Test: Questions You Must Be Able to Answer](#7-self-test-questions)
8. [Questions to Ask Your Teammates](#8-questions-to-ask-your-teammates)
9. [Common Mistakes & Gotchas](#9-common-mistakes--gotchas)

---

## 1. The Big Picture

### What problem are we solving?

A computer cannot compute the "meaning" of a sentence directly. It only understands numbers. The goal of a **semantic search engine** is to map sentences into a **vector space** where:

- Semantically similar sentences are **close together** (small distance).
- Semantically different sentences are **far apart** (large distance).

```
"How does BERT work?"  ──►  [0.12, -0.45, 0.88, ...]   (768 numbers)
"Explain transformer models" ──►  [0.14, -0.43, 0.91, ...]   (very close!)
"What is pasta carbonara?" ──►  [-0.72, 0.31, -0.11, ...]  (far away)
```

### What is a Bi-Encoder specifically?

A **Bi-Encoder** (sometimes called a "dual encoder") runs the **same encoder network independently** on two inputs and compares their output vectors. It is called "bi" because there are conceptually two encoding paths — one for the query, one for the document — even though they share the same weights.

```
                    ┌─────────────────┐
Query ──────────────►  DistilBERT     ├──► mean_pool ──► normalize ──► q_emb [768]
                    └─────────────────┘                                     │
                                                                            ▼
                                                                      Cosine Similarity
                                                                            ▲
                    ┌─────────────────┐                                     │
Document ───────────►  DistilBERT     ├──► mean_pool ──► normalize ──► d_emb [768]
                    └─────────────────┘
                        (same weights)
```

**Contrast with a Cross-Encoder:** A cross-encoder concatenates the query and document *before* feeding them in, which is much more accurate but also much slower (can't pre-compute document vectors). The bi-encoder trades some accuracy for the ability to **pre-compute all document embeddings once** and just do a fast vector lookup at query time.

---

## 2. Key Vocabulary

| Term | Meaning in This Project |
|---|---|
| **Token** | A sub-word piece that the tokenizer breaks text into. "playing" → ["play", "##ing"] |
| **input_ids** | Integer IDs mapping each token to a row in DistilBERT's vocabulary (size 30,522) |
| **attention_mask** | Binary tensor: `1` = real token, `0` = padding. Same shape as `input_ids`. |
| **Padding** | Extra `[PAD]` tokens added to the end of shorter sequences so all sequences in a batch have the same length (256). The `attention_mask` tells the model to ignore these. |
| **Token Embedding / Hidden State** | The 768-dim vector DistilBERT produces for each *individual token* after processing the full context. Shape: `[batch, seq_len, 768]`. |
| **Sentence Embedding** | A *single* 768-dim vector representing the *entire* sentence. This is what your Mean Pooling creates. Shape: `[batch, 768]`. |
| **L2 Normalization** | Scaling a vector so its length (L2 norm) equals exactly 1.0. Projects every embedding onto the unit hypersphere. |
| **Cosine Similarity** | A measure of angle between two vectors, not magnitude. Range: [-1, 1]. Between L2-normalized vectors, this equals the dot product. |
| **Triplet Loss** | Ana's loss function. Takes (anchor, positive, negative) and penalizes the model when the anchor is closer to the negative than the positive. |

---

## 3. The Math of Mean Pooling

### Why not just use the [CLS] token?

BERT and its variants prepend a special `[CLS]` token. In classification tasks, only the `[CLS]` vector is used. For semantic similarity, **research has shown that averaging all real tokens produces better sentence representations** than using `[CLS]` alone. This is the key insight behind your implementation.

### The Problem: Padding Tokens Must Be Ignored

Your sequences are padded to length 256. If you naively average all 256 token embeddings, you are also averaging 200+ padding embeddings, which are meaningless and dilute the real signal.

```
Sentence: "BERT is powerful"  (3 real tokens + 253 padding tokens)

Naive mean: (e_BERT + e_is + e_powerful + e_PAD + e_PAD + ... × 253) / 256
            ─────────────────────────────────────────────────────────────
                          ↑ WRONG — padding dominates!

Correct mean: (e_BERT + e_is + e_powerful) / 3
              ─────────────────────────────────
                          ↑ RIGHT — only real tokens count
```

### Step-by-Step with Real Shapes

Assume: `batch_size=2`, `seq_len=5`, `hidden_size=4` (simplified for illustration).

**Step 0 — Inputs:**
```
token_embeddings shape: [2, 5, 4]
attention_mask   shape: [2, 5]

attention_mask = [[1, 1, 1, 0, 0],   # Sample 0: 3 real tokens, 2 padding
                  [1, 1, 0, 0, 0]]   # Sample 1: 2 real tokens, 3 padding
```

**Step 1 — Expand the mask to match hidden dimension:**
```python
expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
```
```
attention_mask         = [[1, 1, 1, 0, 0],       shape: [2, 5]
                           [1, 1, 0, 0, 0]]

.unsqueeze(-1)         shape becomes: [2, 5, 1]   ← adds a dimension

.expand([2, 5, 4])     shape becomes: [2, 5, 4]   ← copies along the new dim

expanded_mask = [[[1,1,1,1], [1,1,1,1], [1,1,1,1], [0,0,0,0], [0,0,0,0]],
                 [[1,1,1,1], [1,1,1,1], [0,0,0,0], [0,0,0,0], [0,0,0,0]]]
```

**Step 2 — Zero out padding positions and sum:**
```python
sum_embeddings = (token_embeddings * expanded_mask).sum(dim=1)
```
```
Multiplying element-wise sets ALL 4 values of each padding token to 0.
Then we sum along dim=1 (the sequence dimension).
Result shape: [2, 4]  ← one vector per sample, summed over real tokens only
```

**Step 3 — Divide by real token count:**
```python
token_counts = expanded_mask.sum(dim=1).clamp(min=1e-9)
```
```
expanded_mask.sum(dim=1) shape: [2, 4]

For sample 0: sum of mask column = [3, 3, 3, 3]  (3 real tokens)
For sample 1: sum of mask column = [2, 2, 2, 2]  (2 real tokens)

token_counts shape: [2, 4]

mean = sum_embeddings / token_counts  →  shape: [2, 4]
```

> The `.clamp(min=1e-9)` is a safety net. If by some bug a sequence has zero real tokens, dividing by zero would produce `NaN` and crash training silently. The clamp prevents this.

---

## 4. Line-by-Line Code Walkthrough

```python
def mean_pooling(token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
```
**Type hints** tell your teammates (and your future self) exactly what goes in and what comes out. `Tensor` is imported from `torch`.

---

```python
    expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
```
- `.unsqueeze(-1)` adds a new last dimension: `[B, S]` → `[B, S, 1]`
- `.expand(...)` repeats it along the new axis without allocating new memory: `[B, S, 1]` → `[B, S, 768]`
- `.float()` converts from integer (0/1) to float so multiplication works in the next line

---

```python
    sum_embeddings = (token_embeddings * expanded_mask).sum(dim=1)
```
- Element-wise multiplication zeros out the embeddings at all padding positions
- `.sum(dim=1)` collapses the sequence dimension: `[B, S, 768]` → `[B, 768]`

---

```python
    token_counts = expanded_mask.sum(dim=1).clamp(min=1e-9)
```
- Counts how many real tokens each sample has, broadcast to shape `[B, 768]`
- `.clamp(min=1e-9)` prevents division by zero (the denominator safety net)

---

```python
class BiEncoder(nn.Module):
    def __init__(self, pretrained_model_name: str = "distilbert-base-uncased") -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(pretrained_model_name)
```
- `AutoModel` (not `AutoModelForSequenceClassification`) loads DistilBERT **without** a classification head. This is deliberate — we want raw hidden states, not class probabilities.
- `super().__init__()` must be called first in any `nn.Module` subclass. It registers the parameters so PyTorch can find them for `.parameters()`, `.to(device)`, `.state_dict()`, etc.

---

```python
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings: Tensor = outputs.last_hidden_state
```
- `outputs` is a `BaseModelOutput` object. It has `.last_hidden_state` (shape `[B, S, 768]`), `.hidden_states` (all layers), etc.
- We only need `last_hidden_state` — the final layer's representation of each token.

---

```python
        normalized: Tensor = F.normalize(pooled, p=2, dim=1)
```
- `p=2` means L2 norm (Euclidean length).
- `dim=1` means normalize along the embedding dimension (each 768-dim vector independently).
- After this operation: `‖normalized[i]‖₂ = 1.0` for every sample `i`.

---

## 5. Why L2 Normalization?

### The Geometry

When you normalize every vector to have unit length, all embeddings lie on the surface of a **unit hypersphere** in 768-dimensional space.

```
           Without normalization              After L2 normalization
           (vectors have different           (all vectors on the unit circle)
            lengths/magnitudes)

           ↑                                         ↑
     B •   |  • A                               B •  |  • A
           |                                    ←────┼────→
           ←─────────────────→                       |
                                                     ↓
```

### The Key Equation

For **unit-normalized** vectors `u` and `v`:

```
cosine_similarity(u, v) = (u · v) / (‖u‖ · ‖v‖)
                        = (u · v) / (1.0 · 1.0)
                        = u · v   ← just the dot product!
```

**This matters because:**
- Ana's TripletLoss (and most loss functions in metric learning) uses **cosine similarity** or **dot product**.
- By normalizing, we guarantee these are equivalent.
- At retrieval time, comparing a query embedding against millions of document embeddings becomes a matrix multiplication — the fastest possible operation on a GPU.

### What Happens Without It?

Without normalization, two sentences that are semantically identical but one is longer (thus has a larger token average magnitude) would appear "different" by distance metrics. Normalization removes this confounding magnitude factor.

---

## 6. Your Place in the Pipeline — The Three Contracts

```
  TURA (Data)          ELENE (You)              ANA (Loss)
  ──────────           ──────────               ──────────
  Dataset              src/model.py             Loss Function
  DataLoader
       │
       │  Contract 1: What Tura gives you
       │  ─────────────────────────────────────────────────
       │  {
       │    "query_input_ids":      Tensor[batch, 256]  (LongTensor)
       │    "query_attention_mask": Tensor[batch, 256]  (LongTensor)
       │    "pos_input_ids":        Tensor[batch, 256]
       │    "pos_attention_mask":   Tensor[batch, 256]
       │    "neg_input_ids":        Tensor[batch, 256]
       │    "neg_attention_mask":   Tensor[batch, 256]
       │  }
       ▼
  BiEncoder.forward()
       │
       │  Contract 2: What you give Ana
       │  ─────────────────────────────────────────────────
       │  query_embeddings: Tensor[batch, 768]  (normalized)
       │  pos_embeddings:   Tensor[batch, 768]  (normalized)
       │  neg_embeddings:   Tensor[batch, 768]  (normalized)
       ▼
  TripletLoss / ContrastiveLoss
       │
       │  Contract 3: What Ana gives PyTorch
       │  ─────────────────────────────────────────────────
       │  loss: Tensor(scalar)  ← e.g., tensor(0.845)
       ▼
  loss.backward()   ← gradients flow back through Ana → Elene → Tura
```

### Your Training Loop (Week 4 context)

This is how your module will be used in the full training loop:

```python
# Pseudocode — what the training loop looks like
for batch in dataloader:          # Tura's output
    # Your module runs three times — once per triplet element
    q_emb = model(batch["query_input_ids"], batch["query_attention_mask"])
    p_emb = model(batch["pos_input_ids"],   batch["pos_attention_mask"])
    n_emb = model(batch["neg_input_ids"],   batch["neg_attention_mask"])

    loss = triplet_loss(q_emb, p_emb, n_emb)   # Ana's module
    loss.backward()                              # PyTorch computes gradients
    optimizer.step()                             # Update DistilBERT weights
    optimizer.zero_grad()
```

Notice: the **same model** (`BiEncoder`) is called three times. This is why it's called a Bi-Encoder — the weights are shared across the query and document encoders.

---

## 7. Self-Test Questions

These are the questions a professor, teammate, or technical interviewer would ask. Cover each before your sync call.

### On Mean Pooling

**Q1: Why can't you just average all 256 token embeddings naively?**
> Because padding tokens are meaningless filler. Averaging them in dilutes the real signal from actual words. A 3-word sentence padded to 256 would have 253/256 ≈ 98.8% of its average come from padding noise.

**Q2: What would happen to the gradient if `token_counts` was 0 and you didn't clamp?**
> Division by zero produces `NaN`. A `NaN` loss means `loss.backward()` propagates `NaN` gradients everywhere, silently corrupting all weights. The model appears to train but every parameter becomes `NaN`. This is one of the hardest bugs to debug in deep learning.

**Q3: What does `.expand()` do and why use it instead of `.repeat()`?**
> `.expand()` creates a **view** with a new stride — no new memory is allocated. `.repeat()` physically copies the data. For a mask expanded from `[B, S, 1]` to `[B, S, 768]`, `.repeat()` would use 768× more memory. `.expand()` is free.

**Q4: Why do we call `.float()` on the mask?**
> `attention_mask` is a `LongTensor` (integer type). You cannot multiply a `FloatTensor` (the token embeddings) by a `LongTensor` in PyTorch — types must match. `.float()` converts to `float32`.

### On the Architecture

**Q5: Why `AutoModel` and not `AutoModelForSequenceClassification`?**
> `AutoModelForSequenceClassification` adds a linear classification head on top of `[CLS]`. We don't want that — we want raw token-level hidden states to pool ourselves. `AutoModel` gives us the bare transformer with no head.

**Q6: What is `last_hidden_state`? How many layers does DistilBERT have?**
> DistilBERT has **6 transformer layers** (half of BERT-base's 12). `last_hidden_state` is the output of the 6th (final) layer — the richest contextual representation of each token. Shape: `[batch, seq_len, 768]`.

**Q7: How many parameters does DistilBERT have?**
> ~66 million parameters (vs. BERT-base's ~110M). It was trained by distilling BERT — learning to mimic BERT's outputs while being 40% smaller and 60% faster.

**Q8: Why do we call `model.eval()` in the smoke test?**
> DistilBERT contains dropout layers (used during training to prevent overfitting). In `eval()` mode, dropout is disabled. For inference and testing, you want deterministic, reproducible outputs. `torch.no_grad()` additionally disables gradient tracking to save memory.

### On L2 Normalization

**Q9: After normalization, what is the range of cosine similarity between two embeddings?**
> [-1, 1]. A value of 1 means identical direction (semantically identical), 0 means orthogonal (unrelated), -1 means opposite direction (semantically opposite).

**Q10: If you forgot to add `F.normalize`, what would break specifically in Ana's loss function?**
> Triplet loss with cosine similarity would still technically run, but the loss landscape would be wrong — the model could "cheat" by making positive embeddings very large in magnitude instead of semantically aligned. Training would be unstable, and retrieved results would be distance-by-magnitude, not distance-by-meaning.

**Q11: Can you think of a case where your output shape would NOT be `[batch, 768]`?**
> If the base model's hidden size is changed (e.g., a different model with `hidden_size=512`). The `768` is hardcoded into the assert, which is correct for DistilBERT but would catch any accidental model swap.

---

## 8. Questions to Ask Your Teammates

### Questions for Tura (DataLoader)

These probe whether the contract between Tura's output and your input is solid.

1. **"What dtype are the tensors coming out of your DataLoader — LongTensor or FloatTensor?"**
   *(They must be `torch.long` / `LongTensor`. `input_ids` are integer indices.)*

2. **"What value do you use for padding tokens in `input_ids`? Is it guaranteed to be 0?"**
   *(The tokenizer's `pad_token_id` for DistilBERT is 0. The `attention_mask` is what we use in our model — but it's good to know both are consistent.)*

3. **"How exactly do you handle sequences longer than 256 tokens? Truncation from the right?"**
   *(The standard is to truncate from the right — drop the tail. This is fine for short queries but could drop important info in long documents.)*

4. **"When you form a triplet `(Q, D+, D-)`, how do you select the negative `D-`? Is it random from the batch, or hardest negative mining?"**
   *(Random negatives are easiest to implement. Hard negative mining — picking the document most similar to the query but wrong — trains faster but is harder to implement.)*

5. **"Are all three elements (query, positive, negative) tokenized with the same tokenizer instance and the same `max_length=256`?"**
   *(They must be. If there's any inconsistency — e.g., different truncation — the tensor shapes will mismatch.)*

6. **"What does one batch dictionary look like? Can you print a single batch and show me the keys and shapes?"**
   *(This is your mid-week integration test. You cannot feed it into BiEncoder without knowing the exact keys.)*

### Questions for Ana (Loss Function)

These probe whether the contract between your output and Ana's loss function is solid.

1. **"Are you using `TripletMarginLoss` from PyTorch directly, or implementing it from scratch?"**
   *(`torch.nn.TripletMarginLoss` expects inputs of shape `[batch, embed_dim]` — matches our output. Custom implementations might expect different formats.)*

2. **"What distance metric are you using — cosine or Euclidean? And what's the margin value?"**
   *(Since our embeddings are L2-normalized, cosine similarity = dot product. The margin `α` in triplet loss is typically 0.2–0.5. If she uses Euclidean distance, we should discuss whether normalization is still the right choice.)*

3. **"Does your loss function expect normalized inputs? Would it break if I passed un-normalized embeddings?"**
   *(If she assumes cosine similarity is just the dot product, she's implicitly assuming L2 normalization. This is worth making explicit.)*

4. **"What scalar value should we expect from the loss early in training — near 0 or near the margin value?"**
   *(Early in training when the model is random, the loss should be approximately equal to the margin `α` (the model is no better than random). If the loss is 0 from the start, something is wrong.)*

5. **"Can you show me the loss going to zero on fake data where positives are identical to queries and negatives are random noise?"**
   *(This is the integration test for Ana's module — just as your smoke test validates your module.)*

---

## 9. Common Mistakes & Gotchas

### Mistake 1: Using `[CLS]` token instead of mean pooling
```python
# WRONG
sentence_embedding = outputs.last_hidden_state[:, 0, :]  # just takes token 0

# RIGHT
sentence_embedding = mean_pooling(outputs.last_hidden_state, attention_mask)
```
`[CLS]` is fine for classification but empirically worse for semantic similarity.

### Mistake 2: Averaging before masking
```python
# WRONG — averages padding tokens too
sentence_embedding = outputs.last_hidden_state.mean(dim=1)

# RIGHT — masks first, then averages
sentence_embedding = mean_pooling(outputs.last_hidden_state, attention_mask)
```

### Mistake 3: Forgetting `model.eval()` and `torch.no_grad()` at inference time
```python
# In training: just call forward normally (dropout active)
loss = compute_loss(model(ids, mask), ...)

# At inference / retrieval time:
model.eval()
with torch.no_grad():
    embedding = model(ids, mask)
```
Without `no_grad()`, PyTorch builds a computation graph for every forward pass, consuming memory and slowing down retrieval.

### Mistake 4: Device mismatch
```python
model = BiEncoder().to("cuda")
# If Tura's DataLoader doesn't send tensors to GPU:
ids = ids.to("cuda")   # You need to explicitly move tensors to the same device
mask = mask.to("cuda")
```

### Mistake 5: Checking shape with `assert` but not norm
Your smoke test checks both. This is good practice — always verify the norm is 1.0, not just the shape.

---

## Quick Reference Card

```
INPUT  (from Tura):  input_ids [B, 256], attention_mask [B, 256]  — LongTensor
                         │
                    DistilBERT backbone
                         │
             last_hidden_state [B, 256, 768]
                         │
                    mean_pooling()
                    - expand mask to [B, 256, 768]
                    - zero out padding positions
                    - sum over seq dim → [B, 768]
                    - divide by real token count → [B, 768]
                         │
                    F.normalize(p=2, dim=1)
                    - makes ‖v‖₂ = 1.0
                         │
OUTPUT (to Ana):  embeddings [B, 768]  — FloatTensor, L2-normalized
```

---

*This guide covers `src/model.py` as of Week 3. Update this document if the architecture changes.*
