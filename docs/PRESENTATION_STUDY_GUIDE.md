# Presentation Study Guide — Theory + Videos

Four components power this project. For each: a **video** to watch, **where it
lives in our code**, a **60-second explanation**, and **questions the lecturer
might ask** (with answers). Watch the videos at 1.25–1.5× — you don't need every
minute, just the core idea.

---

## 1. BM25 — the baseline (`src/evaluate.py`)

🎥 **Watch**
- A no-nonsense intro (short, intuitive): https://www.youtube.com/watch?v=TW9vHU1GpU4
- University lecture (rigorous, good for talking to a lecturer): https://www.youtube.com/watch?v=p8st3g_Y39I

📍 **In our code:** `evaluate_bm25()` uses `rank_bm25.BM25Okapi` (k1=1.5, b=0.75). It is the **baseline** we compare the neural model against.

🧠 **60-second version:** BM25 is a *bag-of-words* keyword scorer — an improved TF-IDF. Two ideas: **term-frequency saturation** (the 10th occurrence of a word adds little, controlled by `k1`) and **document-length normalization** (long docs don't win just by being long, controlled by `b`). It is fast, needs no training, and is very strong on technical text where the query reuses the document's exact words.

🎤 **Likely questions**
- *Why use BM25 as the baseline?* It's the IR gold standard, zero-shot (no training), and especially strong on technical/keyword queries — a fair, hard baseline.
- *What's its weakness?* Pure lexical matching: it can't match synonyms or paraphrases (no semantics). That gap is exactly what our neural encoder targets.
- *What do `k1` and `b` do?* `k1` = how fast term-frequency saturates; `b` = how much to normalize for document length.

---

## 2. BPE tokenizer — turning text into ids (`src/tokenizer.py`)

🎥 **Watch (this is THE video, he builds BPE from scratch exactly like we did):**
- Andrej Karpathy — "Let's build the GPT Tokenizer": https://www.youtube.com/watch?v=zduSFxRajkE
  (First ~45 min is enough for the BPE algorithm; skip the tiktoken/regex deep end.)

📍 **In our code:** `BPETokenizer.train()` learns merges; `encode()/encode_batch()` turn text into padded id tensors. Specials: `[PAD]=0`, `[UNK]=1`.

🧠 **60-second version:** Start with each word as a sequence of characters + an end-marker `</w>`. Repeatedly find the **most frequent adjacent pair** of symbols across the corpus and merge it into one new symbol; record the merge. After ~8000 merges, frequent words are single tokens (`language`), rare ones split into reusable pieces (`perplex`+`ity`). Encoding replays the learned merges in order.

🎤 **Likely questions**
- *Why BPE instead of splitting on spaces (word-level)?* A word-level vocab explodes in size and turns every rare technical term into one `[UNK]` (no signal). BPE keeps the vocab small and almost never produces `[UNK]`.
- *Why train it on your own corpus and not download one?* Pretrained tokenizers ship with pretrained models, which are prohibited — so we learn the merges ourselves from our data.
- *What is `</w>` for?* It marks word boundaries so the prefix `in` (inside) and the word `in` get different tokens.

---

## 3. The model — a Transformer encoder from scratch (`src/model.py`)

🎥 **Watch**
- Andrej Karpathy — "Let's build GPT: from scratch, in code, spelled out" (he codes self-attention, positional encoding, multi-head, FFN, residuals — every layer we wrote): https://www.youtube.com/watch?v=kCc8FmEb1nY
- 3Blue1Brown — "Attention in transformers, visually explained" (the best *intuition* for self-attention): https://www.youtube.com/watch?v=eMlx5fFNoYc

📍 **In our code:** `BiEncoder` = `TokenEmbedding` (scaled) + `SinusoidalPositionalEncoding` + N × `EncoderLayer` (`MultiHeadSelfAttention` + `FeedForward`, pre-norm residuals) + final LayerNorm + masked mean pooling + L2 normalize.

🧠 **60-second version:** Each token becomes a vector (embedding). We **add positional encoding** because self-attention has no built-in sense of order. **Self-attention** lets every token look at every other token: it builds Query, Key, Value vectors, scores tokens by `softmax(Q·Kᵀ/√d)`, and mixes their Values — so each token's representation becomes context-aware. A **feed-forward** layer then transforms each position. We stack 4 such blocks, **average** the token vectors (mean pooling) into one sentence vector, and **L2-normalize** it. The *same* encoder encodes both the query and the documents — that's the "bi-encoder."

🎤 **Likely questions**
- *Why mean pooling and not a [CLS] token?* `[CLS]` only carries meaning after large-scale pretraining (which we don't have). Averaging all token vectors is the standard, robust way to get a sentence embedding from scratch (cf. Sentence-BERT).
- *Why positional encoding — and why sinusoidal?* Attention is permutation-invariant, so without it "dog bites man" = "man bites dog". Sinusoidal is the original (Vaswani) choice: parameter-free and works for any length. Learned positional embeddings are a valid alternative.
- *Why L2-normalize the output?* With unit vectors, cosine similarity equals the dot product — stable for the loss and fast for retrieval (one matrix multiply).
- *How do you stop attention from looking at padding?* A key-padding mask sets padded positions to −∞ before the softmax, so their attention weight is 0.
- *Why pre-norm (LayerNorm before each sub-layer)?* It trains more stably from random initialization than the original post-norm.

---

## 4. InfoNCE — the contrastive loss that trains it (`src/loss.py`)

🎥 **Watch**
- "Can Contrastive Learning Work? — SimCLR Explained" (explains InfoNCE / NT-Xent and in-batch negatives): https://www.youtube.com/watch?v=7Id8SPH31UE
- (Optional) 3Blue1Brown's videos above also help with the embedding intuition.

📍 **In our code:** `InfoNCELoss` builds a B×B similarity matrix of queries vs positives, divides by temperature τ=0.05, and applies cross-entropy with the **diagonal** as the labels — in **both** directions (symmetric, CLIP-style).

🧠 **60-second version:** Take a batch of B (query, correct-passage) pairs. For each query, its own passage is the positive; **the other B−1 passages in the batch are free negatives**. The loss pushes each query's embedding to be most similar to its own passage and dissimilar from the others. Temperature τ sharpens the distribution. "Symmetric" means we also do it the other way (each passage should pick its own query) and average — it trains more stably.

🎤 **Likely questions**
- *What are in-batch negatives?* The other positives already in the mini-batch, reused as negatives — so a batch of 64 gives 63 negatives per query for free (no extra sampling).
- *Why InfoNCE over triplet loss?* One positive vs many negatives per step = a much richer gradient, and no hard-negative mining needed.
- *What does the temperature do?* Lower τ makes the model more confident/peaked about the correct match; 0.05–0.07 is the standard range (SimCSE/MoCo).
- *How do you know it's learning?* An untrained model can't separate the positive from the negatives, so its loss sits near `log(B)` (=log 64 ≈ 4.16). Watching train/val loss fall well below that line is direct proof of learning.

---

## The one-paragraph story for the presentation

> "We built a neural search engine over the Jurafsky & Martin textbook **entirely
> from scratch** — no pretrained models. We learn our own **BPE** sub-word vocabulary
> from the corpus, feed token ids into a **Transformer encoder** we wrote by hand
> (embeddings, sinusoidal positional encoding, multi-head self-attention, feed-forward,
> residuals), pool to a sentence vector and L2-normalize it. We train it with
> **symmetric InfoNCE** contrastive loss using in-batch negatives, and compare against
> a **BM25** baseline with MRR@10 and Recall@k. BM25 is unbeatable on exact-keyword
> queries; our model's value is on **paraphrased** queries where keywords don't match —
> that's the semantic gap dense retrieval is meant to fill."
