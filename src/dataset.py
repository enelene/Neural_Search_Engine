"""
src/dataset.py — Training data generation and PyTorch Dataset classes.

This module answers the question: "What do we train on?"

The Jurafsky textbook has been chunked into 999 passages (~250 words each).
We don't have human-written (query, relevant_chunk) pairs for training
(only 10 such pairs exist in evaluation_set.csv, and those are kept for testing).

Strategy — Synthetic Query Generation:
    For every chunk we extract its first complete sentence as a synthetic query.
    The assumption: a well-written textbook paragraph opens with a topic sentence
    that captures the core claim of that paragraph. This makes it a natural
    "question" whose answer is the full paragraph.

    Example:
        Chunk:   "Language models assign probabilities to sequences of words.
                  They are used in speech recognition, spelling correction ..."
        Query:   "Language models assign probabilities to sequences of words."
        Positive: full chunk text
        Negative: randomly sampled chunk that is far away in the book

Dataset classes:
    InBatchDataset  — returns (query, positive) pairs; negatives come from the
                      batch itself. Used with InfoNCELoss.
    TripletDataset  — returns (query, positive, negative) triplets. Used with
                      TripletLoss.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Sentence starters that suggest the sentence is NOT a self-contained topic
# sentence (cross-references, figure captions, continuation phrases, etc.)
_BAD_STARTS: Tuple[str, ...] = (
    "figure", "table", "as we", "see ", "for example",
    "however", "in addition", "moreover", "note that",
    "recall that", "this ", "that ", "it ", "in the",
    "in this", "the ", "such ",
)


def _extract_first_sentence(text: str, min_words: int = 8) -> Optional[str]:
    """Return the first usable sentence of *text*, or None.

    A sentence is usable when it:
    - contains at least *min_words* words,
    - does not start with a phrase that suggests it's a continuation or
      cross-reference (filtered by ``_BAD_STARTS``).

    Args:
        text:      Full paragraph/chunk text.
        min_words: Minimum word count for a sentence to be considered.

    Returns:
        The first usable sentence, or None if no suitable sentence is found.
    """
    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sent in sentences:
        sent = sent.strip()
        if len(sent.split()) < min_words:
            continue
        lower = sent.lower()
        if any(lower.startswith(bad) for bad in _BAD_STARTS):
            continue
        return sent
    return None


# ---------------------------------------------------------------------------
# Pair / Triplet Generation
# ---------------------------------------------------------------------------

def generate_pairs(
    chunks: List[Dict],
    min_distance: int = 20,
    seed: int = 42,
) -> List[Dict]:
    """Generate (query, positive, negative) training records from textbook chunks.

    Each record contains:
        query        — synthetic question (first usable sentence of the chunk)
        positive_id  — chunk id of the relevant document
        positive_text — full text of the positive chunk
        negative_id  — chunk id of a randomly sampled irrelevant document
        negative_text — full text of the negative chunk

    The negative is sampled from chunks whose index is at least *min_distance*
    away so we don't accidentally sample a semantically adjacent chunk as a
    negative.

    Args:
        chunks:       List of chunk dicts with keys: id, content, word_count.
        min_distance: Minimum index distance between anchor and negative chunk.
        seed:         Random seed for reproducibility.

    Returns:
        List of pair dicts ready to be passed to InBatchDataset or TripletDataset.
    """
    rng = random.Random(seed)
    pairs: List[Dict] = []

    for i, chunk in enumerate(chunks):
        query = _extract_first_sentence(chunk["content"])
        if query is None:
            continue  # skip chunks with no usable topic sentence

        # Build the pool of valid negatives (far enough away in the book)
        neg_pool = [j for j in range(len(chunks)) if abs(j - i) > min_distance]
        if not neg_pool:
            continue

        neg_idx = rng.choice(neg_pool)
        pairs.append(
            {
                "query": query,
                "positive_id": chunk["id"],
                "positive_text": chunk["content"],
                "negative_id": chunks[neg_idx]["id"],
                "negative_text": chunks[neg_idx]["content"],
            }
        )

    return pairs


def split_pairs(
    pairs: List[Dict],
    val_size: float = 0.10,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Shuffle and split pairs into train / val subsets.

    We keep the 10 hand-crafted queries in ``evaluation_set.csv`` as the test
    set and never use them here.  The split here is only for monitoring
    over-fitting during training.

    Args:
        pairs:    All generated pairs.
        val_size: Fraction to use for validation (default 10 %).
        seed:     Random seed.

    Returns:
        (train_pairs, val_pairs) tuple.
    """
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * (1 - val_size))
    return shuffled[:cut], shuffled[cut:]


def load_chunks(path: str | Path) -> List[Dict]:
    """Load the pre-processed chunk JSON produced by ``main.py``.

    Args:
        path: Path to ``jurafsky_chunks.json``.

    Returns:
        List of chunk dicts with keys: id, content, word_count.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_pairs(pairs: List[Dict], path: str | Path) -> None:
    """Persist generated pairs to disk as JSON.

    Args:
        pairs: List of pair dicts.
        path:  Output file path (should end in .json).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(pairs)} pairs → {path}")


# ---------------------------------------------------------------------------
# PyTorch Datasets
# ---------------------------------------------------------------------------

class InBatchDataset(Dataset):
    """Dataset that yields tokenized (query, positive) pairs.

    Used with InfoNCELoss. The loss treats all other positives in the
    same batch as negatives, so no explicit negative is needed here.

    Args:
        pairs:      List of dicts with keys: query, positive_text.
        tokenizer:  A HuggingFace tokenizer (e.g. AutoTokenizer for DistilBERT).
        max_length: Sequence length to pad/truncate to. Must match BiEncoder.
    """

    def __init__(
        self,
        pairs: List[Dict],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
    ) -> None:
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def _encode(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize and pad a single string to fixed length."""
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc.input_ids.squeeze(0),         # [max_length]
            "attention_mask": enc.attention_mask.squeeze(0),  # [max_length]
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single training example.

        Returns:
            Dict with keys:
                query_input_ids      [max_length]
                query_attention_mask [max_length]
                pos_input_ids        [max_length]
                pos_attention_mask   [max_length]
        """
        pair = self.pairs[idx]
        q_enc = self._encode(pair["query"])
        p_enc = self._encode(pair["positive_text"])
        return {
            "query_input_ids": q_enc["input_ids"],
            "query_attention_mask": q_enc["attention_mask"],
            "pos_input_ids": p_enc["input_ids"],
            "pos_attention_mask": p_enc["attention_mask"],
        }


class TripletDataset(Dataset):
    """Dataset that yields tokenized (query, positive, negative) triplets.

    Used with TripletLoss.  Explicit negatives are included per example.

    Args:
        pairs:      List of dicts with keys: query, positive_text, negative_text.
        tokenizer:  A HuggingFace tokenizer.
        max_length: Sequence length to pad/truncate to.
    """

    def __init__(
        self,
        pairs: List[Dict],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
    ) -> None:
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def _encode(self, text: str) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single training example.

        Returns:
            Dict with keys:
                query_input_ids      [max_length]
                query_attention_mask [max_length]
                pos_input_ids        [max_length]
                pos_attention_mask   [max_length]
                neg_input_ids        [max_length]
                neg_attention_mask   [max_length]
        """
        pair = self.pairs[idx]
        q_enc = self._encode(pair["query"])
        p_enc = self._encode(pair["positive_text"])
        n_enc = self._encode(pair["negative_text"])
        return {
            "query_input_ids": q_enc["input_ids"],
            "query_attention_mask": q_enc["attention_mask"],
            "pos_input_ids": p_enc["input_ids"],
            "pos_attention_mask": p_enc["attention_mask"],
            "neg_input_ids": n_enc["input_ids"],
            "neg_attention_mask": n_enc["attention_mask"],
        }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    chunks_path = root / "data" / "processed" / "jurafsky_chunks.json"

    chunks = load_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks")

    pairs = generate_pairs(chunks)
    print(f"Generated {len(pairs)} training pairs")
    print(f"\nSample pair:")
    p = pairs[5]
    print(f"  query        : {p['query'][:100]}")
    print(f"  positive_id  : {p['positive_id']}")
    print(f"  negative_id  : {p['negative_id']}")

    train, val = split_pairs(pairs)
    print(f"\nTrain: {len(train)}  Val: {len(val)}")

    # Test tokenization
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    ds = InBatchDataset(train[:10], tokenizer, max_length=64)
    sample = ds[0]
    for k, v in sample.items():
        print(f"  {k}: {tuple(v.shape)}")
