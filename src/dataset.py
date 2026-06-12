from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from src.tokenizer import BPETokenizer

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
    print(f"Saved {len(pairs)} pairs -> {path}")


# ---------------------------------------------------------------------------
# PyTorch Datasets
# ---------------------------------------------------------------------------

class InBatchDataset(Dataset):
    """Dataset that yields tokenized (query, positive) pairs.

    Used with InfoNCELoss. The loss treats all other positives in the same batch
    as negatives, so no explicit negative is needed here.

    Queries are short and documents are long, so we tokenize them with separate
    length caps (``query_max_len`` vs ``doc_max_len``) to save compute. All pairs
    are tokenized ONCE up front and cached as id lists; batches are then padded
    dynamically to the longest sequence in the batch via :meth:`collate_fn`.

    Args:
        pairs:         List of dicts with keys: query, positive_text.
        tokenizer:     A trained :class:`BPETokenizer`.
        query_max_len: Max token length for queries.
        doc_max_len:   Max token length for positive documents.
    """

    def __init__(
        self,
        pairs: List[Dict],
        tokenizer: BPETokenizer,
        query_max_len: int = 64,
        doc_max_len: int = 256,
    ) -> None:
        self.tokenizer = tokenizer
        self.query_max_len = query_max_len
        self.doc_max_len = doc_max_len

        # Pre-tokenize everything once (id lists, no padding yet).
        self.query_ids: List[List[int]] = [
            tokenizer.encode(p["query"], max_length=query_max_len) for p in pairs
        ]
        self.pos_ids: List[List[int]] = [
            tokenizer.encode(p["positive_text"], max_length=doc_max_len) for p in pairs
        ]

    def __len__(self) -> int:
        return len(self.query_ids)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.query_ids[idx], self.pos_ids[idx]

    def _pad(self, seqs: Tuple[List[int], ...]) -> Dict[str, torch.Tensor]:
        """Pad a list of id-lists to the batch maximum length."""
        length = max(max((len(s) for s in seqs), default=1), 1)
        input_ids = torch.full((len(seqs), length), self.tokenizer.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(seqs), length), dtype=torch.long)
        for i, seq in enumerate(seqs):
            n = len(seq)
            input_ids[i, :n] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, :n] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def collate_fn(self, batch: List[Tuple[List[int], List[int]]]) -> Dict[str, torch.Tensor]:
        """Collate a list of (query_ids, pos_ids) into padded batch tensors."""
        query_lists, pos_lists = zip(*batch)
        q = self._pad(query_lists)
        p = self._pad(pos_lists)
        return {
            "query_input_ids": q["input_ids"],
            "query_attention_mask": q["attention_mask"],
            "pos_input_ids": p["input_ids"],
            "pos_attention_mask": p["attention_mask"],
        }


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    chunks_path = root / "data" / "processed" / "jurafsky_chunks_v2.json"

    chunks = load_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks")

    pairs = generate_pairs(chunks)
    print(f"Generated {len(pairs)} training pairs")
    print("\nSample pair:")
    p = pairs[5]
    print(f"  query        : {p['query'][:100]}")
    print(f"  positive_id  : {p['positive_id']}")
    print(f"  negative_id  : {p['negative_id']}")

    train, val = split_pairs(pairs)
    print(f"\nTrain: {len(train)}  Val: {len(val)}")

    # Test tokenization with a small from-scratch BPE tokenizer.
    corpus = [c["content"] for c in chunks]
    tokenizer = BPETokenizer.train(corpus, vocab_size=2000, verbose=False)
    ds = InBatchDataset(train[:10], tokenizer, query_max_len=64, doc_max_len=128)
    batch = ds.collate_fn([ds[i] for i in range(4)])
    for k, v in batch.items():
        print(f"  {k}: {tuple(v.shape)}")
