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


def generate_pairs(
    chunks: List[Dict],
    min_distance: int = 20,
    seed: int = 42,
) -> List[Dict]:
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
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * (1 - val_size))
    return shuffled[:cut], shuffled[cut:]


def load_chunks(path: str | Path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_pairs(pairs: List[Dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(pairs)} pairs -> {path}")

class InBatchDataset(Dataset):

    def __init__(
        self,
        pairs: List[Dict],
        tokenizer: BPETokenizer,
        query_max_len: int = 64,
        doc_max_len: int = 256,
        use_negatives: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.query_max_len = query_max_len
        self.doc_max_len = doc_max_len
        # Only use explicit negatives if asked AND every pair actually has one.
        self.use_negatives = use_negatives and all("negative_text" in p for p in pairs)

        # Pre-tokenize everything once (id lists, no padding yet).
        self.query_ids: List[List[int]] = [
            tokenizer.encode(p["query"], max_length=query_max_len) for p in pairs
        ]
        self.pos_ids: List[List[int]] = [
            tokenizer.encode(p["positive_text"], max_length=doc_max_len) for p in pairs
        ]
        self.neg_ids: Optional[List[List[int]]] = None
        if self.use_negatives:
            self.neg_ids = [
                tokenizer.encode(p["negative_text"], max_length=doc_max_len) for p in pairs
            ]

    def __len__(self) -> int:
        return len(self.query_ids)

    def __getitem__(self, idx: int):
        if self.neg_ids is not None:
            return self.query_ids[idx], self.pos_ids[idx], self.neg_ids[idx]
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

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Collate (query, positive[, negative]) id-lists into padded batch tensors."""
        if self.neg_ids is not None:
            query_lists, pos_lists, neg_lists = zip(*batch)
        else:
            query_lists, pos_lists = zip(*batch)
            neg_lists = None

        q = self._pad(query_lists)
        p = self._pad(pos_lists)
        out = {
            "query_input_ids": q["input_ids"],
            "query_attention_mask": q["attention_mask"],
            "pos_input_ids": p["input_ids"],
            "pos_attention_mask": p["attention_mask"],
        }
        if neg_lists is not None:
            n = self._pad(neg_lists)
            out["neg_input_ids"] = n["input_ids"]
            out["neg_attention_mask"] = n["attention_mask"]
        return out

