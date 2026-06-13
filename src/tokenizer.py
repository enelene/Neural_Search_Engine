from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

PAD_TOKEN = "[PAD]"          # id 0 — padding (ignored by the model via the mask)
UNK_TOKEN = "[UNK]"          # id 1 — fallback for an unseen sub-word symbol
SPECIAL_TOKENS = (PAD_TOKEN, UNK_TOKEN)

END_OF_WORD = "</w>"        
_WORD_RE = re.compile(r"[a-z0-9]+|[^a-z0-9\s]", re.UNICODE)


def pretokenize(text: str) -> List[str]:
    
    return _WORD_RE.findall(text.lower())


class BPETokenizer:
    
    def __init__(
        self,
        token2id: Dict[str, int],
        merges: List[Tuple[str, str]],
    ) -> None:
        self.token2id: Dict[str, int] = token2id
        self.id2token: List[str] = [""] * len(token2id)
        for tok, idx in token2id.items():
            self.id2token[idx] = tok

        self.merges: List[Tuple[str, str]] = merges
        # Lower rank = learned earlier = applied first during encoding.
        self.merge_ranks: Dict[Tuple[str, str], int] = {
            pair: rank for rank, pair in enumerate(merges)
        }

        self.pad_id: int = token2id[PAD_TOKEN]
        self.unk_id: int = token2id[UNK_TOKEN]

        # Per-word cache so repeated words are only segmented once.
        self._cache: Dict[str, List[str]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    @classmethod
    def train(
        cls,
        texts: List[str],
        vocab_size: int = 8000,
        min_frequency: int = 2,
        verbose: bool = True,
    ) -> "BPETokenizer":
        word_counter: Counter[str] = Counter()
        for text in texts:
            word_counter.update(pretokenize(text))
        words: List[List[str]] = []
        freqs: List[int] = []
        for word, freq in word_counter.items():
            words.append(list(word) + [END_OF_WORD])
            freqs.append(freq)
        base_vocab = set()
        for symbols in words:
            base_vocab.update(symbols)
        pair_counts: Counter[Tuple[str, str]] = Counter()
        pair_to_words: Dict[Tuple[str, str], set] = defaultdict(set)
        for idx, symbols in enumerate(words):
            f = freqs[idx]
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_counts[pair] += f
                pair_to_words[pair].add(idx)
        num_merges = vocab_size - len(base_vocab) - len(SPECIAL_TOKENS)
        merges: List[Tuple[str, str]] = []

        if verbose:
            print(
                f"[BPE] unique words={len(words)}  base chars={len(base_vocab)}  "
                f"target merges={max(0, num_merges)}"
            )

        for step in range(max(0, num_merges)):
            if not pair_counts:
                break
            best = max(pair_counts, key=pair_counts.__getitem__)
            if pair_counts[best] < min_frequency:
                break

            merges.append(best)
            new_symbol = best[0] + best[1]
            a, b = best

            for idx in list(pair_to_words[best]):
                symbols = words[idx]
                f = freqs[idx]

                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i + 1])
                    pair_counts[pair] -= f
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    holders = pair_to_words.get(pair)
                    if holders is not None:
                        holders.discard(idx)

                merged: List[str] = []
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                        merged.append(new_symbol)
                        i += 2
                    else:
                        merged.append(symbols[i])
                        i += 1
                words[idx] = merged

                for i in range(len(merged) - 1):
                    pair = (merged[i], merged[i + 1])
                    pair_counts[pair] += f
                    pair_to_words[pair].add(idx)

            if verbose and (step + 1) % 1000 == 0:
                print(f"[BPE]   learned {step + 1}/{num_merges} merges")

        token2id: Dict[str, int] = {}
        for tok in SPECIAL_TOKENS:
            token2id[tok] = len(token2id)
        for tok in sorted(base_vocab):
            token2id.setdefault(tok, len(token2id))
        for a, b in merges:
            token2id.setdefault(a + b, len(token2id))

        if verbose:
            print(f"[BPE] done. vocab_size={len(token2id)}  merges={len(merges)}")

        return cls(token2id, merges)

    def _segment_word(self, word: str) -> List[str]:
        """Greedily apply learned merges to a single word -> list of sub-words."""
        cached = self._cache.get(word)
        if cached is not None:
            return cached

        symbols: List[str] = list(word) + [END_OF_WORD]

        while len(symbols) > 1:
            best_rank: Optional[int] = None
            best_i = -1
            for i in range(len(symbols) - 1):
                rank = self.merge_ranks.get((symbols[i], symbols[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_i = i
            if best_i == -1:
                break  
            symbols = (
                symbols[:best_i]
                + [symbols[best_i] + symbols[best_i + 1]]
                + symbols[best_i + 2 :]
            )

        self._cache[word] = symbols
        return symbols

    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        
        ids: List[int] = []
        for word in pretokenize(text):
            for sym in self._segment_word(word):
                ids.append(self.token2id.get(sym, self.unk_id))

        if not ids:
           
            ids = [self.unk_id]

        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def encode_batch(
        self,
        texts: List[str],
        max_length: int = 256,
    ) -> Dict[str, Tensor]:
        
        sequences = [self.encode(t, max_length=max_length) for t in texts]
        length = max((len(s) for s in sequences), default=1)
        length = max(length, 1)

        input_ids = torch.full((len(sequences), length), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), length), dtype=torch.long)
        for i, seq in enumerate(sequences):
            n = len(seq)
            input_ids[i, :n] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, :n] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids: List[int]) -> str:
       
        tokens = [
            self.id2token[i]
            for i in ids
            if 0 <= i < len(self.id2token) and i != self.pad_id
        ]
        return "".join(tokens).replace(END_OF_WORD, " ").strip()

    def save(self, path: str | Path) -> None:
        """Persist the tokenizer to a single JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "token2id": self.token2id,
            "merges": [list(p) for p in self.merges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        print(f"Tokenizer saved -> {path}  (vocab_size={self.vocab_size})")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Load a tokenizer previously written by :meth:`save`."""
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        token2id = {tok: int(idx) for tok, idx in payload["token2id"].items()}
        merges = [tuple(p) for p in payload["merges"]]
        return cls(token2id, merges)

