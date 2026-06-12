"""
src/tokenizer.py — a Byte-Pair-Encoding (BPE) tokenizer implemented from scratch.

WHY FROM SCRATCH?
    The project rules forbid pretrained / fine-tuned models. The DistilBERT
    WordPiece tokenizer we used before ships *with* a pretrained model, so it is
    not allowed either. This module learns its own sub-word vocabulary directly
    from our corpus using the classic BPE merge algorithm (Sennrich et al., 2016,
    "Neural Machine Translation of Rare Words with Subword Units"). No external
    tokenizer library is used — only the Python standard library + torch (for the
    final tensor packing).

WHY BPE (and not word-level)?
    Our corpus is technical (textbook + Wikipedia) and small. A pure word-level
    vocabulary would explode in size and leave many rare technical terms as a
    single [UNK] token (no signal for the model). BPE strikes a balance: frequent
    words stay whole ("language", "model"), while rare words are decomposed into
    reusable sub-word pieces ("perplex" + "ity"), so almost nothing is [UNK].

THE ALGORITHM (training)
    1. Pre-tokenize text into "words" (alphanumeric runs and single punctuation
       chars), lowercased. Each word is written as a sequence of characters with a
       special end-of-word marker ``</w>`` appended, e.g.  low -> (l, o, w, </w>).
    2. Repeatedly find the most frequent adjacent symbol pair across the whole
       corpus and merge it into a new symbol. Record the merge.
    3. Stop once the vocabulary reaches ``vocab_size`` (or no pair repeats enough).

    Encoding a new word replays the learned merges (lowest rank first) until no
    more apply, then maps the resulting sub-word symbols to integer ids.

The training loop below uses an *incremental* pair-count update (only the words
that actually contain the merged pair are touched each step) so that learning an
8k vocabulary over the full corpus stays fast.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAD_TOKEN = "[PAD]"          # id 0 — padding (ignored by the model via the mask)
UNK_TOKEN = "[UNK]"          # id 1 — fallback for an unseen sub-word symbol
SPECIAL_TOKENS = (PAD_TOKEN, UNK_TOKEN)

END_OF_WORD = "</w>"         # marks the end of a word so "in" (inside) and
                             # "in</w>" (the word "in") get different tokens.

# A "word" is either a run of letters/digits OR a single non-space, non-alnum
# character (so punctuation becomes its own token instead of gluing to words).
_WORD_RE = re.compile(r"[a-z0-9]+|[^a-z0-9\s]", re.UNICODE)


def pretokenize(text: str) -> List[str]:
    """Split raw text into lowercased pre-tokens (words + punctuation).

    Example:
        "Bigram models, e.g. P(w)." -> ['bigram', 'models', ',', 'e', '.', 'g',
                                        '.', 'p', '(', 'w', ')', '.']
    """
    return _WORD_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class BPETokenizer:
    """A self-contained Byte-Pair-Encoding tokenizer.

    Attributes:
        token2id:    Mapping from sub-word string -> integer id.
        id2token:    Inverse mapping (list indexed by id).
        merges:      Ordered list of learned (a, b) merges. Order == priority.
        merge_ranks: Mapping (a, b) -> rank (0 = highest priority merge).
    """

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

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @classmethod
    def train(
        cls,
        texts: List[str],
        vocab_size: int = 8000,
        min_frequency: int = 2,
        verbose: bool = True,
    ) -> "BPETokenizer":
        """Learn a BPE vocabulary of ~``vocab_size`` tokens from ``texts``.

        Args:
            texts:         Iterable of raw strings (the more, the better coverage).
            vocab_size:    Target vocabulary size including the base characters
                           and the 2 special tokens.
            min_frequency: Stop merging once the best pair occurs fewer than this
                           many times (prevents over-fitting to typos / rare junk).
            verbose:       Print progress.

        Returns:
            A trained ``BPETokenizer``.
        """
        # --- 1. Count word frequencies (work on UNIQUE words, not raw corpus) ---
        word_counter: Counter[str] = Counter()
        for text in texts:
            word_counter.update(pretokenize(text))

        # Each unique word -> list of symbols (chars + end-of-word marker).
        words: List[List[str]] = []
        freqs: List[int] = []
        for word, freq in word_counter.items():
            words.append(list(word) + [END_OF_WORD])
            freqs.append(freq)

        # --- 2. Base vocabulary = every individual character seen ---
        base_vocab = set()
        for symbols in words:
            base_vocab.update(symbols)

        # --- 3. Build pair statistics with an inverted index for fast updates ---
        pair_counts: Counter[Tuple[str, str]] = Counter()
        pair_to_words: Dict[Tuple[str, str], set] = defaultdict(set)
        for idx, symbols in enumerate(words):
            f = freqs[idx]
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_counts[pair] += f
                pair_to_words[pair].add(idx)

        # How many merges we are allowed to learn.
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
            # Most frequent adjacent pair this step.
            best = max(pair_counts, key=pair_counts.__getitem__)
            if pair_counts[best] < min_frequency:
                break

            merges.append(best)
            new_symbol = best[0] + best[1]
            a, b = best

            # Only the words that contain ``best`` change.
            for idx in list(pair_to_words[best]):
                symbols = words[idx]
                f = freqs[idx]

                # Remove this word's OLD pair contributions from the global stats.
                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i + 1])
                    pair_counts[pair] -= f
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    holders = pair_to_words.get(pair)
                    if holders is not None:
                        holders.discard(idx)

                # Merge every occurrence of (a, b) inside the word.
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

                # Add this word's NEW pair contributions back in.
                for i in range(len(merged) - 1):
                    pair = (merged[i], merged[i + 1])
                    pair_counts[pair] += f
                    pair_to_words[pair].add(idx)

            if verbose and (step + 1) % 1000 == 0:
                print(f"[BPE]   learned {step + 1}/{num_merges} merges")

        # --- 4. Assemble the final token -> id table ---
        # Specials first (so [PAD]=0, [UNK]=1), then base chars, then merges.
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

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _segment_word(self, word: str) -> List[str]:
        """Greedily apply learned merges to a single word -> list of sub-words."""
        cached = self._cache.get(word)
        if cached is not None:
            return cached

        symbols: List[str] = list(word) + [END_OF_WORD]

        # Repeatedly merge the highest-priority (lowest-rank) adjacent pair.
        while len(symbols) > 1:
            best_rank: Optional[int] = None
            best_i = -1
            for i in range(len(symbols) - 1):
                rank = self.merge_ranks.get((symbols[i], symbols[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_i = i
            if best_i == -1:
                break  # no learned merge applies anymore
            symbols = (
                symbols[:best_i]
                + [symbols[best_i] + symbols[best_i + 1]]
                + symbols[best_i + 2 :]
            )

        self._cache[word] = symbols
        return symbols

    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """Tokenize ``text`` into a list of integer ids.

        No [CLS]/[SEP] tokens are added — the encoder uses masked mean pooling,
        so the raw sub-word ids are all it needs.
        """
        ids: List[int] = []
        for word in pretokenize(text):
            for sym in self._segment_word(word):
                ids.append(self.token2id.get(sym, self.unk_id))

        if not ids:
            # Degenerate empty input (e.g. text was only whitespace): keep one
            # real token so the attention mask is never all-zero (avoids NaNs).
            ids = [self.unk_id]

        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def encode_batch(
        self,
        texts: List[str],
        max_length: int = 256,
    ) -> Dict[str, Tensor]:
        """Tokenize and pad a batch of texts.

        Pads to the longest sequence in the batch (capped at ``max_length``) —
        dynamic padding keeps compute down vs. always padding to ``max_length``.

        Returns:
            dict with:
                input_ids      LongTensor [B, L]
                attention_mask LongTensor [B, L]  (1 = real token, 0 = padding)
        """
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
        """Reconstruct (approximately) the text from ids. Used only for sanity."""
        tokens = [
            self.id2token[i]
            for i in ids
            if 0 <= i < len(self.id2token) and i != self.pad_id
        ]
        return "".join(tokens).replace(END_OF_WORD, " ").strip()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

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


if __name__ == "__main__":
    # Smoke test: train a tiny tokenizer and round-trip a few strings.
    sample_corpus = [
        "A bigram language model assigns a probability to a word given the previous word.",
        "Neural networks learn dense vector representations called word embeddings.",
        "Perplexity measures how well a probability model predicts a held-out sample.",
        "The attention mechanism computes a weighted sum of value vectors.",
    ] * 50

    tok = BPETokenizer.train(sample_corpus, vocab_size=300, verbose=True)

    for text in ["bigram language model", "perplexity of the model", "attention!"]:
        ids = tok.encode(text, max_length=32)
        pieces = [tok.id2token[i] for i in ids]
        print(f"\ntext   : {text}")
        print(f"ids    : {ids}")
        print(f"pieces : {pieces}")
        print(f"decoded: {tok.decode(ids)!r}")

    batch = tok.encode_batch(["bigram model", "attention is all you need"], max_length=32)
    print("\nbatch input_ids shape     :", tuple(batch["input_ids"].shape))
    print("batch attention_mask shape:", tuple(batch["attention_mask"].shape))
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    print("\nAll tokenizer smoke checks passed.")
