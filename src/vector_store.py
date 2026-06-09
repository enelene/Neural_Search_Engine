"""
src/vector_store.py — Embedding index for semantic search at retrieval time.

What this module does:
    Given a trained BiEncoder and the 999 Jurafsky chunks, this module:
    1. Encodes every chunk into a 768-dim L2-normalized vector (done ONCE offline).
    2. Stores them as a matrix in memory: corpus_matrix ∈ R^{N × 768}.
    3. At query time: encodes the query → computes cosine similarity against
       all N stored vectors → returns the top-k most similar chunks.

Why not FAISS?
    FAISS is a production-grade approximate nearest-neighbour library.
    For N=999 documents, exact brute-force search takes microseconds on CPU,
    so FAISS adds complexity without benefit.  The dot-product operation
    below is equivalent to FAISS IndexFlatIP on L2-normalized vectors.

Complexity:
    Build:  O(N × D) — one forward pass per chunk.  ~15 s on CPU, ~3 s on GPU.
    Search: O(N × D) — one matmul.  < 1 ms for N=999, D=768 on any hardware.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.model import BiEncoder


class VectorStore:
    """In-memory semantic search index backed by a BiEncoder.

    Usage:
        store = VectorStore(model, tokenizer)
        store.build(chunks)                     # encode all documents once
        results = store.search("language model", top_k=5)

    Args:
        model:      A trained (or pretrained) BiEncoder instance.
        tokenizer:  The HuggingFace tokenizer matching the model's backbone.
        max_length: Token sequence length for encoding — must match training.
        device:     Torch device ("cuda" or "cpu").
        batch_size: Number of documents encoded per forward pass during build.
    """

    def __init__(
        self,
        model: BiEncoder,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 64,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size

        # Populated by build()
        self.corpus_matrix: Optional[Tensor] = None  # [N, 768]
        self.chunk_ids: List[str] = []
        self.chunk_texts: List[str] = []

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _encode_texts(self, texts: List[str]) -> Tensor:
        """Encode a list of strings into L2-normalized embeddings.

        Processes *texts* in mini-batches of ``self.batch_size`` to avoid
        OOM errors on large corpora.

        Args:
            texts: List of raw strings to encode.

        Returns:
            Tensor of shape [len(texts), 768], L2-normalized, on CPU.
        """
        all_embs: List[Tensor] = []

        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start : start + self.batch_size]

            enc = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids      = enc.input_ids.to(self.device)
            attention_mask = enc.attention_mask.to(self.device)

            with torch.no_grad():
                embs = self.model(input_ids, attention_mask)  # [B, 768]

            all_embs.append(embs.cpu())

        return torch.cat(all_embs, dim=0)  # [N, 768]

    # ------------------------------------------------------------------
    # Build index
    # ------------------------------------------------------------------

    def build(self, chunks: List[Dict]) -> None:
        """Encode all chunks and store the embedding matrix.

        This method must be called before ``search``.  It is safe to call
        multiple times (previous index is overwritten).

        Args:
            chunks: List of dicts with at minimum keys: ``id``, ``content``.
        """
        print(f"Building index for {len(chunks)} chunks on {self.device}...")
        texts = [c["content"] for c in chunks]

        self.corpus_matrix = self._encode_texts(texts)  # [N, 768]
        self.chunk_ids = [c["id"] for c in chunks]
        self.chunk_texts = texts

        print(f"  Corpus matrix: {tuple(self.corpus_matrix.shape)}  OK")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve the most semantically similar chunks for a query.

        Args:
            query:  Raw query string (not pre-tokenized).
            top_k:  Number of results to return.

        Returns:
            List of result dicts sorted by descending score:
                id      — chunk identifier
                score   — cosine similarity (between –1 and 1)
                content — full chunk text
                rank    — 1-based rank

        Raises:
            RuntimeError: If ``build()`` has not been called yet.
        """
        if self.corpus_matrix is None:
            raise RuntimeError("Call build() before search().")

        # Encode the query: [1, 768]
        q_emb = self._encode_texts([query])  # [1, 768]

        # Cosine similarity — both sides are L2-normalized, so this is just dot product
        # scores: [N]
        scores = (q_emb @ self.corpus_matrix.T).squeeze(0)  # [N]

        # Get top-k indices sorted by score (descending)
        top_k = min(top_k, len(self.chunk_ids))
        top_indices = torch.topk(scores, k=top_k).indices.tolist()

        results = []
        for rank, idx in enumerate(top_indices, 1):
            results.append(
                {
                    "id": self.chunk_ids[idx],
                    "score": round(scores[idx].item(), 4),
                    "content": self.chunk_texts[idx],
                    "rank": rank,
                }
            )
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the corpus embedding matrix and metadata to disk.

        Args:
            path: Directory where ``embeddings.pt`` and ``metadata.json``
                  will be written.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.corpus_matrix, path / "embeddings.pt")
        meta = {"chunk_ids": self.chunk_ids, "chunk_texts": self.chunk_texts}
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        print(f"Vector store saved -> {path}")

    def load(self, path: str | Path) -> None:
        """Load a previously saved embedding matrix.

        Args:
            path: Directory containing ``embeddings.pt`` and ``metadata.json``.
        """
        path = Path(path)
        self.corpus_matrix = torch.load(path / "embeddings.pt", map_location="cpu")
        with open(path / "metadata.json", encoding="utf-8") as f:
            meta = json.load(f)
        self.chunk_ids = meta["chunk_ids"]
        self.chunk_texts = meta["chunk_texts"]
        print(f"Vector store loaded from {path}  ({len(self.chunk_ids)} chunks)")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    chunks_path = root / "data" / "processed" / "jurafsky_chunks.json"

    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = BiEncoder()

    store = VectorStore(model, tokenizer, batch_size=64)
    store.build(chunks)

    query = "How does a language model assign probability to a sentence?"
    results = store.search(query, top_k=3)

    print(f"\nQuery: {query}\n")
    for r in results:
        print(f"[{r['rank']}] {r['id']}  score={r['score']:.4f}")
        print(f"    {r['content'][:200]}\n")
