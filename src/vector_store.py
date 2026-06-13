from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from src.model import BiEncoder
from src.tokenizer import BPETokenizer


class VectorStore:

    def __init__(
        self,
        model: BiEncoder,
        tokenizer: BPETokenizer,
        max_length: int = 256,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 64,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size

        self.corpus_matrix: Optional[Tensor] = None  # [N, d_model]
        self.chunk_ids: List[str] = []
        self.chunk_texts: List[str] = []

    def _encode_texts(self, texts: List[str]) -> Tensor:
        all_embs: List[Tensor] = []

        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start : start + self.batch_size]

            enc = self.tokenizer.encode_batch(batch_texts, max_length=self.max_length)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            with torch.no_grad():
                embs = self.model(input_ids, attention_mask)

            all_embs.append(embs.cpu())

        return torch.cat(all_embs, dim=0)  # [N, d_model]

    def build(self, chunks: List[Dict]) -> None:
        
        print(f"Building index for {len(chunks)} chunks on {self.device}...")
        texts = [c["content"] for c in chunks]

        self.corpus_matrix = self._encode_texts(texts)  # [N, d_model]
        self.chunk_ids = [c["id"] for c in chunks]
        self.chunk_texts = texts

        print(f"  Corpus matrix: {tuple(self.corpus_matrix.shape)}  OK")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        
        if self.corpus_matrix is None:
            raise RuntimeError("Call build() before search().")

        q_emb = self._encode_texts([query])  # [1, d_model]

        scores = (q_emb @ self.corpus_matrix.T).squeeze(0)  # [N]

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

    def save(self, path: str | Path) -> None:
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.corpus_matrix, path / "embeddings.pt")
        meta = {"chunk_ids": self.chunk_ids, "chunk_texts": self.chunk_texts}
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        print(f"Vector store saved -> {path}")

    def load(self, path: str | Path) -> None:
        
        path = Path(path)
        self.corpus_matrix = torch.load(path / "embeddings.pt", map_location="cpu")
        with open(path / "metadata.json", encoding="utf-8") as f:
            meta = json.load(f)
        self.chunk_ids = meta["chunk_ids"]
        self.chunk_texts = meta["chunk_texts"]
        print(f"Vector store loaded from {path}  ({len(self.chunk_ids)} chunks)")
