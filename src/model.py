"""
Bi-Encoder Architecture for Semantic Search.

Encodes text into dense 768-dim L2-normalized embeddings using DistilBERT
as the backbone with mean pooling over non-padding tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel


def mean_pooling(token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:

    expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    sum_embeddings = (token_embeddings * expanded_mask).sum(dim=1)
    token_counts = expanded_mask.sum(dim=1).clamp(min=1e-9)

    return sum_embeddings / token_counts


class BiEncoder(nn.Module):
    def __init__(self, pretrained_model_name: str = "distilbert-base-uncased") -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(pretrained_model_name)
        self.hidden_size: int = self.backbone.config.hidden_size  # 768 for DistilBERT

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings: Tensor = outputs.last_hidden_state

        pooled: Tensor = mean_pooling(token_embeddings, attention_mask)

        normalized: Tensor = F.normalize(pooled, p=2, dim=1)

        return normalized


if __name__ == "__main__":
    BATCH_SIZE = 2
    MAX_LENGTH = 16
    EXPECTED_DIM = 768

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running smoke test on: {device}")

    model = BiEncoder().to(device)
    model.eval()

    dummy_input_ids = torch.randint(low=0, high=30522, size=(BATCH_SIZE, MAX_LENGTH), device=device)
    dummy_attention_mask = torch.ones(BATCH_SIZE, MAX_LENGTH, dtype=torch.long, device=device)
    dummy_attention_mask[1, -4:] = 0

    with torch.no_grad():
        embeddings = model(dummy_input_ids, dummy_attention_mask)

    assert embeddings.shape == (BATCH_SIZE, EXPECTED_DIM), (
        f"Shape mismatch: expected ({BATCH_SIZE}, {EXPECTED_DIM}), got {tuple(embeddings.shape)}"
    )

    norms = embeddings.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones(BATCH_SIZE, device=device), atol=1e-5), (
        f"Embeddings are not unit-normalized. Norms: {norms.tolist()}"
    )

    print(f"Output shape : {tuple(embeddings.shape)}  (expected ({BATCH_SIZE}, {EXPECTED_DIM}))")
    print(f"L2 norms     : {norms.tolist()}  (expected [1.0, 1.0])")
    print("All assertions passed.")
