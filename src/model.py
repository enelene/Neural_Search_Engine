"""
Bi-Encoder Architecture for Semantic Search.

Encodes text into dense 768-dim L2-normalized embeddings using DistilBERT
as the backbone with mean pooling over non-padding tokens.

Expected inputs (from Data teammate):
    input_ids      : torch.LongTensor  of shape [batch, seq_len]
    attention_mask : torch.LongTensor  of shape [batch, seq_len]  (1=real, 0=pad)

Expected output (to Loss/Math teammate):
    embeddings     : torch.FloatTensor of shape [batch, 768], L2-normalized
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel


def mean_pooling(token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
    """Compute a masked mean over the token dimension.

    Padding tokens (attention_mask == 0) are excluded from the average so they
    do not dilute the sentence representation.  The operation is fully
    differentiable and safe against all-zero masks (clamped denominator).

    Args:
        token_embeddings: Hidden states from the base model.
                          Shape: [batch_size, seq_len, hidden_size]
        attention_mask:   Binary mask where 1 = real token, 0 = padding.
                          Shape: [batch_size, seq_len]

    Returns:
        Tensor of shape [batch_size, hidden_size] — one vector per input.
    """
    # Expand the mask to match the hidden-state dimension so we can broadcast.
    # [batch, seq_len] -> [batch, seq_len, hidden_size]
    expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    # Zero out the embeddings at padding positions, then sum across the sequence.
    sum_embeddings = (token_embeddings * expanded_mask).sum(dim=1)

    # Count real (non-padding) tokens per sample; clamp to avoid division by zero.
    token_counts = expanded_mask.sum(dim=1).clamp(min=1e-9)

    return sum_embeddings / token_counts


class BiEncoder(nn.Module):
    """Bi-Encoder that maps tokenized text to a single dense sentence embedding.

    Architecture:
        DistilBERT backbone  ->  mean pooling  ->  L2 normalization

    The L2 normalization ensures that cosine similarity between two embeddings
    is equivalent to their dot product, which is required by the contrastive /
    triplet loss computed by the Loss/Math teammate.

    Args:
        pretrained_model_name: HuggingFace model hub identifier or local path.
                               Defaults to "distilbert-base-uncased".
    """

    def __init__(self, pretrained_model_name: str = "distilbert-base-uncased") -> None:
        super().__init__()
        # AutoModel returns the bare transformer without a task-specific head,
        # giving us raw token-level hidden states to pool ourselves.
        self.backbone = AutoModel.from_pretrained(pretrained_model_name)
        self.hidden_size: int = self.backbone.config.hidden_size  # 768 for DistilBERT

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Encode a batch of tokenized texts into normalized dense embeddings.

        Args:
            input_ids:      Token id sequences.  Shape: [batch_size, seq_len]
            attention_mask: Padding mask (1=real, 0=pad). Shape: [batch_size, seq_len]

        Returns:
            L2-normalized embeddings. Shape: [batch_size, 768]
        """
        # Pass through DistilBERT; last_hidden_state: [batch, seq_len, 768]
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings: Tensor = outputs.last_hidden_state

        # Aggregate over the sequence dimension, respecting the padding mask.
        pooled: Tensor = mean_pooling(token_embeddings, attention_mask)

        # L2-normalize so ||embedding||₂ = 1, making cosine sim = dot product.
        normalized: Tensor = F.normalize(pooled, p=2, dim=1)

        return normalized


if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    # Smoke test — runs without installing sentence-transformers.          #
    # Verifies output shape [2, 768] and that all embeddings are on the   #
    # unit hypersphere (||v||₂ ≈ 1).                                      #
    # ------------------------------------------------------------------ #
    BATCH_SIZE = 2
    MAX_LENGTH = 16
    EXPECTED_DIM = 768

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running smoke test on: {device}")

    model = BiEncoder().to(device)
    model.eval()

    # Simulate what the Data teammate sends: padded token ids with a mask.
    dummy_input_ids = torch.randint(low=0, high=30522, size=(BATCH_SIZE, MAX_LENGTH), device=device)
    # First sample: all real tokens; second sample: last 4 positions are padding.
    dummy_attention_mask = torch.ones(BATCH_SIZE, MAX_LENGTH, dtype=torch.long, device=device)
    dummy_attention_mask[1, -4:] = 0

    with torch.no_grad():
        embeddings = model(dummy_input_ids, dummy_attention_mask)

    assert embeddings.shape == (BATCH_SIZE, EXPECTED_DIM), (
        f"Shape mismatch: expected ({BATCH_SIZE}, {EXPECTED_DIM}), got {tuple(embeddings.shape)}"
    )

    # Verify L2 norms are 1.0 (within floating-point tolerance).
    norms = embeddings.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones(BATCH_SIZE, device=device), atol=1e-5), (
        f"Embeddings are not unit-normalized. Norms: {norms.tolist()}"
    )

    print(f"Output shape : {tuple(embeddings.shape)}  (expected ({BATCH_SIZE}, {EXPECTED_DIM}))")
    print(f"L2 norms     : {norms.tolist()}  (expected [1.0, 1.0])")
    print("All assertions passed.")
