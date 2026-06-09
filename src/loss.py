"""
src/loss.py — Contrastive loss functions for training the BiEncoder.

Two losses are implemented and explained here:

─────────────────────────────────────────────────────────────────────────────
1.  InfoNCELoss  (recommended — this is what we use for training)
─────────────────────────────────────────────────────────────────────────────
    Also known as NT-Xent, in-batch negatives loss, or SimCLR loss.
    Used by: CLIP, SimCSE, DPR (dense passage retrieval), and most modern
             sentence encoder papers.

    Intuition:
        We have a batch of B (query, positive) pairs. For each query q_i,
        its matching positive is p_i. Every other positive p_j (j ≠ i) in
        the batch is treated as a negative. The loss pushes q_i to be most
        similar to p_i and dissimilar from all p_j (j ≠ i).

    Math:
        sim(u, v) = u · v / τ          (dot product between L2-norm vectors
                                         divided by temperature τ)

        similarity matrix S ∈ R^{B×B}:
            S[i, j] = sim(q_i, p_j)

        Loss = CrossEntropy(S, target=diag)
             = -1/B Σ_i log( exp(S[i,i]) / Σ_j exp(S[i,j]) )

    Why is this better than TripletLoss?
        • With batch_size=32 you get 31 negatives per query for FREE.
          TripletLoss would need 31 separate negative-sampling calls.
        • The gradient signal is richer — the loss cares about the relative
          ranking of ALL pairs in the batch simultaneously.
        • Temperature τ controls the "hardness": small τ → sharper
          distribution → model must discriminate more precisely.

    Temperature advice:
        τ = 0.05 is used by SimCSE. τ = 0.07 by MoCo. Start with 0.07.

─────────────────────────────────────────────────────────────────────────────
2.  TripletLoss  (classic, included for comparison and educational value)
─────────────────────────────────────────────────────────────────────────────
    The original contrastive learning loss.

    Math:
        L = max(0,  margin  −  cos(q, p)  +  cos(q, n))

        Where:
            cos(q, p) = similarity of query and positive  (want this HIGH)
            cos(q, n) = similarity of query and negative  (want this LOW)
            margin    = minimum gap we want between pos/neg similarity

    Intuition:
        "The positive must be at least *margin* closer than the negative.
         If it already is, the loss is zero and we don't update."

    Drawback:
        With random negatives the loss saturates quickly — once all random
        negatives are farther than *margin*, gradients vanish and learning
        stops. Hard negative mining is needed to keep learning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class InfoNCELoss(nn.Module):
    """In-batch negatives contrastive loss (NT-Xent / SimCLR).

    Expects L2-normalized query and positive embeddings from BiEncoder.
    Every non-diagonal element of the similarity matrix is treated as a
    negative sample — no explicit negative sampling is needed.

    Args:
        temperature: Softmax temperature τ. Lower = sharper distribution
                     = harder negatives. Typical range: 0.05 – 0.20.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        # Store as a buffer so it moves to the correct device with .to()
        self.register_buffer("temperature", torch.tensor(temperature))

    def forward(self, query_emb: Tensor, pos_emb: Tensor) -> Tensor:
        """Compute InfoNCE loss over a batch.

        Args:
            query_emb: L2-normalized query embeddings.   Shape [B, D]
            pos_emb:   L2-normalized positive embeddings. Shape [B, D]

        Returns:
            Scalar loss tensor. Call .backward() on this.
        """
        batch_size = query_emb.size(0)

        # Similarity matrix: S[i,j] = cos(query_i, positive_j)
        # Since embeddings are L2-normalized, dot product == cosine similarity
        sim_matrix = torch.matmul(query_emb, pos_emb.T)  # [B, B]
        sim_matrix = sim_matrix / self.temperature

        # Target: each query should match its own positive (the diagonal)
        labels = torch.arange(batch_size, device=query_emb.device)

        # Cross-entropy over the similarity matrix rows
        # This is equivalent to minimising -log(softmax(S)[i, i]) for each i
        loss = F.cross_entropy(sim_matrix, labels)
        return loss


class TripletLoss(nn.Module):
    """Cosine-distance triplet margin loss.

    Requires explicit (anchor, positive, negative) triplets.
    BiEncoder outputs are L2-normalized so cosine similarity = dot product.

    Args:
        margin: Minimum required gap between pos and neg similarity.
                Typical range: 0.2 – 0.5. Larger = stricter constraint.
    """

    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        """Compute triplet margin loss.

        Args:
            anchor:   L2-normalized query embeddings.   Shape [B, D]
            positive: L2-normalized positive embeddings. Shape [B, D]
            negative: L2-normalized negative embeddings. Shape [B, D]

        Returns:
            Scalar mean loss over the batch.
        """
        # cos(anchor, positive) — should be large (close to 1)
        pos_sim = (anchor * positive).sum(dim=1)   # [B]

        # cos(anchor, negative) — should be small (close to -1 or 0)
        neg_sim = (anchor * negative).sum(dim=1)   # [B]

        # max(0, margin - (pos_sim - neg_sim))
        # Loss is 0 when pos is already *margin* more similar than neg
        loss = F.relu(self.margin - (pos_sim - neg_sim))
        return loss.mean()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B, D = 8, 768

    # Fake L2-normalized embeddings (as BiEncoder would produce)
    q = F.normalize(torch.randn(B, D), p=2, dim=1)
    p = F.normalize(torch.randn(B, D), p=2, dim=1)
    n = F.normalize(torch.randn(B, D), p=2, dim=1)

    infonce = InfoNCELoss(temperature=0.07)
    triplet = TripletLoss(margin=0.3)

    loss_info = infonce(q, p)
    loss_trip = triplet(q, p, n)

    print(f"InfoNCE loss  : {loss_info.item():.4f}  (≈ log(B)={torch.log(torch.tensor(float(B))):.2f} for random embs)")
    print(f"Triplet loss  : {loss_trip.item():.4f}  (≈ margin={triplet.margin} for random embs)")

    # Verify gradients flow
    loss_info.backward()
    print("Gradients flow through InfoNCE: ✓")

    loss_trip2 = triplet(q.detach().requires_grad_(True),
                         p.detach().requires_grad_(True),
                         n.detach().requires_grad_(True))
    loss_trip2.backward()
    print("Gradients flow through TripletLoss: ✓")
