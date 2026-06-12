"""
src/loss.py — InfoNCE contrastive loss for training the BiEncoder.

InfoNCELoss (NT-Xent, in-batch negatives, SimCLR-style).

Used by CLIP, SimCSE, DPR and most modern sentence-encoder papers.

Intuition:
    A batch of B (query, positive) pairs. For each query q_i its matching
    positive is p_i. Every other positive p_j (j != i) in the batch is
    treated as a negative. The loss pushes q_i to be most similar to p_i
    and dissimilar from all p_j (j != i).

Math:
    sim(u, v) = u . v / tau      (dot product of L2-normalized vectors
                                  divided by the temperature tau)

    similarity matrix S in R^{B x B}:
        S[i, j] = sim(q_i, p_j)

    Loss = CrossEntropy(S, target=diag)
         = -1/B sum_i log( exp(S[i,i]) / sum_j exp(S[i,j]) )

SYMMETRIC variant (used here, CLIP-style):
    We compute the loss in BOTH directions and average them:
        - query -> positive  (each query should pick its own positive)
        - positive -> query  (each positive should pick its own query)
    Averaging the two gives a cleaner gradient and trains a from-scratch encoder
    more stably than the one-directional version.

Temperature advice: 0.05 (SimCSE) to 0.07 (MoCo). Default: 0.05 for from-scratch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class InfoNCELoss(nn.Module):

    def __init__(self, temperature: float = 0.05) -> None:
        super().__init__()
        self.register_buffer("temperature", torch.tensor(temperature))

    def forward(self, query_emb: Tensor, pos_emb: Tensor) -> Tensor:
        batch_size = query_emb.size(0)

        # S[i, j] = similarity(query_i, positive_j). With L2-normalized inputs the
        # dot product IS the cosine similarity. Divide by the temperature tau.
        sim_matrix = torch.matmul(query_emb, pos_emb.T) / self.temperature
        labels = torch.arange(batch_size, device=query_emb.device)

        # Symmetric: rows (query -> positive) and columns (positive -> query).
        loss_q2p = F.cross_entropy(sim_matrix, labels)
        loss_p2q = F.cross_entropy(sim_matrix.T, labels)
        return 0.5 * (loss_q2p + loss_p2q)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, D = 8, 768

    q = F.normalize(torch.randn(B, D, requires_grad=True), p=2, dim=1)
    p = F.normalize(torch.randn(B, D, requires_grad=True), p=2, dim=1)

    infonce = InfoNCELoss(temperature=0.07)
    loss = infonce(q, p)
    print(f"InfoNCE loss  : {loss.item():.4f}  (~ log(B)={torch.log(torch.tensor(float(B))):.2f} for random embs)")

    loss.backward()
    print("Gradients flow through InfoNCE: OK")
