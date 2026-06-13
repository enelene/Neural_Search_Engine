from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class InfoNCELoss(nn.Module):

    def __init__(self, temperature: float = 0.05) -> None:
        super().__init__()
        self.register_buffer("temperature", torch.tensor(temperature))

    def forward(
        self,
        query_emb: Tensor,
        pos_emb: Tensor,
        neg_emb: Tensor | None = None,
    ) -> Tensor:
        batch_size = query_emb.size(0)

        # S[i, j] = similarity(query_i, positive_j). With L2-normalized inputs the
        # dot product IS the cosine similarity. Divide by the temperature tau.
        sim_pos = torch.matmul(query_emb, pos_emb.T) / self.temperature  # [B, B]
        labels = torch.arange(batch_size, device=query_emb.device)

        if neg_emb is not None:
            # Extra columns: query_i vs every explicit negative. The correct answer
            # is still column i (its own positive), so labels are unchanged.
            sim_neg = torch.matmul(query_emb, neg_emb.T) / self.temperature  # [B, B]
            logits = torch.cat([sim_pos, sim_neg], dim=1)                    # [B, 2B]
        else:
            logits = sim_pos

        # query -> candidate (positives + any negatives)
        loss_q2p = F.cross_entropy(logits, labels)
        # positive -> query (symmetric term, over the positive block only)
        loss_p2q = F.cross_entropy(sim_pos.T, labels)
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
