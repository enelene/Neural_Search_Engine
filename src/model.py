"""
src/model.py — a Transformer text encoder built FROM SCRATCH.

The project rules forbid pretrained / fine-tuned models, so the previous
DistilBERT backbone is gone. This file implements every layer by hand using only
``torch.nn`` primitives (``Linear``, ``Embedding``, ``LayerNorm``, ``Dropout``):

    input_ids [B, S]
        |
        v
    TokenEmbedding            (learned, scaled by sqrt(d_model))
        + SinusoidalPositionalEncoding
        |
        v
    N x EncoderLayer          (pre-norm: MultiHeadSelfAttention -> FFN, residuals)
        |
        v
    final LayerNorm
        |
        v
    masked mean pooling       (average over real tokens only)
        |
        v
    L2 normalize  ->  sentence embedding [B, d_model], ||v|| = 1

The same encoder ("tower") is used for both queries and documents — this is the
classic *bi-encoder* setup, which is why the public class is still ``BiEncoder``.

NOTE on attention: we deliberately implement multi-head self-attention ourselves
(not ``nn.MultiheadAttention`` / ``nn.TransformerEncoderLayer``) so the mechanism
is fully visible and clearly "from scratch".
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EncoderConfig:
    """Hyperparameters that define the encoder architecture.

    The defaults give a ~5M-parameter model that trains comfortably on a single
    T4 GPU and is large enough to learn useful sentence representations from our
    ~8k contrastive pairs without over-fitting.
    """
    vocab_size: int                 # set from the trained tokenizer
    d_model: int = 256              # embedding / hidden size
    n_layers: int = 4               # number of Transformer encoder blocks
    n_heads: int = 4                # attention heads (d_model must divide by this)
    d_ff: int = 1024                # feed-forward inner size (usually 4 * d_model)
    max_len: int = 256              # maximum sequence length (for positional table)
    dropout: float = 0.1
    pad_id: int = 0


# ---------------------------------------------------------------------------
# Pooling helper (kept from the original pipeline)
# ---------------------------------------------------------------------------

def mean_pooling(token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
    """Average the token embeddings, ignoring padding positions.

    Padding tokens are multiplied by 0 so they do not contribute to the sum, and
    we divide by the number of REAL tokens (clamped to avoid division by zero).
    """
    expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = (token_embeddings * expanded_mask).sum(dim=1)
    token_counts = expanded_mask.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / token_counts


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed (non-learned) positional encoding from Vaswani et al. (2017).

    Adds a deterministic signal so the model knows token ORDER (self-attention is
    otherwise permutation-invariant). Even dimensions use sine, odd use cosine,
    each at a different wavelength:

        PE(pos, 2i)   = sin(pos / 10000^(2i/d))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    """

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Registered as a buffer => moves with .to(device) but is not a parameter.
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product multi-head self-attention, implemented by hand."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        # One linear each for queries, keys, values, and the output projection.
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, key_padding_mask: Tensor) -> Tensor:
        """
        Args:
            x:                [B, S, d_model] input token representations.
            key_padding_mask: [B, S] with 1 for real tokens, 0 for padding.
        """
        B, S, _ = x.shape

        def split_heads(t: Tensor) -> Tensor:
            # [B, S, d_model] -> [B, n_heads, S, d_head]
            return t.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        q = split_heads(self.q_proj(x))
        k = split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))

        # Attention scores: [B, n_heads, S, S]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Mask out padded KEYS so no query attends to them. Reshape mask to
        # [B, 1, 1, S] and set those score positions to -inf before softmax.
        mask = key_padding_mask[:, None, None, :].bool()  # [B,1,1,S]
        scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)                  # [B, n_heads, S, d_head]
        context = context.transpose(1, 2).contiguous().view(B, S, -1)  # [B, S, d_model]
        return self.out_proj(context)


class FeedForward(nn.Module):
    """Position-wise feed-forward network: Linear -> GELU -> Linear."""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class EncoderLayer(nn.Module):
    """A single pre-norm Transformer encoder block.

    Pre-norm (LayerNorm BEFORE each sub-layer) trains more stably from random
    initialization than the original post-norm formulation:

        x = x + Attention(LayerNorm(x))
        x = x + FeedForward(LayerNorm(x))
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor, key_padding_mask: Tensor) -> Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), key_padding_mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# The encoder (bi-encoder tower)
# ---------------------------------------------------------------------------

class BiEncoder(nn.Module):
    """From-scratch Transformer encoder that maps text -> unit-norm embedding."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_id
        )
        self.pos_encoding = SinusoidalPositionalEncoding(config.d_model, config.max_len)
        self.emb_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(EncoderLayer(config) for _ in range(config.n_layers))
        self.final_norm = nn.LayerNorm(config.d_model)

        self.embed_scale = math.sqrt(config.d_model)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Small, sane initialization for training from scratch."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Encode a batch of token id sequences into L2-normalized embeddings.

        Args:
            input_ids:      [B, S] token ids.
            attention_mask: [B, S] 1 for real tokens, 0 for padding.

        Returns:
            [B, d_model] unit-norm sentence embeddings.
        """
        # Truncate defensively if a sequence is longer than the positional table.
        if input_ids.size(1) > self.config.max_len:
            input_ids = input_ids[:, : self.config.max_len]
            attention_mask = attention_mask[:, : self.config.max_len]

        x = self.token_embedding(input_ids) * self.embed_scale  # [B, S, d_model]
        x = self.pos_encoding(x)
        x = self.emb_dropout(x)

        for layer in self.layers:
            x = layer(x, attention_mask)

        x = self.final_norm(x)                       # [B, S, d_model]
        pooled = mean_pooling(x, attention_mask)     # [B, d_model]
        return F.normalize(pooled, p=2, dim=1)       # unit length

    # ------------------------------------------------------------------
    # Persistence (config + weights travel together)
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save model weights AND config so :meth:`load` can rebuild exactly."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(self.config), "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> "BiEncoder":
        """Load a model previously written by :meth:`save`."""
        ckpt = torch.load(path, map_location=map_location)
        model = cls(EncoderConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Smoke test: random batch -> correct shape, unit norms, gradients flow.
    BATCH_SIZE, SEQ_LEN, VOCAB = 4, 24, 500
    cfg = EncoderConfig(vocab_size=VOCAB, d_model=128, n_layers=2, n_heads=4, d_ff=256, max_len=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiEncoder(cfg).to(device)
    print(f"Device: {device}   Parameters: {model.num_parameters():,}")

    input_ids = torch.randint(2, VOCAB, (BATCH_SIZE, SEQ_LEN), device=device)
    attention_mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long, device=device)
    attention_mask[0, -6:] = 0  # pad the first example

    emb = model(input_ids, attention_mask)
    assert emb.shape == (BATCH_SIZE, cfg.d_model), emb.shape
    norms = emb.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones(BATCH_SIZE, device=device), atol=1e-5), norms

    emb.sum().backward()
    assert model.token_embedding.weight.grad is not None
    print(f"Output shape: {tuple(emb.shape)}   L2 norms: {[round(n, 4) for n in norms.tolist()]}")
    print("All model smoke checks passed.")
