from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class EncoderConfig:
    vocab_size: int                 # set from the trained tokenizer
    d_model: int = 256              # embedding / hidden size
    n_layers: int = 4               # number of Transformer encoder blocks
    n_heads: int = 4                # attention heads (d_model must divide by this)
    d_ff: int = 1024                # feed-forward inner size (usually 4 * d_model)
    max_len: int = 256              # maximum sequence length (for positional table)
    dropout: float = 0.1
    pad_id: int = 0


def mean_pooling(token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
    expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = (token_embeddings * expanded_mask).sum(dim=1)
    token_counts = expanded_mask.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / token_counts


class SinusoidalPositionalEncoding(nn.Module):

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
        B, S, _ = x.shape

        def split_heads(t: Tensor) -> Tensor:
            # [B, S, d_model] -> [B, n_heads, S, d_head]
            return t.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        q = split_heads(self.q_proj(x))
        k = split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))

        # Attention scores: [B, n_heads, S, S]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        mask = key_padding_mask[:, None, None, :].bool()  # [B,1,1,S]
        scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)                  # [B, n_heads, S, d_head]
        context = context.transpose(1, 2).contiguous().view(B, S, -1)  # [B, S, d_model]
        return self.out_proj(context)


class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class EncoderLayer(nn.Module):

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


class BiEncoder(nn.Module):

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

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(self.config), "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> "BiEncoder":
        ckpt = torch.load(path, map_location=map_location)
        model = cls(EncoderConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

