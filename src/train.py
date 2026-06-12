"""
src/train.py — training loop for the from-scratch BiEncoder with InfoNCE.

The trainer is deliberately plain PyTorch (no HuggingFace Trainer): it builds two
DataLoaders, runs an AdamW + linear-warmup loop, logs per-epoch losses to CSV,
and saves the best/final checkpoints together with the tokenizer so the model can
be reloaded for evaluation and the demo.

Hyperparameters differ from BERT fine-tuning because we train FROM SCRATCH:
a higher learning rate (3e-4) and more epochs are needed for the randomly
initialized weights to converge.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.dataset import InBatchDataset
from src.loss import InfoNCELoss
from src.model import BiEncoder
from src.tokenizer import BPETokenizer


def _get_linear_warmup_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Linear warmup from 0 -> peak over ``warmup_steps``, then linear decay -> 0."""

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)


def _forward_pass(
    model: BiEncoder,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode the query and positive halves of a batch into embeddings."""
    q_ids = batch["query_input_ids"].to(device)
    q_mask = batch["query_attention_mask"].to(device)
    p_ids = batch["pos_input_ids"].to(device)
    p_mask = batch["pos_attention_mask"].to(device)

    query_emb = model(q_ids, q_mask)
    pos_emb = model(p_ids, p_mask)
    return query_emb, pos_emb


class Trainer:
    def __init__(
        self,
        model: BiEncoder,
        tokenizer: BPETokenizer,
        train_pairs: List[Dict],
        val_pairs: List[Dict],
        output_dir: str | Path = "checkpoints",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        query_max_len: int = 64,
        doc_max_len: int = 256,
        batch_size: int = 64,
        lr: float = 3e-4,
        epochs: int = 20,
        temperature: float = 0.05,
        warmup_ratio: float = 0.10,
        weight_decay: float = 1e-2,
        grad_clip: Optional[float] = 1.0,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.epochs = epochs
        self.grad_clip = grad_clip

        # DataLoaders (dynamic padding via the dataset's collate_fn).
        train_ds = InBatchDataset(train_pairs, tokenizer, query_max_len, doc_max_len)
        val_ds = InBatchDataset(val_pairs, tokenizer, query_max_len, doc_max_len)
        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            collate_fn=train_ds.collate_fn,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            collate_fn=val_ds.collate_fn,
        )

        # Loss
        self.criterion = InfoNCELoss(temperature=temperature).to(self.device)

        # Optimizer: weight decay only on 2D+ weights (matrices/embeddings), not
        # on biases or LayerNorm gains (the standard convention).
        decay, no_decay = [], []
        for param in model.parameters():
            if not param.requires_grad:
                continue
            (decay if param.ndim >= 2 else no_decay).append(param)
        self.optimizer = AdamW(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=lr,
        )

        # LR scheduler
        total_steps = len(self.train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = _get_linear_warmup_scheduler(self.optimizer, warmup_steps, total_steps)

        # Save the tokenizer next to the checkpoints so eval/demo can reload it.
        self.tokenizer.save(self.output_dir / "tokenizer.json")

        # Log file
        self.log_path = self.output_dir / "training_log.csv"
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "step", "train_loss", "val_loss", "lr"])

        self.best_val_loss = float("inf")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        steps = len(self.train_loader)

        for step, batch in enumerate(self.train_loader, 1):
            self.optimizer.zero_grad()
            query_emb, pos_emb = _forward_pass(self.model, batch, self.device)
            loss = self.criterion(query_emb, pos_emb)
            loss.backward()

            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            if step % 10 == 0 or step == steps:
                avg = total_loss / step
                lr = self.scheduler.get_last_lr()[0]
                print(
                    f"  Epoch {epoch} | step {step:>4}/{steps} "
                    f"| loss {loss.item():.4f} | avg {avg:.4f} | lr {lr:.2e}",
                    end="\r",
                )
        print()
        return total_loss / steps

    @torch.no_grad()
    def _val_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0
        for batch in self.val_loader:
            query_emb, pos_emb = _forward_pass(self.model, batch, self.device)
            total_loss += self.criterion(query_emb, pos_emb).item()
        return total_loss / len(self.val_loader)

    def _save_checkpoint(self, tag: str) -> Path:
        path = self.output_dir / f"{tag}.pt"
        self.model.save(path)  # saves config + weights together
        return path

    def _log_row(self, epoch: int, step: int, train_loss: float, val_loss: float) -> None:
        lr = self.scheduler.get_last_lr()[0]
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, step, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{lr:.2e}"]
            )

    def fit(self) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        print(f"Training on: {self.device}   model params: {self.model.num_parameters():,}")
        print(f"Train batches/epoch: {len(self.train_loader)}   "
              f"Total steps: {len(self.train_loader) * self.epochs}\n")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(epoch)
            val_loss = self._val_epoch()
            elapsed = time.time() - t0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            self._log_row(epoch, len(self.train_loader) * epoch, train_loss, val_loss)

            improved = ""
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = self._save_checkpoint("best")
                improved = f"  saved best -> {best_path.name}"

            print(
                f"Epoch {epoch}/{self.epochs} | train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  time={elapsed:.1f}s{improved}"
            )

        final_path = self._save_checkpoint("final")
        print(f"\nFinal checkpoint saved -> {final_path}")
        print(f"Log saved -> {self.log_path}")
        return history


def build_tokenizer_from_pairs(
    chunks: List[Dict],
    train_pairs: List[Dict],
    vocab_size: int = 8000,
    verbose: bool = True,
) -> BPETokenizer:
    """Train a BPE tokenizer on the search corpus + all training-pair text.

    Including the queries and (Wikipedia) positives/negatives — not just the
    Jurafsky chunks — gives the tokenizer good coverage of every word it will
    actually see at train and search time.
    """
    texts: List[str] = [c["content"] for c in chunks]
    for p in train_pairs:
        texts.append(p["query"])
        texts.append(p["positive_text"])
        if "negative_text" in p:
            texts.append(p["negative_text"])
    return BPETokenizer.train(texts, vocab_size=vocab_size, verbose=verbose)


if __name__ == "__main__":
    import argparse
    import json

    from src.model import EncoderConfig

    parser = argparse.ArgumentParser(description="Train the from-scratch BiEncoder")
    parser.add_argument("--chunks", default="data/processed/jurafsky_chunks_v2.json")
    parser.add_argument("--train_pairs", default="data/processed/train_pairs_combined.json")
    parser.add_argument("--val_pairs", default="data/processed/val_pairs_combined.json")
    parser.add_argument("--output", default="checkpoints")
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--doc_max_len", type=int, default=256)
    parser.add_argument("--query_max_len", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=0, help="cap #train pairs (0 = all; for quick tests)")
    args = parser.parse_args()

    def _load(path: str):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    chunks = _load(args.chunks)
    train_pairs = _load(args.train_pairs)
    val_pairs = _load(args.val_pairs)
    if args.limit:
        train_pairs = train_pairs[: args.limit]
        val_pairs = val_pairs[: max(1, args.limit // 10)]
    print(f"Chunks: {len(chunks)}  Train pairs: {len(train_pairs)}  Val pairs: {len(val_pairs)}")

    tokenizer = build_tokenizer_from_pairs(chunks, train_pairs, vocab_size=args.vocab_size)
    config = EncoderConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_model * 4,
        max_len=args.doc_max_len,
    )
    model = BiEncoder(config)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        output_dir=args.output,
        query_max_len=args.query_max_len,
        doc_max_len=args.doc_max_len,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        temperature=args.temperature,
    )
    trainer.fit()
