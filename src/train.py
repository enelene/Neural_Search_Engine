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
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Encode the query, positive (and optional negative) halves of a batch."""
    query_emb = model(
        batch["query_input_ids"].to(device), batch["query_attention_mask"].to(device)
    )
    pos_emb = model(
        batch["pos_input_ids"].to(device), batch["pos_attention_mask"].to(device)
    )
    neg_emb = None
    if "neg_input_ids" in batch:
        neg_emb = model(
            batch["neg_input_ids"].to(device), batch["neg_attention_mask"].to(device)
        )
    return query_emb, pos_emb, neg_emb


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
        use_negatives: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.epochs = epochs
        self.grad_clip = grad_clip

        train_ds = InBatchDataset(train_pairs, tokenizer, query_max_len, doc_max_len, use_negatives)
        val_ds = InBatchDataset(val_pairs, tokenizer, query_max_len, doc_max_len, use_negatives)
        self.use_negatives = train_ds.use_negatives
        print(f"Explicit hard negatives: {'ON' if self.use_negatives else 'OFF'}")
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

        total_steps = len(self.train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = _get_linear_warmup_scheduler(self.optimizer, warmup_steps, total_steps)

        self.tokenizer.save(self.output_dir / "tokenizer.json")

        # Log file
        self.log_path = self.output_dir / "training_log.csv"
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "step", "train_loss", "val_loss", "lr"])

        self.best_val_loss = float("inf")


    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        steps = len(self.train_loader)

        for step, batch in enumerate(self.train_loader, 1):
            self.optimizer.zero_grad()
            query_emb, pos_emb, neg_emb = _forward_pass(self.model, batch, self.device)
            loss = self.criterion(query_emb, pos_emb, neg_emb)
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
            query_emb, pos_emb, neg_emb = _forward_pass(self.model, batch, self.device)
            total_loss += self.criterion(query_emb, pos_emb, neg_emb).item()
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
    texts: List[str] = [c["content"] for c in chunks]
    for p in train_pairs:
        texts.append(p["query"])
        texts.append(p["positive_text"])
        if "negative_text" in p:
            texts.append(p["negative_text"])
    return BPETokenizer.train(texts, vocab_size=vocab_size, verbose=verbose)

