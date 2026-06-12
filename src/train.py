
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.dataset import InBatchDataset, load_chunks, generate_pairs, split_pairs
from src.loss import InfoNCELoss
from src.model import BiEncoder

def _get_linear_warmup_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    
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

    q_ids  = batch["query_input_ids"].to(device)
    q_mask = batch["query_attention_mask"].to(device)
    p_ids  = batch["pos_input_ids"].to(device)
    p_mask = batch["pos_attention_mask"].to(device)

    query_emb = model(q_ids, q_mask)
    pos_emb   = model(p_ids, p_mask)
    return query_emb, pos_emb

class Trainer:
    def __init__(
        self,
        model: BiEncoder,
        train_pairs: List[Dict],
        val_pairs: List[Dict],
        output_dir: str | Path = "checkpoints",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 256,
        batch_size: int = 32,
        lr: float = 2e-5,
        epochs: int = 5,
        temperature: float = 0.07,
        warmup_ratio: float = 0.10,
        grad_clip: Optional[float] = 1.0,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.epochs = epochs
        self.grad_clip = grad_clip

        # Tokenizer — must match the backbone loaded in BiEncoder
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        # DataLoaders
        train_ds = InBatchDataset(train_pairs, tokenizer, max_length)
        val_ds   = InBatchDataset(val_pairs, tokenizer, max_length)

        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, pin_memory=True,
        )

        # Loss
        self.criterion = InfoNCELoss(temperature=temperature).to(self.device)

        # Optimizer — no weight decay on bias and LayerNorm parameters
        no_decay = {"bias", "LayerNorm.weight"}
        param_groups = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 1e-2,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(param_groups, lr=lr)

        # LR scheduler
        total_steps = len(self.train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = _get_linear_warmup_scheduler(
            self.optimizer, warmup_steps, total_steps
        )

        # Log file
        self.log_path = self.output_dir / "training_log.csv"
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "step", "train_loss", "val_loss", "lr"])

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
            current_lr = self.scheduler.get_last_lr()[0]

            # Print progress every 5 steps
            if step % 5 == 0 or step == steps:
                avg = total_loss / step
                print(
                    f"  Epoch {epoch} | step {step:>3}/{steps} "
                    f"| loss {loss.item():.4f} | avg {avg:.4f} | lr {current_lr:.2e}",
                    end="\r",
                )

        print()  # newline after \r progress
        return total_loss / steps

    @torch.no_grad()
    def _val_epoch(self) -> float:
        
        self.model.eval()
        total_loss = 0.0
        for batch in self.val_loader:
            query_emb, pos_emb = _forward_pass(self.model, batch, self.device)
            loss = self.criterion(query_emb, pos_emb)
            total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def _save_checkpoint(self, tag: str) -> Path:
        
        path = self.output_dir / f"{tag}.pt"
        torch.save(self.model.state_dict(), path)
        return path

    def _log_row(self, epoch: int, step: int, train_loss: float, val_loss: float) -> None:
        current_lr = self.scheduler.get_last_lr()[0]
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, step, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{current_lr:.2e}"])

    def fit(self) -> Dict[str, List[float]]:
        
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        print(f"Training on: {self.device}")
        print(f"Train batches per epoch: {len(self.train_loader)}")
        print(f"Val batches per epoch  : {len(self.val_loader)}")
        print(f"Total steps            : {len(self.train_loader) * self.epochs}\n")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(epoch)
            val_loss   = self._val_epoch()
            elapsed    = time.time() - t0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            self._log_row(epoch, len(self.train_loader) * epoch, train_loss, val_loss)

            tag = "best"
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = self._save_checkpoint(tag)
                improved = f"  saved best -> {best_path.name}"
            else:
                improved = ""

            print(
                f"Epoch {epoch}/{self.epochs} | "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"time={elapsed:.1f}s{improved}"
            )

        final_path = self._save_checkpoint("final")
        print(f"\nFinal checkpoint saved -> {final_path}")
        print(f"Log saved -> {self.log_path}")
        return history


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Train the BiEncoder")
    parser.add_argument("--chunks", default="data/processed/jurafsky_chunks.json")
    parser.add_argument("--output", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.07)
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    pairs  = generate_pairs(chunks)
    train_pairs, val_pairs = split_pairs(pairs)
    print(f"Train pairs: {len(train_pairs)}  Val pairs: {len(val_pairs)}")

    model = BiEncoder()
    trainer = Trainer(
        model=model,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        output_dir=args.output,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        max_length=args.max_length,
        temperature=args.temperature,
    )
    history = trainer.fit()
