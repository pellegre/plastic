from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from tqdm import tqdm

from .network import BaseGraphNetwork


BatchHook = Optional[Callable[[BaseGraphNetwork, int], None]]
EpochHook = Optional[Callable[[BaseGraphNetwork, int, Dict[str, float]], None]]


@dataclass
class TrainingResult:
    train_loss: List[float]
    val_loss: List[float]
    train_accuracy: List[float]
    val_accuracy: List[float]
    lr_history: List[float]
    grad_norms: List[float]
    best_epoch: int
    best_val_loss: float
    best_state_dict: Dict[str, torch.Tensor]
    config: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        # Tensors in best_state_dict are already CPU tensors; convert to lists for JSON friendliness if needed.
        result["best_state_dict"] = {
            k: v.tolist() for k, v in self.best_state_dict.items()
        }
        return result


class Trainer:
    """
    Reusable training loop for graph sequence models.
    Collects metrics for plotting but leaves rendering to callers.
    """

    def __init__(
        self,
        epochs: int,
        batch_size: int,
        lr: float = 5e-4,
        val_split: float = 0.2,
        scheduler: str = "none",
        seed: Optional[int] = None,
        on_batch_start: BatchHook = None,
        on_epoch_end: EpochHook = None,
        device: Optional[torch.device] = None,
        world_size: int = 1,
        rank: int = 0,
    ) -> None:
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.val_split = float(val_split)
        self.scheduler_name = scheduler
        self.seed = seed
        self.on_batch_start = on_batch_start
        self.on_epoch_end = on_epoch_end
        self.device = device or torch.device("cpu")
        self.world_size = int(world_size)
        self.rank = int(rank)
        self._set_seed(seed)

    def to_config(self) -> Dict[str, Any]:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "val_split": self.val_split,
            "scheduler": self.scheduler_name,
            "seed": self.seed,
            "device": str(self.device),
            "world_size": self.world_size,
            "rank": self.rank,
        }

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        on_batch_start: BatchHook = None,
        on_epoch_end: EpochHook = None,
    ) -> "Trainer":
        cfg = dict(config)
        device = cfg.pop("device", None)
        world_size_value = int(cfg.pop("world_size", 1))
        rank_value = int(cfg.pop("rank", 0))
        return cls(
            on_batch_start=on_batch_start,
            on_epoch_end=on_epoch_end,
            device=torch.device(device) if device else None,
            world_size=world_size_value,
            rank=rank_value,
            **cfg,
        )

    def _set_seed(self, seed: Optional[int]) -> None:
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _split_data(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        num_samples = data.shape[0]
        indices = np.arange(num_samples)
        rng.shuffle(indices)
        val_count = max(1, int(num_samples * self.val_split))
        val_idx = indices[:val_count]
        train_idx = indices[val_count:]
        if train_idx.size == 0:
            train_idx = val_idx  # fallback to avoid empty train
        return data[train_idx], data[val_idx]

    def train(self, model: BaseGraphNetwork, data: np.ndarray) -> TrainingResult:
        model = model.to(self.device)
        model.train()

        base_model = model.module if hasattr(model, "module") else model

        train_data_np, val_data_np = self._split_data(data)
        train_tensor = torch.tensor(train_data_np, dtype=torch.float32)
        val_tensor = torch.tensor(val_data_np, dtype=torch.float32)

        train_dataset = TensorDataset(train_tensor)
        val_dataset = TensorDataset(val_tensor)

        train_sampler: Optional[DistributedSampler] = None
        val_sampler: Optional[DistributedSampler] = None
        per_rank_batch_size = self.batch_size
        if self.world_size > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=self.seed or 0,
                drop_last=False,
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                seed=self.seed or 0,
                drop_last=False,
            )
            per_rank_batch_size = int(
                math.ceil(self.batch_size / float(max(1, self.world_size)))
            )

        pin_memory = self.device.type == "cuda"
        train_loader = DataLoader(
            train_dataset,
            batch_size=per_rank_batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=0,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=per_rank_batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=0,
            pin_memory=pin_memory,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        if self.scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs
            )
        else:
            scheduler = None

        loss_fn = nn.CrossEntropyLoss()

        history_train_loss: List[float] = []
        history_val_loss: List[float] = []
        history_train_acc: List[float] = []
        history_val_acc: List[float] = []
        history_lr: List[float] = []
        history_grad_norm: List[float] = []

        best_val_loss = float("inf")
        best_epoch = -1
        best_state_dict: Dict[str, torch.Tensor] = {}

        epoch_iter = tqdm(range(self.epochs), desc="training", leave=False)
        for epoch in epoch_iter:
            epoch_train_loss = 0.0
            epoch_train_correct = 0
            epoch_train_total = 0
            epoch_train_samples = 0
            batch_grad_norms: List[float] = []

            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()
            for batch_idx, (batch_tensor,) in enumerate(train_loader):
                if self.on_batch_start:
                    self.on_batch_start(model, batch_idx)

                batch = batch_tensor.to(self.device, non_blocking=pin_memory)

                optimizer.zero_grad()

                logits = model(batch)  # [B, T-1, num_nodes]
                targets = batch[:, 1:, : base_model.num_nodes].argmax(dim=2)  # [B, T-1]

                loss = loss_fn(
                    logits.reshape(-1, base_model.num_nodes), targets.reshape(-1)
                )
                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=float("inf")
                )
                batch_grad_norms.append(float(grad_norm))
                optimizer.step()

                preds = logits.argmax(dim=2)
                correct = (preds == targets).sum().item()
                total = targets.numel()

                epoch_train_loss += loss.item() * batch.shape[0]
                epoch_train_correct += correct
                epoch_train_total += total
                epoch_train_samples += batch.shape[0]

            if scheduler is not None:
                scheduler.step()

            if (
                self.world_size > 1
                and dist.is_available()
                and dist.is_initialized()
            ):
                train_reduce = torch.tensor(
                    [
                        epoch_train_loss,
                        epoch_train_correct,
                        epoch_train_total,
                        epoch_train_samples,
                    ],
                    device=self.device,
                    dtype=torch.float64,
                )
                dist.all_reduce(train_reduce, op=dist.ReduceOp.SUM)
                epoch_train_loss = train_reduce[0].item()
                epoch_train_correct = int(train_reduce[1].item())
                epoch_train_total = int(train_reduce[2].item())
                epoch_train_samples = int(train_reduce[3].item())

            avg_train_loss = epoch_train_loss / max(1, epoch_train_samples)
            train_acc = epoch_train_correct / max(1, epoch_train_total)
            avg_grad_norm = (
                float(np.mean(batch_grad_norms)) if batch_grad_norms else 0.0
            )

            # Validation
            model.eval()
            with torch.no_grad():
                val_loss_total = 0.0
                val_correct = 0
                val_total = 0
                val_samples = 0
                for batch_idx, (batch_tensor,) in enumerate(val_loader):
                    if self.on_batch_start:
                        self.on_batch_start(model, batch_idx)

                    batch = batch_tensor.to(self.device, non_blocking=pin_memory)

                    logits = model(batch)
                    targets = batch[:, 1:, : base_model.num_nodes].argmax(dim=2)
                    loss = loss_fn(
                        logits.reshape(-1, base_model.num_nodes), targets.reshape(-1)
                    )

                    preds = logits.argmax(dim=2)
                    correct = (preds == targets).sum().item()
                    total = targets.numel()

                    val_loss_total += loss.item() * batch.shape[0]
                    val_correct += correct
                    val_total += total
                    val_samples += batch.shape[0]

                if (
                    self.world_size > 1
                    and dist.is_available()
                    and dist.is_initialized()
                ):
                    val_reduce = torch.tensor(
                        [val_loss_total, val_correct, val_total, val_samples],
                        device=self.device,
                        dtype=torch.float64,
                    )
                    dist.all_reduce(val_reduce, op=dist.ReduceOp.SUM)
                    val_loss_total = val_reduce[0].item()
                    val_correct = int(val_reduce[1].item())
                    val_total = int(val_reduce[2].item())
                    val_samples = int(val_reduce[3].item())

            avg_val_loss = val_loss_total / max(1, val_samples)
            val_acc = val_correct / max(1, val_total)

            history_train_loss.append(avg_train_loss)
            history_val_loss.append(avg_val_loss)
            history_train_acc.append(train_acc)
            history_val_acc.append(val_acc)
            history_grad_norm.append(avg_grad_norm)
            history_lr.append(optimizer.param_groups[0]["lr"])

            metrics = {
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
                "grad_norm": avg_grad_norm,
            }
            if self.on_epoch_end:
                self.on_epoch_end(model, epoch, metrics)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                best_state_dict = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }

            epoch_iter.set_postfix(
                train_loss=f"{avg_train_loss:.4f}",
                val_loss=f"{avg_val_loss:.4f}",
                train_acc=f"{train_acc:.4f}",
                val_acc=f"{val_acc:.4f}",
            )

        result = TrainingResult(
            train_loss=history_train_loss,
            val_loss=history_val_loss,
            train_accuracy=history_train_acc,
            val_accuracy=history_val_acc,
            lr_history=history_lr,
            grad_norms=history_grad_norm,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            best_state_dict=best_state_dict,
            config={
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "val_split": self.val_split,
                "scheduler": self.scheduler_name,
                "seed": self.seed,
            },
        )
        return result
