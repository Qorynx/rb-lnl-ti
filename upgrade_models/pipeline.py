"""Shared data, training and evaluation utilities for RB-LNL-Ti.

The original notebooks duplicated most of the training code and accidentally
used the official GTSRB test split as validation data.  This module is the
single source of truth for the refactored stage scripts.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_config(path: Optional[str | os.PathLike[str]] = None) -> Dict[str, Any]:
    import yaml

    config_path = Path(path) if path else Path(__file__).with_name("config.yaml")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _targets_from_dataset(dataset: Dataset) -> np.ndarray:
    for name in ("targets", "_targets", "_labels"):
        values = getattr(dataset, name, None)
        if values is not None:
            return np.asarray(values, dtype=np.int64)
    samples = getattr(dataset, "_samples", None)
    if samples is not None:
        return np.asarray([sample[1] for sample in samples], dtype=np.int64)
    raise AttributeError("Unable to find labels on the torchvision GTSRB dataset")


class IndexedDataset(Dataset):
    """Return the original dataset index alongside image and target."""

    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        image, target = self.dataset[int(self.indices[item])]
        return image, int(target), int(self.indices[item])


def _stratified_split(targets: np.ndarray, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for label in np.unique(targets):
        class_indices = np.flatnonzero(targets == label)
        rng.shuffle(class_indices)
        n_val = max(1, int(round(len(class_indices) * val_fraction)))
        val_indices.extend(class_indices[:n_val].tolist())
        train_indices.extend(class_indices[n_val:].tolist())
    return np.asarray(sorted(train_indices)), np.asarray(sorted(val_indices))


def build_transforms(train: bool, clean: bool = False):
    if train and not clean:
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )
    elif train and clean:
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )
    else:
        transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
        )
    return transform


def build_gtsrb_splits(
    data_dir: str,
    val_fraction: float,
    seed: int,
    clean: bool = False,
    include_test: bool = False,
):
    """Create train/validation datasets and optionally the official test set."""

    data_root = Path(data_dir)
    data_root.mkdir(parents=True, exist_ok=True)
    train_meta = datasets.GTSRB(root=data_root, split="train", transform=None, download=True)
    targets = _targets_from_dataset(train_meta)
    train_indices, val_indices = _stratified_split(targets, val_fraction, seed)

    train_base = datasets.GTSRB(
        root=data_root, split="train", transform=build_transforms(True, clean=clean), download=False
    )
    eval_base = datasets.GTSRB(root=data_root, split="train", transform=build_transforms(False), download=False)
    result = {
        "train": IndexedDataset(train_base, train_indices),
        "train_eval": IndexedDataset(eval_base, train_indices),
        "val": IndexedDataset(eval_base, val_indices),
        "test": None,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "full_train_size": len(train_meta),
    }
    if include_test:
        test_base = datasets.GTSRB(root=data_root, split="test", transform=build_transforms(False), download=True)
        result["test"] = IndexedDataset(test_base, np.arange(len(test_base)))
    return result


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9998):
        self.decay = decay
        self.shadow = {name: value.detach().clone() for name, value in model.state_dict().items()}
        self.backup: Optional[Dict[str, torch.Tensor]] = None

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, value in model.state_dict().items():
            if value.is_floating_point():
                self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[name].copy_(value)

    def store(self, model: nn.Module) -> None:
        self.backup = {name: value.detach().clone() for name, value in model.state_dict().items()}

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model: nn.Module) -> None:
        if self.backup is not None:
            model.load_state_dict(self.backup, strict=True)
            self.backup = None

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {name: value.clone() for name, value in self.shadow.items()}

    def load_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        self.shadow = {name: value.clone() for name, value in state.items()}


def autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


def make_scaler(device: torch.device, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled and device.type == "cuda")
    return torch.cuda.amp.GradScaler(enabled=enabled and device.type == "cuda")


def cosine_with_warmup(optimizer: Optimizer, warmup_epochs: int, total_epochs: int, min_lr_ratio: float = 0.01):
    def factor(epoch: int) -> float:
        if warmup_epochs and epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs - 1)
        cosine = 0.5 * (1.0 + np.cos(np.pi * np.clip(progress, 0.0, 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * float(cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, ema: Optional[ModelEMA] = None):
    model.eval()
    if ema is not None:
        ema.store(model)
        ema.copy_to(model)
    correct = 0
    total = 0
    loss_sum = 0.0
    try:
        for images, targets, _ in loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with autocast_context(device, enabled=False):
                logits = model(images)
                loss = F.cross_entropy(logits, targets)
            loss_sum += float(loss.item()) * targets.size(0)
            correct += int(logits.argmax(dim=1).eq(targets).sum().item())
            total += targets.size(0)
    finally:
        if ema is not None:
            ema.restore(model)
    return {"loss": loss_sum / max(1, total), "accuracy": 100.0 * correct / max(1, total)}


@torch.no_grad()
def log_difficulty(model: nn.Module, loader: DataLoader, device: torch.device, path: str, ema: Optional[ModelEMA] = None):
    model.eval()
    if ema is not None:
        ema.store(model)
        ema.copy_to(model)
    rows = []
    try:
        for images, targets, indices in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            probabilities = logits.softmax(dim=1)
            losses = F.cross_entropy(logits, targets, reduction="none")
            top_probs, top_classes = probabilities.topk(2, dim=1)
            pred = top_classes[:, 0]
            pred_confidence = top_probs[:, 0]
            true_confidence = probabilities.gather(1, targets[:, None]).squeeze(1)
            margin = top_probs[:, 0] - top_probs[:, 1]
            for row in range(targets.size(0)):
                is_correct = bool(pred[row] == targets[row])
                if is_correct and margin[row] >= 0.5 and losses[row] < 0.3:
                    group = "easy"
                elif is_correct:
                    group = "ambiguous"
                elif pred_confidence[row] >= 0.9:
                    group = "suspected_noise"
                else:
                    group = "hard"
                rows.append(
                    {
                        "image_id": int(indices[row]),
                        "true_label": int(targets[row]),
                        "predicted_label": int(pred[row]),
                        "loss": float(losses[row]),
                        "true_confidence": float(true_confidence[row]),
                        "pred_confidence": float(pred_confidence[row]),
                        "top1_prob": float(top_probs[row, 0]),
                        "top2_prob": float(top_probs[row, 1]),
                        "margin": float(margin[row]),
                        "difficulty_group": group,
                    }
                )
    finally:
        if ema is not None:
            ema.restore(model)
    frame = pd.DataFrame(rows).sort_values("image_id")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


@torch.no_grad()
def log_confusion(model: nn.Module, loader: DataLoader, device: torch.device, path: str, num_classes: int, ema=None):
    model.eval()
    if ema is not None:
        ema.store(model)
        ema.copy_to(model)
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    try:
        for images, targets, _ in loader:
            logits = model(images.to(device, non_blocking=True))
            predictions = logits.argmax(dim=1).cpu()
            for true_label, predicted_label in zip(targets, predictions):
                matrix[int(true_label), int(predicted_label)] += 1
    finally:
        if ema is not None:
            ema.restore(model)
    frame = pd.DataFrame(matrix.numpy())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


@torch.no_grad()
def log_predictions(model: nn.Module, loader: DataLoader, device: torch.device, path: str, num_classes: int, ema=None):
    """Write final predictions and return the corresponding confusion matrix."""
    model.eval()
    if ema is not None:
        ema.store(model)
        ema.copy_to(model)
    rows = []
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    try:
        for images, targets, indices in loader:
            logits = model(images.to(device, non_blocking=True))
            probabilities = logits.softmax(dim=1)
            confidence, predictions = probabilities.max(dim=1)
            for index, target, prediction, score in zip(indices, targets, predictions.cpu(), confidence.cpu()):
                rows.append(
                    {
                        "image_id": int(index),
                        "true_label": int(target),
                        "predicted_label": int(prediction),
                        "confidence": float(score),
                    }
                )
                matrix[int(target), int(prediction)] += 1
    finally:
        if ema is not None:
            ema.restore(model)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("image_id").to_csv(path, index=False)
    return pd.DataFrame(matrix.numpy())


def sample_weights_from_csv(path: str, dataset_size: int, alpha: float, max_weight: float) -> torch.Tensor:
    frame = pd.read_csv(path)
    weights = torch.ones(dataset_size, dtype=torch.float32)
    if frame.empty:
        return weights
    lower, upper = frame["loss"].quantile(0.05), frame["loss"].quantile(0.95)
    normalized = ((frame["loss"] - lower) / max(float(upper - lower), 1e-8)).clip(0.0, 1.0)
    eligible = frame["difficulty_group"].isin(["hard", "ambiguous"])
    frame = frame.assign(weight=1.0)
    frame.loc[eligible, "weight"] = np.clip(1.0 + alpha * normalized[eligible], 1.0, max_weight)
    for row in frame.itertuples(index=False):
        image_id = int(row.image_id)
        if 0 <= image_id < dataset_size:
            weights[image_id] = float(row.weight)
    return weights


def save_checkpoint(path: str, model: nn.Module, optimizer: Optimizer, scheduler, scaler, ema: Optional[ModelEMA], epoch: int, best: float, config: Dict[str, Any], stage: str):
    payload = {
        "epoch": epoch,
        "stage": stage,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "ema_state_dict": ema.state_dict() if ema else None,
        "best_val_accuracy": best,
        "config": config,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str, model: nn.Module, optimizer=None, scheduler=None, scaler=None, ema=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if ema is not None and checkpoint.get("ema_state_dict"):
        ema.load_state_dict(checkpoint["ema_state_dict"])
    return checkpoint


def save_metrics(path: str, metrics: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(dict(metrics), handle, indent=2)
