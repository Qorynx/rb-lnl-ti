"""Implementation of the four reproducible RB-LNL-Ti training stages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn, optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from upgrade_models.pipeline import (
    ModelEMA,
    autocast_context,
    build_gtsrb_splits,
    cosine_with_warmup,
    evaluate,
    load_checkpoint,
    load_config,
    log_confusion,
    log_difficulty,
    log_predictions,
    make_loader,
    make_scaler,
    sample_weights_from_csv,
    save_checkpoint,
    save_metrics,
    set_seed,
)


def _paths(config: Dict[str, Any]) -> Dict[str, Path]:
    root = Path(config["output"]["root"])
    results = Path(config["output"]["results"])
    root.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "results": results,
        "stage1_best": root / "rb_lnl_ti_stage1.pth",
        "stage2_best": root / "rb_lnl_ti_stage2.pth",
        "stage3_best": root / "rb_lnl_ti_stage3.pth",
        "final": root / "rb_lnl_ti_gtsrb.pth",
    }


def _model(config: Dict[str, Any], residual_enabled: bool):
    # Import lazily so --help/config inspection works before the legacy timm
    # dependency is installed.
    from upgrade_models.rb_lnl_ti import RB_LNL_Ti

    model_cfg = config["model"]
    fixed_base = {"image_size": 224, "embed_dim": 192, "depth": 12}
    for key, expected in fixed_base.items():
        if int(model_cfg[key]) != expected:
            raise ValueError(f"{key}={model_cfg[key]} is incompatible with the untouched LNL-Ti base; expected {expected}")
    return RB_LNL_Ti(
        num_classes=model_cfg["num_classes"],
        residual_hidden_dim=model_cfg["residual_hidden_dim"],
        residual_gate_hidden_dim=model_cfg["residual_gate_hidden_dim"],
        residual_dropout=model_cfg["residual_dropout"],
        residual_scale_init=model_cfg["residual_scale_init"],
        residual_gate_init=model_cfg["residual_gate_init"],
        drop_path_rate=model_cfg["drop_path_rate"],
        residual_enabled=residual_enabled,
    )


def _load_weights(model: nn.Module, path: Path, device: torch.device) -> None:
    state = torch.load(path, map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)


def _make_loaders(config: Dict[str, Any], clean: bool = False, include_test: bool = False):
    data_cfg = config["data"]
    splits = build_gtsrb_splits(
        data_cfg["root"],
        val_fraction=float(data_cfg["validation_fraction"]),
        seed=int(config["seed"]),
        clean=clean,
        include_test=include_test,
    )
    loader_cfg = config["data"]
    batch_size = int(config["train"]["batch_size"])
    workers = int(loader_cfg["num_workers"])
    pin_memory = bool(loader_cfg["pin_memory"])
    return splits, {
        "train": make_loader(splits["train"], batch_size, True, workers, pin_memory),
        "train_eval": make_loader(splits["train_eval"], batch_size, False, workers, pin_memory),
        "val": make_loader(splits["val"], batch_size, False, workers, pin_memory),
        "test": make_loader(splits["test"], batch_size, False, workers, pin_memory) if splits["test"] else None,
    }


def _optimizer_for_stage(model: nn.Module, config: Dict[str, Any], stage: str):
    train_cfg = config["train"]
    stage_cfg = config["stages"][stage]
    if stage != "stage3":
        lr = float(stage_cfg["learning_rate"])
        params = [parameter for parameter in model.parameters() if parameter.requires_grad]
        return optim.AdamW(params, lr=lr, weight_decay=float(train_cfg["weight_decay"]))

    for parameter in model.parameters():
        parameter.requires_grad = False
    late, base_head, residual = [], [], []
    for name, parameter in model.named_parameters():
        if "residual_head" in name or "residual_gate" in name or "residual_scale" in name:
            parameter.requires_grad = True
            residual.append(parameter)
        elif "backbone.head" in name:
            parameter.requires_grad = True
            base_head.append(parameter)
        elif "backbone.blocks" in name:
            block_index = int(name.split(".")[2])
            if block_index >= 8:
                parameter.requires_grad = True
                late.append(parameter)
        elif "backbone.norm" in name:
            parameter.requires_grad = True
            late.append(parameter)
    return optim.AdamW(
        [
            {"params": late, "lr": float(stage_cfg["late_backbone_learning_rate"])},
            {"params": base_head, "lr": float(stage_cfg["base_head_learning_rate"])},
            {"params": residual, "lr": float(stage_cfg["residual_learning_rate"])},
        ],
        weight_decay=float(train_cfg["weight_decay"]),
    )


def _train_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    optimizer,
    scaler,
    ema: Optional[ModelEMA],
    accumulation_steps: int,
    label_smoothing: float,
    sample_weights: Optional[torch.Tensor] = None,
    preserve_base: bool = False,
    preserve_weight: float = 0.02,
    residual_l2_weight: float = 0.0001,
    clip_norm: float = 1.0,
    amp: bool = True,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_total = 0.0
    correct = 0
    total = 0
    steps = 0
    for step, (images, targets, indices) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast_context(device, amp):
            logits, aux = model(images, return_aux=True)
            losses = F.cross_entropy(logits, targets, label_smoothing=label_smoothing, reduction="none")
            if sample_weights is not None:
                losses = losses * sample_weights[indices].to(device)
            loss = losses.mean()
            if preserve_base:
                base_distribution = aux["base_logits"].detach().softmax(dim=1)
                preservation = F.kl_div(
                    logits.log_softmax(dim=1), base_distribution, reduction="batchmean"
                )
                residual_penalty = aux["residual_logits"].pow(2).mean()
                loss = loss + preserve_weight * preservation + residual_l2_weight * residual_penalty
            scaled_loss = loss / max(1, accumulation_steps)
        scaler.scale(scaled_loss).backward()
        should_update = (step + 1) % accumulation_steps == 0 or step + 1 == len(loader)
        if should_update:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
        loss_total += float(loss.detach().item())
        correct += int(logits.argmax(dim=1).eq(targets).sum().item())
        total += targets.size(0)
        steps += 1
    return {"loss": loss_total / max(1, steps), "accuracy": 100.0 * correct / max(1, total)}


def _save_best(model, ema: Optional[ModelEMA], path: Path) -> None:
    if ema is None:
        torch.save(model.state_dict(), path)
        return
    ema.store(model)
    ema.copy_to(model)
    torch.save(model.state_dict(), path)
    ema.restore(model)


def _stage_epochs(config: Dict[str, Any], stage: str):
    stage_cfg = config["stages"][stage]
    return int(stage_cfg["start_epoch"]), int(stage_cfg["end_epoch"])


def run_stage(stage: str, config_path: Optional[str] = None, resume: bool = False) -> Dict[str, Any]:
    config = load_config(config_path)
    set_seed(int(config["seed"]))
    paths = _paths(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_epoch, end_epoch = _stage_epochs(config, stage)
    if start_epoch < 0 or end_epoch < start_epoch or end_epoch >= int(config["train"]["epochs"]):
        raise ValueError("config train.epochs and stage epoch ranges are inconsistent")
    required_inputs = {
        "stage2": [paths["stage1_best"], paths["results"] / "sample_difficulty.csv"],
        "stage3": [paths["stage2_best"]],
        "stage4": [paths["stage3_best"]],
    }.get(stage, [])
    missing = [str(path) for path in required_inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{stage} cannot start; missing input artifacts: {missing}")
    clean = stage == "stage4"
    residual_enabled = stage in {"stage3", "stage4"}
    splits, loaders = _make_loaders(config, clean=clean, include_test=False)
    model = _model(config, residual_enabled=residual_enabled).to(device)

    if stage == "stage2":
        _load_weights(model, paths["stage1_best"], device)
        model.set_residual_enabled(False)
    elif stage == "stage3":
        _load_weights(model, paths["stage2_best"], device)
        model.set_residual_enabled(True)
    elif stage == "stage4":
        _load_weights(model, paths["stage3_best"], device)
        model.set_residual_enabled(True)

    if stage == "stage4":
        for parameter in model.parameters():
            parameter.requires_grad = True

    optimizer = _optimizer_for_stage(model, config, stage)
    train_cfg = config["train"]
    stage_cfg = config["stages"][stage]
    total_epochs = end_epoch - start_epoch + 1
    scheduler = cosine_with_warmup(
        optimizer,
        warmup_epochs=int(train_cfg["warmup_epochs"]) if stage == "stage1" else 0,
        total_epochs=total_epochs,
        min_lr_ratio=float(stage_cfg.get("minimum_learning_rate", train_cfg["minimum_learning_rate"]))
        / max(float(stage_cfg.get("learning_rate", train_cfg["learning_rate"])), 1e-12),
    )
    scaler = make_scaler(device, bool(train_cfg["amp"]))
    ema = ModelEMA(model, float(config["ema"]["decay"])) if config["ema"]["enabled"] else None

    sample_weights = None
    if stage == "stage2":
        sample_weights = sample_weights_from_csv(
            str(paths["results"] / "sample_difficulty.csv"),
            int(splits["full_train_size"]),
            float(config["hard_example"]["alpha"]),
            float(config["hard_example"]["max_weight"]),
        )

    best_accuracy = -1.0
    history = []
    checkpoint_path = paths["root"] / f"latest_{stage}.pt"
    if resume and checkpoint_path.exists():
        checkpoint = load_checkpoint(
            str(checkpoint_path), model, optimizer, scheduler, scaler, ema, device=device
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_accuracy = float(checkpoint.get("best_val_accuracy", -1.0))
        print(f"Resuming {stage} from epoch {start_epoch}")
    for epoch in range(start_epoch, end_epoch + 1):
        train_metrics = _train_epoch(
            model,
            loaders["train"],
            device,
            optimizer,
            scaler,
            ema,
            int(train_cfg["accumulation_steps"]),
            float(stage_cfg.get("label_smoothing", train_cfg["label_smoothing"])),
            sample_weights=sample_weights,
            preserve_base=stage == "stage3",
            preserve_weight=float(config["residual_correction"]["base_preservation_weight"]),
            residual_l2_weight=float(config["residual_correction"]["residual_l2_weight"]),
            clip_norm=float(train_cfg["gradient_clip_norm"]),
            amp=bool(train_cfg["amp"]),
        )
        scheduler.step()
        val_metrics = evaluate(model, loaders["val"], device, ema=ema)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"{stage} epoch {epoch}/{end_epoch} | "
            f"train loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.2f}% | "
            f"val loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.2f}%"
        )
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            _save_best(model, ema, paths[f"{stage}_best"] if stage != "stage4" else paths["final"])
        save_checkpoint(
            str(checkpoint_path), model, optimizer, scheduler, scaler, ema, epoch, best_accuracy, config, stage
        )

    best_path = paths[f"{stage}_best"] if stage != "stage4" else paths["final"]
    _load_weights(model, best_path, device)
    model.set_residual_enabled(residual_enabled)
    if stage == "stage1":
        log_difficulty(
            model,
            loaders["train_eval"],
            device,
            str(paths["results"] / "sample_difficulty.csv"),
            ema=None,
        )
        log_confusion(
            model,
            loaders["val"],
            device,
            str(paths["results"] / "confusion_matrix_val.csv"),
            int(config["model"]["num_classes"]),
        )
    if stage == "stage4":
        _, final_loaders = _make_loaders(config, clean=True, include_test=True)
        test_metrics = evaluate(model, final_loaders["test"], device)
        test_confusion = log_predictions(
            model,
            final_loaders["test"],
            device,
            str(paths["results"] / "predictions.csv"),
            int(config["model"]["num_classes"]),
        )
        test_confusion.to_csv(paths["results"] / "confusion_matrix.csv", index=False)
        save_metrics(str(paths["results"] / "metrics.json"), {"best_validation": best_accuracy, "official_test": test_metrics, "history": history})
        print(f"Official test accuracy: {test_metrics['accuracy']:.2f}%")
    else:
        save_metrics(str(paths["results"] / f"{stage}_metrics.json"), {"best_validation": best_accuracy, "history": history})
    return {"stage": stage, "best_validation": best_accuracy, "path": str(best_path)}


def run_all_stages(config_path: Optional[str] = None, resume: bool = True):
    """Run the complete 4-stage workflow from one notebook entrypoint."""
    results = []
    for stage in ("stage1", "stage2", "stage3", "stage4"):
        results.append(run_stage(stage, config_path=config_path, resume=resume))
    return results


def evaluate_checkpoint(config_path: str, checkpoint_path: str):
    """Load a final checkpoint and evaluate it on official GTSRB test data."""
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, loaders = _make_loaders(config, clean=True, include_test=True)
    model = _model(config, residual_enabled=True).to(device)
    _load_weights(model, Path(checkpoint_path), device)
    model.set_residual_enabled(True)
    metrics = evaluate(model, loaders["test"], device)
    results_dir = Path(config["output"]["results"])
    confusion = log_predictions(
        model,
        loaders["test"],
        device,
        str(results_dir / "predictions.csv"),
        int(config["model"]["num_classes"]),
    )
    confusion.to_csv(results_dir / "confusion_matrix.csv", index=False)
    save_metrics(str(results_dir / "metrics.json"), {"official_test": metrics})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one RB-LNL-Ti stage")
    parser.add_argument("stage", choices=["stage1", "stage2", "stage3", "stage4"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_stage(args.stage, args.config, resume=args.resume)


if __name__ == "__main__":
    main()
