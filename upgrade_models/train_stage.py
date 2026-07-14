"""Implementation of the four reproducible RB-LNL-Ti training stages."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
import pandas as pd
from torch import nn, optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from upgrade_models.pipeline import (
    ModelEMA,
    autocast_context,
    build_gtsrb_splits,
    cosine_with_warmup,
    average_state_dicts,
    evaluate,
    confusion_pairs_from_matrix,
    load_confusion_pairs,
    load_checkpoint,
    load_config,
    log_confusion,
    log_difficulty,
    log_predictions,
    make_loader,
    moment_exchange,
    make_scaler,
    pairwise_margin_loss,
    save_confusion_pairs,
    save_confusion_plot,
    sample_weights_from_csv,
    save_checkpoint,
    save_metrics,
    save_split_manifest,
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
        "stage4_best": root / "rb_lnl_ti_stage4_best.pth",
        "final": root / "rb_lnl_ti_gtsrb.pth",
        "averaged": root / "rb_lnl_ti_gtsrb_avg.pth",
        "candidate_manifest": root / "stage4_candidates.json",
    }


def _features(config: Dict[str, Any]) -> Dict[str, bool]:
    defaults = {
        "ema_enabled": True,
        "moex_enabled": True,
        "hard_example_enabled": True,
        "residual_enabled": True,
        "pairwise_margin_enabled": True,
        "clean_calibration_enabled": True,
        "checkpoint_averaging_enabled": False,
        "tta_enabled": False,
    }
    defaults.update(config.get("features", {}))
    return {key: bool(value) for key, value in defaults.items()}


class BaseLNLAdapter(nn.Module):
    """Training adapter for a true LNL-Ti baseline ablation.

    The untouched base model remains unchanged; this adapter only supplies the
    same ``return_aux`` and optional MoEx contract as RB-LNL-Ti.
    """

    def __init__(self, num_classes: int, drop_path_rate: float):
        super().__init__()
        from LNL import LNL_Ti

        self.backbone = LNL_Ti(
            pretrained=False,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        )

    def set_residual_enabled(self, enabled: bool) -> None:
        del enabled

    def forward(self, x, vis: bool = False, return_aux: bool = False, moex_strength: float = 0.0, moex_permutation=None):
        features, attn_weights = self.backbone.forward_features(x)
        if self.training and moex_strength > 0:
            features = moment_exchange(features, moex_strength, moex_permutation)
        logits = self.backbone.head(features)
        if return_aux:
            return logits, {
                "logits": logits,
                "base_logits": logits,
                "residual_logits": torch.zeros_like(logits),
                "gate": torch.zeros((x.size(0), 1), device=x.device),
                "alpha": logits.new_zeros(()),
                "features": features,
                "raw_features": features,
            }
        if vis:
            return logits, attn_weights
        return logits


def _model(config: Dict[str, Any], residual_enabled: bool):
    # Import lazily so --help/config inspection works before the legacy timm
    # dependency is installed.
    model_cfg = config["model"]
    fixed_base = {"image_size": 224, "embed_dim": 192, "depth": 12}
    for key, expected in fixed_base.items():
        if int(model_cfg[key]) != expected:
            raise ValueError(f"{key}={model_cfg[key]} is incompatible with the untouched LNL-Ti base; expected {expected}")
    features = _features(config)
    if not features["residual_enabled"]:
        return BaseLNLAdapter(model_cfg["num_classes"], model_cfg["drop_path_rate"])
    from upgrade_models.rb_lnl_ti import RB_LNL_Ti

    residual_enabled = bool(residual_enabled)
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
        group_manifest=data_cfg.get("group_manifest"),
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


def _moex_strength(config: Dict[str, Any], stage: str, epoch: int) -> float:
    """Decay MoEx over Stage 1; all later stages use clean features."""
    features = _features(config)
    cfg = config.get("moex", {})
    if stage != "stage1" or not features["moex_enabled"] or not bool(cfg.get("enabled", True)):
        return 0.0
    stage_cfg = config["stages"][stage]
    start = int(stage_cfg["start_epoch"])
    end = int(stage_cfg["end_epoch"])
    disable_last = int(cfg.get("disable_last_epochs", 0))
    if epoch >= end - max(0, disable_last) + 1:
        return 0.0
    progress = (epoch - start) / max(1, end - start)
    start_probability = float(cfg.get("probability_start", 0.6))
    end_probability = float(cfg.get("probability_end", 0.0))
    probability = start_probability + (end_probability - start_probability) * max(0.0, min(1.0, progress))
    if torch.rand(()) > probability:
        return 0.0
    return float(cfg.get("lambda", 0.9))


def _optimizer_for_stage(model: nn.Module, config: Dict[str, Any], stage: str):
    train_cfg = config["train"]
    stage_cfg = config["stages"][stage]
    if stage != "stage3":
        lr = float(stage_cfg["learning_rate"])
        params = [parameter for parameter in model.parameters() if parameter.requires_grad]
        return optim.AdamW(params, lr=lr, weight_decay=float(train_cfg["weight_decay"]))

    residual_enabled = _features(config)["residual_enabled"]
    for parameter in model.parameters():
        parameter.requires_grad = False
    late, base_head, residual = [], [], []
    for name, parameter in model.named_parameters():
        if residual_enabled and ("residual_head" in name or "residual_gate" in name or "residual_scale" in name):
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
    groups = [
        {"params": late, "lr": float(stage_cfg["late_backbone_learning_rate"])},
        {"params": base_head, "lr": float(stage_cfg["base_head_learning_rate"])},
    ]
    if residual:
        groups.append({"params": residual, "lr": float(stage_cfg["residual_learning_rate"])})
    return optim.AdamW(
        groups,
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
    confusion_pairs: Optional[Dict[int, list[int]]] = None,
    pairwise_weight: float = 0.05,
    pairwise_margin: float = 0.2,
    moex_strength: float = 0.0,
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
            logits, aux = model(images, return_aux=True, moex_strength=moex_strength)
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
            if confusion_pairs and pairwise_weight > 0:
                loss = loss + pairwise_weight * pairwise_margin_loss(
                    logits, targets, confusion_pairs, margin=pairwise_margin
                )
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


def _update_stage4_candidates(
    model: nn.Module,
    ema: Optional[ModelEMA],
    paths: Dict[str, Path],
    epoch: int,
    score: float,
    top_k: int,
) -> None:
    """Keep only the top-k validation checkpoints for optional averaging."""
    manifest_path = paths["candidate_manifest"]
    if manifest_path.exists():
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        entries = []
    entries = [entry for entry in entries if Path(entry["path"]).exists()]
    candidate_path = paths["root"] / f"stage4_candidate_{epoch}.pth"
    _save_best(model, ema, candidate_path)
    entries.append({"epoch": int(epoch), "score": float(score), "path": str(candidate_path)})
    entries.sort(key=lambda entry: float(entry["score"]), reverse=True)
    for removed in entries[top_k:]:
        removed_path = Path(removed["path"])
        if removed_path != candidate_path and removed_path.exists():
            removed_path.unlink()
    entries = entries[:top_k]
    manifest_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def _candidate_paths(paths: Dict[str, Path]) -> list[Path]:
    manifest_path = paths["candidate_manifest"]
    if not manifest_path.exists():
        return []
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [Path(entry["path"]) for entry in entries if Path(entry["path"]).exists()]


def _stage_epochs(config: Dict[str, Any], stage: str):
    stage_cfg = config["stages"][stage]
    return int(stage_cfg["start_epoch"]), int(stage_cfg["end_epoch"])


def run_stage(stage: str, config_path: Optional[str] = None, resume: bool = False) -> Dict[str, Any]:
    config = load_config(config_path)
    features = _features(config)
    set_seed(int(config["seed"]))
    paths = _paths(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_epoch, end_epoch = _stage_epochs(config, stage)
    if start_epoch < 0 or end_epoch < start_epoch or end_epoch >= int(config["train"]["epochs"]):
        raise ValueError("config train.epochs and stage epoch ranges are inconsistent")
    required_inputs = {
        "stage2": [paths["stage1_best"], paths["results"] / "sample_difficulty.csv"],
        "stage3": [paths["stage2_best"], paths["results"] / "confusion_pairs.json"]
        if features["pairwise_margin_enabled"]
        else [paths["stage2_best"]],
        "stage4": [paths["stage3_best"]],
    }.get(stage, [])
    missing = [str(path) for path in required_inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{stage} cannot start; missing input artifacts: {missing}")
    clean = stage == "stage4" and features["clean_calibration_enabled"]
    residual_enabled = stage in {"stage3", "stage4"} and features["residual_enabled"]
    splits, loaders = _make_loaders(config, clean=clean, include_test=False)
    save_split_manifest(splits, str(paths["results"] / "split_manifest.json"))
    model = _model(config, residual_enabled=residual_enabled).to(device)

    if stage == "stage2":
        _load_weights(model, paths["stage1_best"], device)
        model.set_residual_enabled(False)
    elif stage == "stage3":
        _load_weights(model, paths["stage2_best"], device)
        model.set_residual_enabled(residual_enabled)
    elif stage == "stage4":
        _load_weights(model, paths["stage3_best"], device)
        model.set_residual_enabled(residual_enabled)

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
    ema = (
        ModelEMA(model, float(config["ema"]["decay"]))
        if config["ema"]["enabled"] and features["ema_enabled"]
        else None
    )

    sample_weights = None
    if stage == "stage2" and features["hard_example_enabled"]:
        sample_weights = sample_weights_from_csv(
            str(paths["results"] / "sample_difficulty.csv"),
            int(splits["full_train_size"]),
            float(config["hard_example"]["alpha"]),
            float(config["hard_example"]["max_weight"]),
        )

    confusion_pairs = None
    if stage == "stage3" and features["pairwise_margin_enabled"]:
        confusion_pairs = load_confusion_pairs(str(paths["results"] / "confusion_pairs.json"))
    pairwise_cfg = config.get("pairwise_margin", {})

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
            preserve_base=stage == "stage3" and features["residual_enabled"],
            preserve_weight=float(config["residual_correction"]["base_preservation_weight"]),
            residual_l2_weight=float(config["residual_correction"]["residual_l2_weight"]),
            confusion_pairs=confusion_pairs,
            pairwise_weight=float(pairwise_cfg.get("weight", 0.05))
            if features["pairwise_margin_enabled"]
            else 0.0,
            pairwise_margin=float(pairwise_cfg.get("margin", 0.2)),
            moex_strength=_moex_strength(config, stage, epoch),
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
            _save_best(model, ema, paths[f"{stage}_best"] if stage != "stage4" else paths["stage4_best"])
        if stage == "stage4" and features["checkpoint_averaging_enabled"]:
            _update_stage4_candidates(
                model,
                ema,
                paths,
                epoch,
                val_metrics["accuracy"],
                int(config.get("checkpoint_averaging", {}).get("top_k", 3)),
            )
        save_checkpoint(
            str(checkpoint_path), model, optimizer, scheduler, scaler, ema, epoch, best_accuracy, config, stage
        )

    if stage == "stage4":
        if features["checkpoint_averaging_enabled"]:
            candidates = _candidate_paths(paths)
            if candidates:
                torch.save(average_state_dicts(candidates), paths["averaged"])
                shutil.copy2(paths["averaged"], paths["final"])
        elif paths["stage4_best"].exists():
            shutil.copy2(paths["stage4_best"], paths["final"])
        elif paths["final"].exists():
            shutil.copy2(paths["final"], paths["stage4_best"])
        best_path = paths["final"] if paths["final"].exists() else paths["stage4_best"]
    else:
        best_path = paths[f"{stage}_best"]
    _load_weights(model, best_path, device)
    model.set_residual_enabled(residual_enabled)
    if stage == "stage1":
        train_difficulty = log_difficulty(
            model,
            loaders["train_eval"],
            device,
            path=None,
            ema=None,
            split="train",
        )
        val_difficulty = log_difficulty(
            model,
            loaders["val"],
            device,
            path=None,
            ema=None,
            split="validation",
        )
        pd.concat([train_difficulty, val_difficulty], ignore_index=True).sort_values("image_id").to_csv(
            paths["results"] / "sample_difficulty.csv", index=False
        )
        confusion_path = paths["results"] / "confusion_matrix_val.csv"
        confusion_frame = log_confusion(
            model,
            loaders["val"],
            device,
            str(confusion_path),
            int(config["model"]["num_classes"]),
        )
        confusion_pairs = confusion_pairs_from_matrix(
            confusion_frame,
            max_pairs=int(pairwise_cfg.get("max_pairs", 32)),
            min_count=int(pairwise_cfg.get("min_count", 1)),
        )
        save_confusion_pairs(str(paths["results"] / "confusion_pairs.json"), confusion_pairs)
    if stage == "stage4" and bool(config.get("evaluation", {}).get("run_official_test", True)):
        _, final_loaders = _make_loaders(config, clean=True, include_test=True)
        test_metrics = evaluate(model, final_loaders["test"], device, tta=False)
        test_confusion = log_predictions(
            model,
            final_loaders["test"],
            device,
            str(paths["results"] / "predictions.csv"),
            int(config["model"]["num_classes"]),
        )
        test_confusion.to_csv(paths["results"] / "confusion_matrix.csv", index=False)
        save_confusion_plot(test_confusion, str(paths["results"] / "test_result.png"), "RB-LNL-Ti official test confusion matrix")
        report = {
            "best_validation": best_accuracy,
            "official_test": test_metrics,
            "official_test_single_view": test_metrics,
            "history": history,
        }
        if features["tta_enabled"]:
            report["official_test_tta"] = evaluate(model, final_loaders["test"], device, tta=True)
            log_predictions(
                model,
                final_loaders["test"],
                device,
                str(paths["results"] / "predictions_tta.csv"),
                int(config["model"]["num_classes"]),
                tta=True,
            )
        save_metrics(str(paths["results"] / "metrics.json"), report)
        print(f"Official test accuracy: {test_metrics['accuracy']:.2f}%")
    elif stage != "stage4" or not bool(config.get("evaluation", {}).get("run_official_test", True)):
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
    features = _features(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, loaders = _make_loaders(config, clean=True, include_test=True)
    model = _model(config, residual_enabled=True).to(device)
    _load_weights(model, Path(checkpoint_path), device)
    model.set_residual_enabled(True)
    metrics = evaluate(model, loaders["test"], device, tta=False)
    results_dir = Path(config["output"]["results"])
    confusion = log_predictions(
        model,
        loaders["test"],
        device,
        str(results_dir / "predictions.csv"),
        int(config["model"]["num_classes"]),
    )
    confusion.to_csv(results_dir / "confusion_matrix.csv", index=False)
    save_confusion_plot(confusion, str(results_dir / "test_result.png"), "RB-LNL-Ti official test confusion matrix")
    report = {"official_test": metrics, "official_test_single_view": metrics}
    if features["tta_enabled"]:
        report["official_test_tta"] = evaluate(model, loaders["test"], device, tta=True)
    save_metrics(str(results_dir / "metrics.json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one RB-LNL-Ti stage")
    parser.add_argument("stage", choices=["stage1", "stage2", "stage3", "stage4"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_stage(args.stage, args.config, resume=args.resume)


if __name__ == "__main__":
    main()
