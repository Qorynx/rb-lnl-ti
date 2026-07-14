"""Run or materialize the A0-A7 ablation matrix from the development plan.

Official GTSRB test evaluation is disabled for ablations.  Variants are
selected using validation only; run the normal final notebook once after
choosing the best variant, then evaluate the official test exactly once.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from upgrade_models.train_stage import run_all_stages


_ALL_OFF = {
    "ema_enabled": False,
    "moex_enabled": False,
    "hard_example_enabled": False,
    "residual_enabled": False,
    "pairwise_margin_enabled": False,
    "clean_calibration_enabled": False,
    "checkpoint_averaging_enabled": False,
    "tta_enabled": False,
}


def _features(**enabled: bool) -> Dict[str, bool]:
    result = dict(_ALL_OFF)
    result.update(enabled)
    return result


VARIANTS: Dict[str, Dict[str, Any]] = {
    "A0": {"features": _features()},
    "A1": {"features": _features(ema_enabled=True, moex_enabled=True)},
    "A2": {"features": _features(ema_enabled=True, moex_enabled=True, hard_example_enabled=True)},
    "A3": {
        "features": _features(
            ema_enabled=True, moex_enabled=True, hard_example_enabled=True, residual_enabled=True
        )
    },
    "A4": {
        "features": _features(
            ema_enabled=True,
            moex_enabled=True,
            hard_example_enabled=True,
            residual_enabled=True,
            pairwise_margin_enabled=True,
        )
    },
    "A5": {
        "features": _features(
            ema_enabled=True,
            moex_enabled=True,
            hard_example_enabled=True,
            residual_enabled=True,
            pairwise_margin_enabled=True,
            clean_calibration_enabled=True,
        )
    },
    "A6": {
        "features": _features(
            ema_enabled=True,
            moex_enabled=True,
            hard_example_enabled=True,
            residual_enabled=True,
            pairwise_margin_enabled=True,
            clean_calibration_enabled=True,
            checkpoint_averaging_enabled=True,
        )
    },
    "A7": {
        "features": _features(
            ema_enabled=True,
            moex_enabled=True,
            hard_example_enabled=True,
            residual_enabled=True,
            pairwise_margin_enabled=True,
            clean_calibration_enabled=True,
            tta_enabled=True,
        )
    },
}


def _deep_update(target: Dict[str, Any], update: Dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _variant_config(base: Dict[str, Any], variant: str, output_root: Path) -> Dict[str, Any]:
    config = copy.deepcopy(base)
    _deep_update(config, VARIANTS[variant])
    config["evaluation"] = {"run_official_test": False}
    config["output"] = {
        "root": str(output_root / variant),
        "results": str(output_root / variant / "results"),
    }
    return config


def _write_config(config: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def run_ablation(
    config_path: str,
    output_root: str,
    variants: Iterable[str],
    resume: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    base = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    root = Path(output_root)
    selected = list(variants)
    unknown = [variant for variant in selected if variant not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown ablation variants: {unknown}")
    manifest: Dict[str, Any] = {"official_test_used": False, "variants": {}}
    for variant in selected:
        config = _variant_config(base, variant, root)
        variant_dir = root / variant
        config_file = variant_dir / "config.yaml"
        _write_config(config, config_file)
        manifest["variants"][variant] = {
            "config": str(config_file),
            "features": config["features"],
            "status": "planned" if dry_run else "running",
        }
        if dry_run:
            continue
        run_all_stages(str(config_file), resume=resume)
        final = Path(config["output"]["root"]) / "rb_lnl_ti_gtsrb.pth"
        stage4_metrics = Path(config["output"]["results"]) / "stage4_metrics.json"
        metrics = json.loads(stage4_metrics.read_text(encoding="utf-8")) if stage4_metrics.exists() else {}
        manifest["variants"][variant].update(
            {"status": "completed", "checkpoint": str(final), "validation": metrics.get("best_validation")}
        )
    root.mkdir(parents=True, exist_ok=True)
    (root / "ablation_plan.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RB-LNL-Ti A0-A7 ablation matrix")
    parser.add_argument("--config", default="upgrade_models/config.yaml")
    parser.add_argument("--output", default="./submission/ablations")
    parser.add_argument("--variants", default=",".join(VARIANTS), help="Comma-separated variants, e.g. A0,A1,A2")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write variant configs without training")
    args = parser.parse_args()
    result = run_ablation(args.config, args.output, [item.strip() for item in args.variants.split(",")], args.resume, args.dry_run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
