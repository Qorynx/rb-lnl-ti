"""Export a trained RB-LNL-Ti checkpoint as a clean plug-and-play artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RB-LNL-Ti model and pure state_dict checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Stage 4 .pth or .pt checkpoint")
    parser.add_argument("--output", default="./submission", help="Directory for the submission artifact")
    args = parser.parse_args()
    from upgrade_models.rb_lnl_ti import RB_LNL_Ti

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    output.mkdir(parents=True, exist_ok=True)

    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model = RB_LNL_Ti(num_classes=43)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    exported_checkpoint = output / "rb_lnl_ti_gtsrb.pth"
    torch.save(model.state_dict(), exported_checkpoint)
    source_upgrade = Path(__file__).resolve().parent
    source_root = source_upgrade.parent
    for filename in ("LNL.py", "LNL_MoEx.py"):
        source = source_root / filename
        if source.exists():
            shutil.copy2(source, output / filename)
    if (source_root / "models").exists():
        shutil.copytree(
            source_root / "models",
            output / "models",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

    for filename in ("rb_lnl_ti.py", "requirements.txt", "README.md"):
        source = source_upgrade / filename
        if source.exists():
            shutil.copy2(source, output / filename)
    shutil.copy2(source_upgrade / "config.yaml", output / "config.yaml")

    # Keep the complete upgrade package so the supplied notebook can train or
    # evaluate from the exported directory, while the root model file remains
    # directly importable by the upstream Instructions notebook.
    shutil.copytree(
        source_upgrade,
        output / "upgrade_models",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    notebook = Path(__file__).with_name("Instructions_RB_LNL_Ti.ipynb")
    if notebook.exists():
        shutil.copy2(notebook, output / notebook.name)
    results_source = checkpoint_path.parent / "results"
    if results_source.exists():
        shutil.copytree(results_source, output / "results", dirs_exist_ok=True)

    manifest = {
        "model": "RB-LNL-Ti",
        "checkpoint": exported_checkpoint.name,
        "sha256": _sha256(exported_checkpoint),
        "input_shape": ["B", 3, 224, 224],
        "output_shape": ["B", 43],
        "checkpoint_format": "plain PyTorch state_dict",
        "residual_enabled_by_default": True,
    }
    (output / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
