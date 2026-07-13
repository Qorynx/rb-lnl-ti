# RB-LNL-Ti upgrade pipeline

This folder contains the upgrade only. The original implementation in
`models/`, `LNL.py`, and `LNL_MoEx.py` is not modified.

## What changed

- The official GTSRB test split is reserved for the final evaluation.
- A deterministic stratified train/validation split is used for all decisions.
- Stage 1 logs difficulty for train samples and confusion for validation only.
- Stage 2 applies clipped hard-example weights with correct original indices.
- Stage 3 uses a confidence-gated residual head and a small base-preservation
  loss instead of a hard-coded confusion-pair loss.
- EMA, AMP, gradient accumulation, warm-up, checkpoint resume, and artifact
  logging are shared across stages.
- A fresh `RB_LNL_Ti()` uses residual correction by default and returns
  `[B, 43]` logits from `model(images)`.

The repository intentionally does not contain a pretrained `.pth` before a
successful Stage 4 run. A checkpoint without an actual training run would not
be a valid accuracy result. Generate the verified artifact with
`export_submission.py` after Stage 4.

## Install

From the repository root:

```powershell
pip install -r upgrade_models/requirements.txt
```

The original repository's compatible `timm` version may be required because
the untouched base files use the legacy timm import paths.

## Colab workflow

1. Push the repository to GitHub.
2. In Colab, clone it once per runtime:

```python
!git clone https://github.com/<username>/<repository>.git /content/rb-lnl-ti
%cd /content/rb-lnl-ti
```

3. Open and run `upgrade_models/Instructions_RB_LNL_Ti.ipynb`.
4. Leave `RUN_MODE = 'AUTO'`: it trains all four stages when the final
   checkpoint is absent, and evaluates it when the checkpoint is present.
   The unified notebook is the only notebook needed for the normal flow.

If a runtime is interrupted, reopen the same notebook with `RUN_MODE = 'AUTO'`.
The notebook calls all stages with resume enabled, so it continues from the
latest checkpoint. Do not commit the Drive-backed `data/` or `submission/`
directories to GitHub.

## Run one stage from Python (advanced/debug only)

```powershell
python -m upgrade_models.train_stage stage1 --config upgrade_models/config.yaml
```

To continue an interrupted stage:

```powershell
python -m upgrade_models.train_stage stage2 --config upgrade_models/config.yaml --resume
```

The scripts write checkpoints under `submission/` and analysis artifacts under
`submission/results/`. The official test is evaluated only at the end of Stage
4 and is written to `submission/results/metrics.json`, together with
`predictions.csv` and the final `confusion_matrix.csv`.

## Export a plug-and-play model

After Stage 4, export a clean model artifact from the Drive checkpoint:

```powershell
python upgrade_models/export_submission.py `
  --checkpoint /content/drive/MyDrive/rb-lnl-ti/submission/rb_lnl_ti_gtsrb.pth `
  --output /content/drive/MyDrive/rb-lnl-ti/submission_export
```

The export contains `rb_lnl_ti.py`, a pure `rb_lnl_ti_gtsrb.pth`
`state_dict`, a manifest, and `Instructions_RB_LNL_Ti.ipynb`.

The original upstream notebook creates a base `LNL_Ti` and replaces its head.
For RB-LNL-Ti, replace that model cell with:

```python
from upgrade_models.rb_lnl_ti import RB_LNL_Ti

model = RB_LNL_Ti(num_classes=43)
state = torch.load(
    "/content/drive/MyDrive/rb-lnl-ti/submission/rb_lnl_ti_gtsrb.pth",
    map_location="cpu",
)
model.load_state_dict(state, strict=True)
model = model.cuda().eval()
```

The existing evaluation loop can continue to call `outputs = model(images)`.
The input transform must include the same `Normalize([0.5] * 3, [0.5] * 3)`
used during training. See `Instructions_RB_LNL_Ti.ipynb` for the complete
verification flow.

## Direct inference

```python
import torch
from upgrade_models.rb_lnl_ti import RB_LNL_Ti

model = RB_LNL_Ti(num_classes=43)
model.load_state_dict(torch.load("submission/rb_lnl_ti_gtsrb.pth", map_location="cpu"))
model.eval()
logits = model(images)  # [B, 43]
```

For new image files:

```powershell
python upgrade_models/predict.py path\to\sign.jpg
```

With the Colab/Drive checkpoint:

```powershell
python upgrade_models/predict.py path/to/sign.jpg --checkpoint /content/drive/MyDrive/rb-lnl-ti/submission/rb_lnl_ti_gtsrb.pth
```
