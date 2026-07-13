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

3. Open and run `upgrade_models/00_setup_environment.ipynb`.
4. Run the four stage notebooks in order. They use `config_colab.yaml`, which
   stores data, checkpoints and results under Google Drive.

If a runtime is interrupted, rerun setup and add `--resume` to the stage
command that was interrupted. Do not commit the Drive-backed `data/` or
`submission/` directories to GitHub.

## Run

```powershell
python upgrade_models/train_colab.py
python upgrade_models/train_colab_stage2.py
python upgrade_models/train_colab_stage3.py
python upgrade_models/train_colab_stage4.py
```

To continue an interrupted stage:

```powershell
python upgrade_models/train_colab_stage2.py --resume
```

The scripts write checkpoints under `submission/` and analysis artifacts under
`submission/results/`. The official test is evaluated only at the end of Stage
4 and is written to `submission/results/metrics.json`, together with
`predictions.csv` and the final `confusion_matrix.csv`.

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
