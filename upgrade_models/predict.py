"""Standalone inference for new GTSRB images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main() -> None:
    parser = argparse.ArgumentParser(description="Predict GTSRB labels for new images")
    parser.add_argument("images", nargs="+", help="Image paths")
    parser.add_argument("--checkpoint", default="./submission/rb_lnl_ti_gtsrb.pth")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    from upgrade_models.rb_lnl_ti import RB_LNL_Ti

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    model = RB_LNL_Ti(num_classes=43).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.eval()
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
    )

    results = []
    with torch.no_grad():
        for image_path in args.images:
            image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
            probabilities = model(image).softmax(dim=1)[0]
            values, indices = probabilities.topk(3)
            results.append(
                {
                    "image": str(image_path),
                    "predicted_label": int(indices[0]),
                    "confidence": float(values[0]),
                    "top3": [{"label": int(label), "probability": float(value)} for value, label in zip(values, indices)],
                }
            )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
