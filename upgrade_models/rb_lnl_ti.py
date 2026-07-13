"""RB-LNL-Ti wrapper.

The base implementation is intentionally imported unchanged from ``LNL.py``.
This file owns only the upgrade head and its inference/training interface.
"""

from __future__ import annotations

import torch
from torch import nn

from LNL import LNL_Ti


class RB_LNL_Ti(nn.Module):
    """LNL-Ti with a confidence-gated residual correction head.

    The gate is initialized small, so the model starts as the original base
    classifier.  During Stage 3 it learns when a residual correction is useful
    instead of applying the same correction strength to every sample.
    """

    def __init__(
        self,
        num_classes: int = 43,
        pretrained: bool = False,
        residual_hidden_dim: int = 256,
        residual_gate_hidden_dim: int = 64,
        residual_dropout: float = 0.1,
        residual_scale_init: float = -2.0,
        residual_gate_init: float = -1.5,
        residual_enabled: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.backbone = LNL_Ti(pretrained=pretrained, num_classes=num_classes, **kwargs)
        self.embed_dim = self.backbone.embed_dim
        self.num_classes = num_classes

        self.residual_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, residual_hidden_dim),
            nn.GELU(),
            nn.Dropout(residual_dropout),
            nn.Linear(residual_hidden_dim, num_classes),
        )
        self.residual_gate = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, residual_gate_hidden_dim),
            nn.GELU(),
            nn.Linear(residual_gate_hidden_dim, 1),
        )
        nn.init.constant_(self.residual_gate[-1].bias, residual_gate_init)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

        # A buffer makes the stage/inference mode part of state_dict.  A fresh
        # model defaults to the final product mode: residual correction on.
        self.register_buffer(
            "residual_enabled",
            torch.tensor(1 if residual_enabled else 0, dtype=torch.uint8),
            persistent=True,
        )

    def set_residual_enabled(self, enabled: bool) -> None:
        self.residual_enabled.fill_(1 if enabled else 0)

    def is_residual_enabled(self) -> bool:
        return bool(self.residual_enabled.item())

    def forward(self, x, vis: bool = False, return_aux: bool = False):
        features, attn_weights = self.backbone.forward_features(x)
        base_logits = self.backbone.head(features)

        if not self.is_residual_enabled():
            if return_aux:
                return base_logits, {
                    "logits": base_logits,
                    "base_logits": base_logits,
                    "residual_logits": torch.zeros_like(base_logits),
                    "gate": torch.zeros((x.size(0), 1), device=x.device),
                    "alpha": base_logits.new_zeros(()),
                }
            if vis:
                return base_logits, attn_weights
            return base_logits

        residual_logits = self.residual_head(features)
        gate = torch.sigmoid(self.residual_gate(features))
        alpha = torch.sigmoid(self.residual_scale)
        logits = base_logits + alpha * gate * residual_logits

        if return_aux:
            return logits, {
                "logits": logits,
                "base_logits": base_logits,
                "residual_logits": residual_logits,
                "gate": gate,
                "alpha": alpha,
            }
        if vis:
            return logits, attn_weights
        return logits

    def get_classifier(self):
        return self.backbone.head

    def reset_classifier(self, num_classes: int, global_pool: str = ""):
        self.backbone.reset_classifier(num_classes, global_pool)
        self.num_classes = num_classes
        self.residual_head[-1] = nn.Linear(self.residual_head[-1].in_features, num_classes)


def rb_lnl_ti(pretrained: bool = False, **kwargs) -> RB_LNL_Ti:
    return RB_LNL_Ti(pretrained=pretrained, **kwargs)
