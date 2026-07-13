import torch
import torch.nn as nn
from models.tnt import TNT

# Import the original LNL_Ti factory function
from LNL import LNL_Ti

class RB_LNL_Ti(nn.Module):
    def __init__(self, num_classes=43, pretrained=False, **kwargs):
        """
        Residual-Boosted Locality-iN-Locality Tiny (RB-LNL-Ti)
        
        Args:
            num_classes (int): Number of target classes. GTSRB uses 43.
            pretrained (bool): Whether to load pre-trained weights for the backbone.
        """
        super().__init__()
        
        # Load the base backbone from original LNL_Ti
        self.backbone = LNL_Ti(pretrained=pretrained, num_classes=num_classes, **kwargs)
        
        # LNL_Ti embed_dim is 192 (from tnt_t_conv_patch16_224 config)
        self.embed_dim = self.backbone.embed_dim
        
        # The base head is already initialized in the backbone (self.backbone.head)
        # It handles 192 -> 43 prediction.
        
        # Residual correction head
        self.residual_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        # Residual scale initialized to -2.0 as per the plan
        self.residual_scale = nn.Parameter(torch.tensor(-2.0))
        
        # Flag to control if residual head is used during forward pass
        self.use_residual = False

    def forward(self, x, vis=False):
        """
        Forward pass.
        
        If self.use_residual is False, only base_logits are returned.
        If self.use_residual is True, logits = base_logits + alpha * residual_logits.
        """
        # forward_features in TNT returns (features, attn_weights)
        features, attn_weights = self.backbone.forward_features(x)
        
        # Base prediction
        base_logits = self.backbone.head(features)
        
        if not self.use_residual:
            if vis:
                return base_logits, attn_weights
            return base_logits
            
        # Residual prediction
        residual_logits = self.residual_head(features)
        alpha = torch.sigmoid(self.residual_scale)
        
        # Final prediction
        logits = base_logits + alpha * residual_logits
        
        if vis:
            return logits, attn_weights
        return logits

    def get_classifier(self):
        return self.backbone.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.backbone.reset_classifier(num_classes, global_pool)
        
        # We also need to reset the residual head's last linear layer
        self.residual_head[-1] = nn.Linear(256, num_classes)

def rb_lnl_ti(pretrained=False, **kwargs):
    """
    Factory function for RB-LNL-Ti model.
    """
    model = RB_LNL_Ti(pretrained=pretrained, **kwargs)
    return model
