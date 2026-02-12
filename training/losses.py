"""Loss functions for MTR-Lite training.

Combines:
1. Classification loss (CE): which intention query best matches GT
2. Regression loss (Smooth L1): trajectory accuracy for best-matching query
3. Deep supervision: weighted sum across decoder layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MTRLiteLoss(nn.Module):
    """Combined classification + regression loss with deep supervision.

    For each target agent:
    1. Find the intention query whose predicted endpoint is closest to GT endpoint
    2. Classification: CE loss over intention query scores (label = best query)
    3. Regression: Smooth L1 between best query's trajectory and GT

    Deep supervision applies this at every decoder layer with specified weights.

    Args:
        cls_weight: weight for classification loss
        reg_weight: weight for regression loss
        deep_supervision_weights: list of per-layer loss weights (should sum ~1.0)
    """

    def __init__(
        self,
        cls_weight: float = 1.0,
        reg_weight: float = 1.0,
        deep_supervision_weights: list = None,
    ):
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.deep_supervision_weights = deep_supervision_weights or [0.2, 0.2, 0.2, 0.4]

    def forward(
        self,
        layer_trajectories: list,
        layer_scores: list,
        target_future: torch.Tensor,
        target_future_valid: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> dict:
        """Compute training loss.

        Args:
            layer_trajectories: list of (B, T, K, future_len, 2) per decoder layer
            layer_scores: list of (B, T, K) per decoder layer
            target_future: (B, T, future_len, 2) GT trajectories
            target_future_valid: (B, T, future_len) bool validity
            target_mask: (B, T) bool which target slots are valid

        Returns:
            dict with 'total_loss', 'cls_loss', 'reg_loss', per-layer losses
        """
        device = target_future.device
        B, T = target_mask.shape
        num_layers = len(layer_trajectories)

        # Ensure weights match number of layers
        weights = self.deep_supervision_weights
        if len(weights) < num_layers:
            weights = weights + [weights[-1]] * (num_layers - len(weights))
        weights = weights[:num_layers]
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]  # normalize

        total_cls = torch.tensor(0.0, device=device)
        total_reg = torch.tensor(0.0, device=device)
        layer_losses = []
        n_valid = 0

        for layer_i in range(num_layers):
            pred_trajs = layer_trajectories[layer_i]  # (B, T, K, future_len, 2)
            pred_scores = layer_scores[layer_i]        # (B, T, K)
            K = pred_trajs.shape[2]

            layer_cls = torch.tensor(0.0, device=device)
            layer_reg = torch.tensor(0.0, device=device)

            for b in range(B):
                for t in range(T):
                    if not target_mask[b, t]:
                        continue

                    gt_traj = target_future[b, t]           # (future_len, 2)
                    gt_valid = target_future_valid[b, t]     # (future_len,)
                    preds = pred_trajs[b, t]                 # (K, future_len, 2)
                    scores = pred_scores[b, t]               # (K,)

                    if not gt_valid.any():
                        continue

                    # Find best-matching intention by endpoint distance
                    gt_endpoint = gt_traj[-1]  # (2,)
                    pred_endpoints = preds[:, -1, :]  # (K, 2)
                    endpoint_dists = torch.norm(
                        pred_endpoints - gt_endpoint.unsqueeze(0), dim=1
                    )  # (K,)
                    best_k = endpoint_dists.argmin()

                    # Classification loss
                    cls_loss = F.cross_entropy(scores.unsqueeze(0), best_k.unsqueeze(0))
                    layer_cls = layer_cls + cls_loss

                    # Regression loss (only on valid future frames)
                    best_traj = preds[best_k]  # (future_len, 2)
                    diff = best_traj - gt_traj  # (future_len, 2)
                    reg_loss = F.smooth_l1_loss(
                        best_traj[gt_valid], gt_traj[gt_valid], reduction="mean",
                    )
                    layer_reg = layer_reg + reg_loss

                    if layer_i == num_layers - 1:
                        n_valid += 1

            # Average over valid targets
            if n_valid > 0 or layer_i < num_layers - 1:
                denom = max(n_valid, 1)
                layer_cls = layer_cls / denom
                layer_reg = layer_reg / denom

            w = weights[layer_i]
            total_cls = total_cls + w * layer_cls
            total_reg = total_reg + w * layer_reg
            layer_losses.append((layer_cls.item(), layer_reg.item()))

        cls_loss = self.cls_weight * total_cls
        reg_loss = self.reg_weight * total_reg
        total_loss = cls_loss + reg_loss

        return {
            "total_loss": total_loss,
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
            "layer_losses": layer_losses,
            "n_valid_targets": n_valid,
        }
