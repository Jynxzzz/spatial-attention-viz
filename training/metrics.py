"""Evaluation metrics for trajectory prediction.

Standard Waymo metrics:
- minADE@K: minimum Average Displacement Error across K modes
- minFDE@K: minimum Final Displacement Error across K modes
- MR@K: Miss Rate (fraction where minFDE > threshold)

Per-type breakdown: vehicle, pedestrian, cyclist (matching Waymo evaluation).
"""

import torch
import numpy as np

# Must match data/agent_features.py AGENT_TYPES
AGENT_TYPE_NAMES = ["vehicle", "pedestrian", "cyclist", "other", "unknown"]


def compute_ade(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Compute ADE (Average Displacement Error) for one trajectory.

    Args:
        pred: (future_len, 2) predicted trajectory
        gt: (future_len, 2) ground truth trajectory
        valid: (future_len,) bool validity mask

    Returns:
        scalar ADE value
    """
    if not valid.any():
        return torch.tensor(0.0, device=pred.device)
    errors = torch.norm(pred[valid] - gt[valid], dim=1)  # (n_valid,)
    return errors.mean()


def compute_fde(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Compute FDE (Final Displacement Error).

    Uses the last valid frame's error.
    """
    if not valid.any():
        return torch.tensor(0.0, device=pred.device)
    last_valid = valid.nonzero()[-1].item()
    return torch.norm(pred[last_valid] - gt[last_valid])


def compute_min_ade_fde(
    pred_trajs: torch.Tensor,
    gt_traj: torch.Tensor,
    gt_valid: torch.Tensor,
    miss_threshold: float = 2.0,
) -> dict:
    """Compute minADE, minFDE, and miss rate across K modes.

    Args:
        pred_trajs: (K, future_len, 2) multi-modal predictions
        gt_traj: (future_len, 2) ground truth
        gt_valid: (future_len,) bool mask
        miss_threshold: FDE threshold for miss rate (meters)

    Returns:
        dict with min_ade, min_fde, miss (bool)
    """
    K = pred_trajs.shape[0]
    ades = []
    fdes = []

    for k in range(K):
        ades.append(compute_ade(pred_trajs[k], gt_traj, gt_valid))
        fdes.append(compute_fde(pred_trajs[k], gt_traj, gt_valid))

    ades = torch.stack(ades)
    fdes = torch.stack(fdes)

    best_k = ades.argmin()

    return {
        "min_ade": ades[best_k].item(),
        "min_fde": fdes[best_k].item(),
        "miss": fdes[best_k].item() > miss_threshold,
    }


def compute_horizon_ade(
    pred_trajs: torch.Tensor,
    gt_traj: torch.Tensor,
    gt_valid: torch.Tensor,
    horizons: dict = None,
) -> dict:
    """Compute minADE at multiple time horizons.

    Args:
        pred_trajs: (K, future_len, 2)
        gt_traj: (future_len, 2)
        gt_valid: (future_len,) bool
        horizons: dict of {name: timestep_index}, default 3s/5s/8s

    Returns:
        dict with ade at each horizon
    """
    if horizons is None:
        horizons = {"3s": 30, "5s": 50, "8s": 80}

    result = {}
    K = pred_trajs.shape[0]

    for name, t_end in horizons.items():
        t_end = min(t_end, gt_traj.shape[0])
        best_ade = float("inf")
        for k in range(K):
            valid_slice = gt_valid[:t_end]
            if not valid_slice.any():
                best_ade = 0.0
                break
            errors = torch.norm(
                pred_trajs[k, :t_end][valid_slice] - gt_traj[:t_end][valid_slice],
                dim=1,
            )
            ade = errors.mean().item()
            best_ade = min(best_ade, ade)
        result[f"ade_{name}"] = best_ade

    return result


class MetricAggregator:
    """Accumulates metrics across batches for epoch-level reporting.

    Tracks both overall metrics and per-agent-type breakdown
    (vehicle, pedestrian, cyclist) matching Waymo evaluation protocol.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.ades = []
        self.fdes = []
        self.misses = []
        self.horizon_ades = {}
        # Per-type tracking: {type_idx: {"ades": [], "fdes": [], "misses": []}}
        self.per_type = {}

    def update(
        self,
        pred_trajectories: torch.Tensor,
        pred_scores: torch.Tensor,
        target_future: torch.Tensor,
        target_future_valid: torch.Tensor,
        target_mask: torch.Tensor,
        target_agent_types: torch.Tensor = None,
    ):
        """Update metrics with a batch of predictions.

        Args:
            pred_trajectories: (B, T, M, future_len, 2)
            pred_scores: (B, T, M)
            target_future: (B, T, future_len, 2)
            target_future_valid: (B, T, future_len)
            target_mask: (B, T)
            target_agent_types: (B, T) int64, 0=vehicle,1=ped,2=cyc. None = skip per-type.
        """
        B, T = target_mask.shape

        for b in range(B):
            for t in range(T):
                if not target_mask[b, t]:
                    continue

                pred = pred_trajectories[b, t]       # (M, future_len, 2)
                gt = target_future[b, t]             # (future_len, 2)
                valid = target_future_valid[b, t]    # (future_len,)

                if not valid.any():
                    continue

                metrics = compute_min_ade_fde(pred, gt, valid)
                self.ades.append(metrics["min_ade"])
                self.fdes.append(metrics["min_fde"])
                self.misses.append(metrics["miss"])

                h_metrics = compute_horizon_ade(pred, gt, valid)
                for key, val in h_metrics.items():
                    if key not in self.horizon_ades:
                        self.horizon_ades[key] = []
                    self.horizon_ades[key].append(val)

                # Per-type tracking
                if target_agent_types is not None:
                    type_idx = target_agent_types[b, t].item()
                    if type_idx >= 0:
                        if type_idx not in self.per_type:
                            self.per_type[type_idx] = {"ades": [], "fdes": [], "misses": []}
                        self.per_type[type_idx]["ades"].append(metrics["min_ade"])
                        self.per_type[type_idx]["fdes"].append(metrics["min_fde"])
                        self.per_type[type_idx]["misses"].append(metrics["miss"])

    def compute(self) -> dict:
        """Compute aggregated metrics including per-type breakdown."""
        if not self.ades:
            return {"minADE@6": 0, "minFDE@6": 0, "MR@6": 0}

        result = {
            "minADE@6": np.mean(self.ades),
            "minFDE@6": np.mean(self.fdes),
            "MR@6": np.mean(self.misses),
        }

        for key, vals in self.horizon_ades.items():
            result[key] = np.mean(vals)

        # Per-type metrics
        for type_idx, data in self.per_type.items():
            if not data["ades"]:
                continue
            type_name = AGENT_TYPE_NAMES[type_idx] if type_idx < len(AGENT_TYPE_NAMES) else f"type{type_idx}"
            n = len(data["ades"])
            result[f"{type_name}/minADE@6"] = np.mean(data["ades"])
            result[f"{type_name}/minFDE@6"] = np.mean(data["fdes"])
            result[f"{type_name}/MR@6"] = np.mean(data["misses"])
            result[f"{type_name}/count"] = n

        return result
