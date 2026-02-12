"""PointNet-style polyline encoder for agents and map lanes.

Architecture per polyline:
  MLP(feat_dim -> 64 -> 128 -> 256) applied per-point
  -> max_pool over points
  -> MLP(256 -> 256 -> 256) post-aggregation
  -> output: (batch, d_model) token embedding

Shared architecture class, but agent and map encoders have separate weights.
"""

import torch
import torch.nn as nn


class PolylineEncoder(nn.Module):
    """PointNet encoder that converts variable-length polylines into fixed-size tokens.

    Args:
        input_dim: per-point feature dimension (29 for agents, 9 for map)
        d_model: output embedding dimension (256)
        hidden_dims: MLP hidden layer sizes before max-pool
    """

    def __init__(self, input_dim: int, d_model: int = 256, hidden_dims: tuple = (64, 128)):
        super().__init__()

        # Per-point MLP: input_dim -> 64 -> 128 -> d_model
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU()])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, d_model))
        layers.append(nn.ReLU())
        self.point_mlp = nn.Sequential(*layers)

        # Post-aggregation MLP: d_model -> d_model -> d_model
        self.post_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, polylines: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Encode polylines to token embeddings.

        Args:
            polylines: (B, N, P, D) where N=num_polylines, P=points_per_polyline, D=feat_dim
            valid_mask: (B, N, P) bool mask for valid points

        Returns:
            tokens: (B, N, d_model) one embedding per polyline
        """
        B, N, P, D = polylines.shape

        # Reshape for MLP: (B*N*P, D)
        x = polylines.reshape(B * N * P, D)
        x = self.point_mlp(x)  # (B*N*P, d_model)
        x = x.reshape(B, N, P, -1)  # (B, N, P, d_model)

        # Mask invalid points before pooling
        mask = valid_mask.unsqueeze(-1)  # (B, N, P, 1)
        x = x * mask.float() + (~mask).float() * (-1e9)  # large negative for invalid

        # Max pool over points dimension
        x = x.max(dim=2).values  # (B, N, d_model)

        # Handle fully-invalid polylines (all points masked)
        polyline_valid = valid_mask.any(dim=2)  # (B, N)
        x = x * polyline_valid.unsqueeze(-1).float()

        # Post-aggregation MLP
        x = self.post_mlp(x)  # (B, N, d_model)
        x = self.norm(x)

        # Re-apply mask after norm (LayerNorm can produce non-zero from zeros)
        x = x * polyline_valid.unsqueeze(-1).float()

        return x
