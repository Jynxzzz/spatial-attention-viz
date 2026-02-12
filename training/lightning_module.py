"""PyTorch Lightning module for MTR-Lite training.

Handles:
- AdamW optimizer with warmup + cosine decay
- AMP (mixed precision) training
- Gradient accumulation
- Metric logging
- Checkpoint management
"""

import torch
import pytorch_lightning as pl

from model.mtr_lite import MTRLite
from training.losses import MTRLiteLoss
from training.metrics import MetricAggregator


class MTRLiteModule(pl.LightningModule):
    """Lightning module wrapping MTR-Lite model + loss + optimizer.

    Args:
        model_cfg: dict with model hyperparameters
        training_cfg: dict with training hyperparameters
        loss_cfg: dict with loss weights
    """

    def __init__(self, model_cfg: dict, training_cfg: dict, loss_cfg: dict):
        super().__init__()
        self.save_hyperparameters()

        self.model = MTRLite(**model_cfg)
        self.loss_fn = MTRLiteLoss(
            cls_weight=loss_cfg.get("cls_weight", 1.0),
            reg_weight=loss_cfg.get("reg_weight", 1.0),
            deep_supervision_weights=loss_cfg.get("deep_supervision_weights", [0.2, 0.2, 0.2, 0.4]),
        )

        self.training_cfg = training_cfg
        self.val_metrics = MetricAggregator()

    def forward(self, batch, capture_attention=False):
        return self.model(batch, capture_attention=capture_attention)

    def training_step(self, batch, batch_idx):
        if batch is None:
            return None

        output = self.model(batch, capture_attention=False)

        loss_dict = self.loss_fn(
            layer_trajectories=output["layer_trajectories"],
            layer_scores=output["layer_scores"],
            target_future=batch["target_future"],
            target_future_valid=batch["target_future_valid"],
            target_mask=batch["target_mask"],
        )

        self.log("train/total_loss", loss_dict["total_loss"], prog_bar=True, sync_dist=True)
        self.log("train/cls_loss", loss_dict["cls_loss"], sync_dist=True)
        self.log("train/reg_loss", loss_dict["reg_loss"], sync_dist=True)

        return loss_dict["total_loss"]

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return

        output = self.model(batch, capture_attention=False)

        loss_dict = self.loss_fn(
            layer_trajectories=output["layer_trajectories"],
            layer_scores=output["layer_scores"],
            target_future=batch["target_future"],
            target_future_valid=batch["target_future_valid"],
            target_mask=batch["target_mask"],
        )

        self.log("val/total_loss", loss_dict["total_loss"], prog_bar=True, sync_dist=True)

        # Compute trajectory metrics (with per-type breakdown if available)
        self.val_metrics.update(
            pred_trajectories=output["trajectories"].detach(),
            pred_scores=output["scores"].detach(),
            target_future=batch["target_future"],
            target_future_valid=batch["target_future_valid"],
            target_mask=batch["target_mask"],
            target_agent_types=batch.get("target_agent_types"),
        )

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        for key, value in metrics.items():
            self.log(f"val/{key}", value, prog_bar=(key == "minADE@6"))
        self.val_metrics.reset()

    def configure_optimizers(self):
        cfg = self.training_cfg
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=cfg.get("lr", 1e-4),
            weight_decay=cfg.get("weight_decay", 0.01),
        )

        warmup_epochs = cfg.get("warmup_epochs", 5)
        max_epochs = cfg.get("max_epochs", 60)

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
