"""Training entry point for MTR-Lite.

Usage:
    python training/train.py --config configs/mtr_lite.yaml
    python training/train.py --config configs/mtr_lite_debug.yaml
"""

import argparse
import os

import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from data.collate import mtr_collate_fn
from data.polyline_dataset import PolylineDataset
from training.lightning_module import MTRLiteModule


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train MTR-Lite")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument(
        "--intent-points", type=str, default=None,
        help="Path to pre-computed intention points .npy file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    pl.seed_everything(cfg.get("seed", 42))

    # Datasets
    data_cfg = cfg["data"]
    train_dataset = PolylineDataset(
        scene_list_path=data_cfg["scene_list"],
        split="train",
        val_ratio=data_cfg.get("val_ratio", 0.15),
        data_fraction=data_cfg.get("data_fraction", 1.0),
        history_len=data_cfg["history_len"],
        future_len=data_cfg["future_len"],
        max_agents=data_cfg["max_agents"],
        max_map_polylines=data_cfg["max_map_polylines"],
        map_points_per_lane=data_cfg["map_points_per_lane"],
        neighbor_distance=data_cfg.get("neighbor_distance", 50.0),
        anchor_frames=data_cfg.get("anchor_frames", [10]),
        augment=data_cfg.get("augment", True),
        augment_rotation=data_cfg.get("augment_rotation", True),
        seed=cfg.get("seed", 42),
    )

    val_dataset = PolylineDataset(
        scene_list_path=data_cfg["scene_list"],
        split="val",
        val_ratio=data_cfg.get("val_ratio", 0.15),
        data_fraction=data_cfg.get("data_fraction", 1.0),
        history_len=data_cfg["history_len"],
        future_len=data_cfg["future_len"],
        max_agents=data_cfg["max_agents"],
        max_map_polylines=data_cfg["max_map_polylines"],
        map_points_per_lane=data_cfg["map_points_per_lane"],
        neighbor_distance=data_cfg.get("neighbor_distance", 50.0),
        anchor_frames=data_cfg.get("anchor_frames", [10]),
        augment=False,
        seed=cfg.get("seed", 42),
    )

    train_cfg = cfg["training"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        collate_fn=mtr_collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=train_cfg.get("num_workers", 4) > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        collate_fn=mtr_collate_fn,
        pin_memory=True,
        persistent_workers=train_cfg.get("num_workers", 4) > 0,
    )

    # Model
    module = MTRLiteModule(
        model_cfg=cfg["model"],
        training_cfg=train_cfg,
        loss_cfg=cfg.get("loss", {}),
    )

    # Load intention points if provided
    if args.intent_points and os.path.exists(args.intent_points):
        import numpy as np
        intent_pts = torch.from_numpy(np.load(args.intent_points)).float()
        module.model.motion_decoder.load_intention_points(intent_pts)
        print(f"Loaded intention points from {args.intent_points}")

    # Output directory
    output_dir = train_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="mtr_lite-{epoch:02d}-{val/minADE@6:.3f}",
        monitor="val/minADE@6",
        mode="min",
        save_top_k=3,
        every_n_epochs=train_cfg.get("checkpoint_every_n_epochs", 5),
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name="tb_logs",
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=train_cfg["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=train_cfg.get("precision", "16-mixed"),
        accumulate_grad_batches=train_cfg.get("gradient_accumulation", 8),
        gradient_clip_val=train_cfg.get("gradient_clip_val", 1.0),
        callbacks=[checkpoint_cb, lr_monitor],
        logger=logger,
        log_every_n_steps=train_cfg.get("log_every_n_steps", 50),
        val_check_interval=1.0,
        enable_progress_bar=True,
        default_root_dir=output_dir,
    )

    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume,
    )

    print(f"Training complete. Checkpoints saved to {output_dir}/checkpoints/")


if __name__ == "__main__":
    main()
