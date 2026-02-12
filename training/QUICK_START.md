# MTR-Lite Training - Quick Start Guide

**Last Updated**: 2026-02-10
**Status**: ✓ Verified and Ready

## TL;DR

```bash
# 1. Verify setup (optional but recommended)
bash scripts/verify_training_setup.sh

# 2. Run smoke test (3 minutes)
python training/train.py \
  --config configs/mtr_lite_smoke_test.yaml \
  --intent-points /mnt/hdd12t/models/mtr_lite/intent_points_64.npy

# 3. Run full training (50 hours)
python training/train.py \
  --config configs/mtr_lite.yaml \
  --intent-points /mnt/hdd12t/models/mtr_lite/intent_points_64.npy

# 4. Monitor progress
tensorboard --logdir /mnt/hdd12t/outputs/mtr_lite/tb_logs/
```

## Prerequisites

All dependencies are already installed:
- ✓ Python 3.10
- ✓ PyTorch 2.7.0+cu126
- ✓ PyTorch Lightning 2.6.1
- ✓ CUDA 12.6
- ✓ NVIDIA Driver 580.82

Data is ready:
- ✓ Scene list: 89,000 scenes at `/home/xingnan/projects/scenario-dreamer/scene_list_123k_signal_ssd.txt`
- ✓ Intention points: 64 points at `/mnt/hdd12t/models/mtr_lite/intent_points_64.npy`

## Training Options

### Option 1: Smoke Test (3 minutes)
**Purpose**: Quick verification that everything works
**Data**: 90 scenes (0.1%)
**Model**: 1.2M parameters
**Time**: ~3 minutes
**Command**:
```bash
python training/train.py \
  --config configs/mtr_lite_smoke_test.yaml \
  --intent-points /mnt/hdd12t/models/mtr_lite/intent_points_64.npy
```

### Option 2: Debug Run (5 minutes)
**Purpose**: Rapid iteration for development
**Data**: 450 scenes (0.5%)
**Model**: 0.6M parameters
**Time**: ~5 minutes
**Command**:
```bash
python training/train.py \
  --config configs/mtr_lite_debug.yaml \
  --intent-points /mnt/hdd12t/models/mtr_lite/intent_points_64.npy
```

### Option 3: Production Training (50 hours)
**Purpose**: Train final model for paper
**Data**: 17,800 scenes (20%)
**Model**: 8M parameters
**Time**: ~50 hours on RTX 4090
**Expected Results**: minADE@6: 0.6-1.0m
**Command**:
```bash
python training/train.py \
  --config configs/mtr_lite.yaml \
  --intent-points /mnt/hdd12t/models/mtr_lite/intent_points_64.npy
```

## Monitoring Training

### TensorBoard (Recommended)
```bash
tensorboard --logdir /mnt/hdd12t/outputs/mtr_lite/tb_logs/
# Open browser to http://localhost:6006
```

### Key Metrics to Watch
- **train/total_loss**: Should decrease from ~150 to ~20 over 60 epochs
- **val/minADE@6**: Should decrease from ~25m to ~0.6-1.0m (primary metric)
- **val/minFDE@6**: Should decrease from ~60m to ~1.5-3.0m
- **val/MR@6**: Should decrease from ~0.95 to ~0.20-0.30

### Training Progress (Expected)
```
Epoch  0: loss ~150, minADE ~25m  (random init)
Epoch  5: loss ~100, minADE ~10m  (learning basic motion)
Epoch 10: loss ~60,  minADE ~4m   (capturing multi-modality)
Epoch 20: loss ~40,  minADE ~2m   (refining trajectories)
Epoch 40: loss ~25,  minADE ~1m   (near convergence)
Epoch 60: loss ~20,  minADE ~0.7m (converged)
```

## Outputs

Checkpoints are saved to `/mnt/hdd12t/outputs/mtr_lite/checkpoints/`:
- `last.ckpt` - Latest checkpoint (use this to resume)
- `mtr_lite-epoch=XX-val_minADE@6=Y.YYY.ckpt` - Top 3 models by validation minADE

TensorBoard logs are saved to `/mnt/hdd12t/outputs/mtr_lite/tb_logs/`.

## Resume Training

If training is interrupted, resume from the last checkpoint:

```bash
python training/train.py \
  --config configs/mtr_lite.yaml \
  --intent-points /mnt/hdd12t/models/mtr_lite/intent_points_64.npy \
  --resume /mnt/hdd12t/outputs/mtr_lite/checkpoints/last.ckpt
```

## Load Trained Model for Inference

```python
from training.lightning_module import MTRLiteModule
import torch

# Load checkpoint
module = MTRLiteModule.load_from_checkpoint(
    "/mnt/hdd12t/outputs/mtr_lite/checkpoints/best.ckpt"
)
model = module.model
model.eval()
model.cuda()

# Run inference
with torch.no_grad():
    output = model(batch, capture_attention=False)

# Output shapes:
# output["trajectories"]: (B, T, 6, 80, 2) - 6 modes, 80 timesteps, 2D positions
# output["scores"]: (B, T, 6) - confidence scores for each mode
# output["target_mask"]: (B, T) - which targets are valid
```

## Common Issues

### Out of Memory (OOM)
**Solution**: Edit `configs/mtr_lite.yaml`:
```yaml
training:
  batch_size: 2  # reduce from 4
  gradient_accumulation: 16  # increase from 8 to maintain effective batch
```

### Training too slow
**Solution**: Already optimized with:
- AMP (mixed precision) enabled
- 8 data loading workers
- Gradient accumulation for larger effective batch

### Loss not decreasing
**Check**:
1. Is the learning rate too high? Try `lr: 5e-5` instead of `1e-4`
2. Is data loading correctly? Check first batch shapes
3. Is loss NaN? Check for invalid data

### Need help?
See comprehensive troubleshooting guide in `training/README.md` (400+ lines).

## Verification

Run the verification script to check all components:

```bash
bash scripts/verify_training_setup.sh
```

This checks:
1. Python and PyTorch versions
2. CUDA availability
3. PyTorch Lightning installation
4. Training files present
5. Config files present
6. Scene list and intention points
7. Output directories writable
8. Disk space available
9. GPU status

## Next Steps After Training

1. **Evaluate on validation set**:
   ```bash
   python evaluation/evaluate.py --checkpoint best.ckpt
   ```

2. **Generate attention visualizations**:
   ```bash
   python visualization/attention_extractor.py --checkpoint best.ckpt
   ```

3. **Create paper figures**:
   ```bash
   python scripts/generate_paper_figures.py
   ```

4. **Run ablations**:
   ```bash
   python scripts/run_ablation_suite.py
   ```

## Hardware Requirements

**Minimum**:
- GPU: RTX 3090 (24GB VRAM)
- RAM: 32GB
- Storage: 200GB free on SSD, 50GB free on HDD

**Recommended** (current setup):
- GPU: RTX 4090 (24GB VRAM) ✓
- RAM: 64GB ✓
- Storage: 1TB free on SSD, 6TB free on HDD ✓

## System Resources

Current status:
- **GPU**: RTX 4090, 24GB VRAM, 0% utilization (idle)
- **SSD**: 1020GB free on `/` (42% used)
- **HDD**: 6.8TB free on `/mnt/hdd12t` (35% used)
- **RAM**: 64GB total

## Support

For detailed documentation, see:
- `training/README.md` - Comprehensive training guide (400+ lines)
- `AGENT_D_COMPLETION_REPORT.md` - Implementation details and verification results
- `data/README.md` - Data pipeline documentation
- `model/QUICK_REFERENCE.md` - Model architecture reference

## Citation

```bibtex
@article{mtr_lite_2026,
  title={Thinking on the Map: Interpretable Attention Visualization for Trajectory Prediction},
  journal={MDPI Sustainability},
  year={2026}
}
```

---

**Training infrastructure verified**: 2026-02-10
**Smoke test status**: ✓ PASSED (2 epochs completed successfully)
**Ready for production**: ✓ YES
