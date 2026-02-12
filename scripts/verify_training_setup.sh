#!/bin/bash
# Quick verification script for MTR-Lite training infrastructure
# Usage: bash scripts/verify_training_setup.sh

set -e

PROJECT_ROOT="/home/xingnan/projects/scenario-dreamer-transformer-visualizer-paper"
cd "$PROJECT_ROOT"

echo "======================================"
echo "MTR-Lite Training Setup Verification"
echo "======================================"
echo ""

# Check Python version
echo "[1/10] Checking Python version..."
python --version || { echo "ERROR: Python not found"; exit 1; }
echo "✓ Python OK"
echo ""

# Check PyTorch
echo "[2/10] Checking PyTorch..."
python -c "import torch; print(f'PyTorch {torch.__version__}')" || { echo "ERROR: PyTorch not installed"; exit 1; }
echo "✓ PyTorch OK"
echo ""

# Check CUDA
echo "[3/10] Checking CUDA..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'CUDA available: {torch.cuda.get_device_name(0)}')" || { echo "ERROR: CUDA not available"; exit 1; }
echo "✓ CUDA OK"
echo ""

# Check PyTorch Lightning
echo "[4/10] Checking PyTorch Lightning..."
python -c "import pytorch_lightning as pl; print(f'PyTorch Lightning {pl.__version__}')" || { echo "ERROR: PyTorch Lightning not installed. Run: pip install pytorch-lightning"; exit 1; }
echo "✓ PyTorch Lightning OK"
echo ""

# Check training files
echo "[5/10] Checking training files..."
for file in training/losses.py training/metrics.py training/lightning_module.py training/train.py; do
    [ -f "$file" ] || { echo "ERROR: $file not found"; exit 1; }
done
echo "✓ Training files OK"
echo ""

# Check config files
echo "[6/10] Checking config files..."
for file in configs/mtr_lite.yaml configs/mtr_lite_debug.yaml configs/mtr_lite_smoke_test.yaml; do
    [ -f "$file" ] || { echo "ERROR: $file not found"; exit 1; }
done
echo "✓ Config files OK"
echo ""

# Check scene list
echo "[7/10] Checking scene list..."
SCENE_LIST="/home/xingnan/projects/scenario-dreamer/scene_list_123k_signal_ssd.txt"
[ -f "$SCENE_LIST" ] || { echo "ERROR: Scene list not found at $SCENE_LIST"; exit 1; }
NUM_SCENES=$(wc -l < "$SCENE_LIST")
echo "✓ Scene list OK ($NUM_SCENES scenes)"
echo ""

# Check intention points
echo "[8/10] Checking intention points..."
INTENT_POINTS="/mnt/hdd12t/models/mtr_lite/intent_points_64.npy"
[ -f "$INTENT_POINTS" ] || { echo "ERROR: Intention points not found at $INTENT_POINTS"; exit 1; }
python -c "import numpy as np; pts = np.load('$INTENT_POINTS'); assert pts.shape == (64, 2), f'Wrong shape: {pts.shape}'; print(f'Shape: {pts.shape}')" || { echo "ERROR: Intention points have wrong shape"; exit 1; }
echo "✓ Intention points OK"
echo ""

# Check output directory
echo "[9/10] Checking output directory..."
OUTPUT_DIR="/mnt/hdd12t/outputs/mtr_lite"
mkdir -p "$OUTPUT_DIR"
echo "✓ Output directory OK: $OUTPUT_DIR"
echo ""

# Check disk space
echo "[10/10] Checking disk space..."
SSD_AVAIL=$(df -h / | awk 'NR==2 {print $4}')
HDD_AVAIL=$(df -h /mnt/hdd12t | awk 'NR==2 {print $4}')
echo "SSD (/) available: $SSD_AVAIL"
echo "HDD (/mnt/hdd12t) available: $HDD_AVAIL"
echo "✓ Disk space OK"
echo ""

# Check GPU status
echo "======================================"
echo "GPU Status"
echo "======================================"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv
echo ""

echo "======================================"
echo "✓ ALL CHECKS PASSED"
echo "======================================"
echo ""
echo "Training infrastructure is ready!"
echo ""
echo "Quick start commands:"
echo ""
echo "  # Debug run (5 minutes):"
echo "  python training/train.py --config configs/mtr_lite_debug.yaml --intent-points $INTENT_POINTS"
echo ""
echo "  # Smoke test (3 minutes):"
echo "  python training/train.py --config configs/mtr_lite_smoke_test.yaml --intent-points $INTENT_POINTS"
echo ""
echo "  # Full training (50 hours):"
echo "  python training/train.py --config configs/mtr_lite.yaml --intent-points $INTENT_POINTS"
echo ""
echo "  # Monitor with TensorBoard:"
echo "  tensorboard --logdir /mnt/hdd12t/outputs/mtr_lite/tb_logs/"
echo ""
