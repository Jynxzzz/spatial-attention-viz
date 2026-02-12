#!/bin/bash
# =============================================================================
# Post-Training Experiment Pipeline
# Run this script after training completes (60 epochs)
# =============================================================================

set -euo pipefail

# --- Configuration ---
PROJECT_DIR="/home/xingnan/projects/scenario-dreamer-transformer-visualizer-paper"
CHECKPOINT="/mnt/hdd12t/outputs/mtr_lite/checkpoints/last.ckpt"
CONFIG="${PROJECT_DIR}/configs/mtr_lite.yaml"
OUTPUT_BASE="/mnt/hdd12t/outputs/mtr_lite"
INTENT_POINTS="/mnt/hdd12t/models/mtr_lite/intent_points_64.npy"

echo "========================================"
echo "Post-Training Experiment Pipeline"
echo "========================================"
echo "Checkpoint: ${CHECKPOINT}"
echo "Config: ${CONFIG}"
echo ""

# Check checkpoint exists
if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found at ${CHECKPOINT}"
    echo "Is training still running?"
    exit 1
fi

# --- Step 1: Attention Analysis (200 validation scenes) ---
echo ""
echo "=== Step 1: Attention Analysis ==="
echo "Running on 200 validation scenes..."
python "${PROJECT_DIR}/evaluation/attention_analysis.py" \
    --checkpoint "${CHECKPOINT}" \
    --config "${CONFIG}" \
    --n-scenes 200 \
    --output-dir "${OUTPUT_BASE}/attention_analysis/" \
    --device cuda

echo "Attention analysis complete. Results: ${OUTPUT_BASE}/attention_analysis/"

# --- Step 2: Main Evaluation (full val set) ---
echo ""
echo "=== Step 2: Evaluation ==="
if [ -f "${PROJECT_DIR}/evaluation/evaluate.py" ]; then
    python "${PROJECT_DIR}/evaluation/evaluate.py" \
        --checkpoint "${CHECKPOINT}" \
        --config "${CONFIG}" \
        --output-dir "${OUTPUT_BASE}/evaluation/"
    echo "Evaluation complete."
else
    echo "WARNING: evaluate.py not found, skipping."
fi

# --- Step 3: Paper Figure Generation ---
echo ""
echo "=== Step 3: Paper Figures ==="
echo "Generating publication-quality figures..."
python "${PROJECT_DIR}/scripts/generate_paper_demo_figures.py" 2>/dev/null || \
    echo "WARNING: Paper figure generation had errors (may need real model data)."

echo ""
echo "========================================"
echo "Pipeline Complete!"
echo "========================================"
echo ""
echo "Results directories:"
echo "  Attention analysis: ${OUTPUT_BASE}/attention_analysis/"
echo "  Evaluation: ${OUTPUT_BASE}/evaluation/"
echo "  Paper figures: /tmp/paper_figures/"
echo ""
echo "Next steps:"
echo "  1. Review attention_analysis_results.json"
echo "  2. Review generated plots"
echo "  3. Fill in paper Results section with real metrics"
echo "  4. Compile final paper PDF"
