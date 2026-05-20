#!/bin/bash
# SkillHub — Full Pipeline Runner
# Usage: bash run_all.sh
set -e

PYTHON=/Users/lsslcj/anaconda3/envs/genericagent/bin/python3
echo "=========================================="
echo "  SkillHub Full Pipeline"
echo "=========================================="

# ── Task B: Anomaly Detection ──
echo ""
echo "[1/6] Task B: Label + Detect"
$PYTHON src/pipeline/02_label_and_detect.py
echo "  → output/task_b/detection_results.csv"

# ── Layer 2: LLM Interpretation ──
echo ""
echo "[2/6] Layer 2: LLM Post-Hoc Interpretation"
$PYTHON src/pipeline/05_layer2_explain.py --max 30
echo "  → output/layer2/layer2_results.json"

# ── Task C: Adversarial Generation ──
echo ""
echo "[3/6] Task C: Adversarial Generation (10 seeds)"
$PYTHON src/legacy/task_c_adversarial_gen.py
echo "  → output/task_c/adversarial_results.json"

# ── Adversarial Training ──
echo ""
echo "[4/6] Adversarial Training"
$PYTHON src/training/adversarial_training.py
echo "  → output/adversarial_training/"

# ── Task D: De-risking ──
echo ""
echo "[5/6] Task D: De-risking + Audit"
$PYTHON src/pipeline/04_derisk.py --audit 5
echo "  → output/task_d/derisking_results.json"

# ── Visualization ──
echo ""
echo "[6/6] Visualization"
$PYTHON src/legacy/visualization.py
echo "  → output/visualization/"

echo ""
echo "=========================================="
echo "  Pipeline Complete!"
echo "  Reports: output/Final_Report.md"
echo "=========================================="
