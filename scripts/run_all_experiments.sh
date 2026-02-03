#!/bin/bash
# ============================================================
# Full Experiment Suite for Revised Paper
# ============================================================
#
# Runs ALL experiments needed for the paper revision:
#   1. Standard-10 baseline (10 epochs, 100% data)
#   2. Curriculum-10 (10 epochs, progressive 33%->67%->100%)
#   3. Standard-7 matched-compute baseline (7 epochs, 100% data)
#   4. Schedule ablations (two-phase, reverse, random)
#
# For each condition: 2 architectures x 2 datasets x 3 seeds
#
# Usage:
#   bash scripts/run_all_experiments.sh           # Run all
#   bash scripts/run_all_experiments.sh --quick    # Quick test (1 seed)
# ============================================================

set -e

RESULTS_DIR="results/revision_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

SEEDS="42 123 456"
DATASETS="funsd cord"

# Quick mode: 1 seed only
if [ "$1" == "--quick" ]; then
    SEEDS="42"
    echo "Quick mode: single seed"
fi

echo "============================================================"
echo "Architecture-Agnostic Curriculum Learning - Full Experiments"
echo "Results directory: $RESULTS_DIR"
echo "Seeds: $SEEDS"
echo "============================================================"

# ============================================================
# Phase 1: Primary experiments (Standard-10, Curriculum-10, Standard-7)
# ============================================================
echo ""
echo "=== Phase 1: Primary experiments ==="

for DS in $DATASETS; do
    for SEED in $SEEDS; do
        echo ""
        echo "--- BERT / $DS / seed=$SEED ---"

        # Standard-10
        python scripts/train_bert.py --dataset "$DS" --epochs 10 --batch_size 16 --no_curriculum --seed "$SEED"

        # Curriculum-10 (progressive)
        python scripts/train_bert.py --dataset "$DS" --epochs 10 --batch_size 16 --curriculum --schedule progressive --seed "$SEED"

        # Standard-7 (matched compute)
        python scripts/train_bert.py --dataset "$DS" --epochs 7 --batch_size 16 --no_curriculum --seed "$SEED"

        echo ""
        echo "--- LayoutLMv3 / $DS / seed=$SEED ---"

        # Standard-10
        python scripts/train_layoutlmv3.py --dataset "$DS" --epochs 10 --batch_size 4 --no_curriculum --seed "$SEED"

        # Curriculum-10 (progressive)
        python scripts/train_layoutlmv3.py --dataset "$DS" --epochs 10 --batch_size 4 --curriculum --schedule progressive --seed "$SEED"

        # Standard-7 (matched compute)
        python scripts/train_layoutlmv3.py --dataset "$DS" --epochs 7 --batch_size 4 --no_curriculum --seed "$SEED"
    done
done

# ============================================================
# Phase 2: Schedule ablations (on CORD, both architectures)
# ============================================================
echo ""
echo "=== Phase 2: Schedule ablations on CORD ==="

ABLATION_SCHEDULES="two_phase reverse random"

for SCHEDULE in $ABLATION_SCHEDULES; do
    for SEED in $SEEDS; do
        echo "--- Ablation: BERT / cord / $SCHEDULE / seed=$SEED ---"
        python scripts/train_bert.py --dataset cord --epochs 10 --batch_size 16 --curriculum --schedule "$SCHEDULE" --seed "$SEED"

        echo "--- Ablation: LayoutLMv3 / cord / $SCHEDULE / seed=$SEED ---"
        python scripts/train_layoutlmv3.py --dataset cord --epochs 10 --batch_size 4 --curriculum --schedule "$SCHEDULE" --seed "$SEED"
    done
done

# ============================================================
# Phase 3: Extended domain evaluation (BERT + LayoutLMv3)
# ============================================================
echo ""
echo "=== Phase 3: Extended domain evaluation ==="

EXT_DATASETS="docvqa financial legal technical"

for DS in $EXT_DATASETS; do
    for SEED in $SEEDS; do
        python scripts/train_bert.py --dataset "$DS" --epochs 10 --batch_size 16 --curriculum --seed "$SEED"
        python scripts/train_bert.py --dataset "$DS" --epochs 7 --batch_size 16 --no_curriculum --seed "$SEED"
        python scripts/train_layoutlmv3.py --dataset "$DS" --epochs 10 --batch_size 4 --curriculum --seed "$SEED"
        python scripts/train_layoutlmv3.py --dataset "$DS" --epochs 7 --batch_size 4 --no_curriculum --seed "$SEED"
    done
done

echo ""
echo "============================================================"
echo "All experiments complete!"
echo "Results saved to: $RESULTS_DIR"
echo "============================================================"
