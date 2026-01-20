#!/bin/bash
# Run all experiments for Architecture-Agnostic Document Understanding
# This reproduces the full experimental setup from the paper

set -e

echo "=============================================="
echo "Architecture-Agnostic Document Understanding"
echo "Full Experiment Suite"
echo "=============================================="

# Create results directory
mkdir -p results

# Datasets
DATASETS="funsd cord docvqa financial legal technical"
EPOCHS=10

echo ""
echo "=============================================="
echo "PHASE 1: BERT Training (Text-Only)"
echo "=============================================="

for dataset in $DATASETS; do
    echo ""
    echo "Training BERT on $dataset..."
    python scripts/train_bert.py \
        --dataset $dataset \
        --epochs $EPOCHS \
        --batch_size 16 \
        --curriculum
done

echo ""
echo "=============================================="
echo "PHASE 2: LayoutLMv3 Training (Multimodal)"
echo "=============================================="

for dataset in $DATASETS; do
    echo ""
    echo "Training LayoutLMv3 on $dataset..."
    python scripts/train_layoutlmv3.py \
        --dataset $dataset \
        --epochs $EPOCHS \
        --batch_size 8 \
        --curriculum
done

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "Results saved to results/"
echo "=============================================="
