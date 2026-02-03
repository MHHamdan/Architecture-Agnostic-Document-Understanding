# Architecture-Agnostic Curriculum Learning for Document Understanding

A PyTorch implementation of architecture-agnostic curriculum learning for document understanding, evaluated across BERT and LayoutLMv3 architectures.

## Overview

This repository provides the implementation and experimental validation for a progressive data scheduling approach to curriculum learning in document understanding. The three-phase schedule (33% -> 67% -> 100% data) operates independently of model architecture, delivering consistent training time reductions across both text-only and multimodal models.

## Research Contributions

1. **Architecture-Agnostic Progressive Scheduling**: A three-phase data schedule that reduces wall-clock training time by ~33% across both BERT (text-only) and LayoutLMv3 (multimodal) without requiring schedule modification per architecture.

2. **Matched-Compute Baselines**: Controlled experiments that separate curriculum effects from compute reduction by comparing against a 7-epoch standard baseline with matched gradient updates.

3. **Schedule Ablations**: Comparison of progressive, two-phase, reverse, and random pacing schedules showing that efficiency gains derive from reduced data volume rather than specific ordering.

4. **Statistical Analysis**: Paired t-tests with Cohen's d_z effect sizes across shared random seeds, revealing architecture-dependent curriculum benefits (significant for BERT on FUNSD, not for LayoutLMv3).

## Datasets

Six document understanding datasets spanning multiple domains:

| Dataset | Documents | Task | Domain |
|---------|-----------|------|--------|
| FUNSD | 199 | Entity recognition | Forms |
| CORD | 1,000 | Key-value extraction | Receipts |
| DocVQA | 200 | Visual question answering | Mixed documents |
| Financial | 50 | Classification | Financial documents |
| Legal | 50 | Information extraction | Legal contracts |
| Technical | 50 | Classification | Technical manuals |

> FUNSD and CORD are standard benchmarks. DocVQA, Financial, Legal, and Technical use synthetic data for framework validation.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU with 11GB+ memory (tested on RTX 2080 Ti)
- transformers 4.30+

## Installation

```bash
git clone https://github.com/MHHamdan/Architecture-Agnostic-Document-Understanding.git
cd Architecture-Agnostic-Document-Understanding

pip install -r requirements.txt
```

## Quick Start

### Training BERT with Curriculum Learning

```bash
python scripts/train_bert.py \
    --dataset funsd \
    --epochs 10 \
    --batch_size 16 \
    --curriculum
```

### Training LayoutLMv3 with Curriculum Learning

```bash
python scripts/train_layoutlmv3.py \
    --dataset cord \
    --epochs 10 \
    --batch_size 4 \
    --curriculum
```

### Reproducing All Experiments

```bash
# Full experiment suite (102 runs: primary + ablations + extended domains)
python scripts/run_experiments.py

# Quick test with a single seed
python scripts/run_experiments.py --quick

# Specific phase only
python scripts/run_experiments.py --phase 1   # Primary experiments
python scripts/run_experiments.py --phase 2   # Schedule ablations
python scripts/run_experiments.py --phase 3   # Extended domain evaluation
```

## Repository Structure

```
Architecture-Agnostic-Document-Understanding/
├── src/                              # Source code
│   ├── curriculum/                   # Curriculum scheduler
│   │   └── scheduler.py             # Progressive data scheduling
│   ├── data/                         # Data loading
│   │   └── loader.py                # Unified loader for all datasets
│   ├── evaluation/                   # Evaluation metrics
│   │   └── metrics.py               # Entity F1, ANLS, statistical tests
│   ├── models/                       # Model trainers
│   │   ├── bert_trainer.py           # BERT training pipeline
│   │   └── layoutlmv3_trainer.py     # LayoutLMv3 training pipeline
│   └── training/                     # Training utilities
│       └── utils.py                  # Seed, optimizer, scheduler, checkpoints
├── scripts/                          # Training scripts
│   ├── train_bert.py                 # BERT training CLI
│   ├── train_layoutlmv3.py           # LayoutLMv3 training CLI
│   ├── run_experiments.py            # Full experiment suite
│   └── run_phase3_recovery.py        # Recovery for failed experiments
├── configs/                          # Configuration files
│   └── default.yaml                  # Default training configuration
├── data/                             # Datasets directory
├── figures/                          # Generated figures
├── updated-writing-paper/            # LaTeX paper source
├── requirements.txt                  # Python dependencies
├── LICENSE                           # MIT license
└── README.md
```

## Curriculum Learning Schedule

The progressive schedule partitions 10 training epochs into three phases:

| Phase | Epochs | Data Ratio | Description |
|-------|--------|------------|-------------|
| Phase 1 (Easy) | 1-3 | 33% | Initial training on data subset |
| Phase 2 (Medium) | 4-7 | 67% | Expanded data exposure |
| Phase 3 (Hard) | 8-10 | 100% | Full dataset training |

**Effective data exposure**: 3 x 0.33 + 4 x 0.67 + 3 x 1.00 = 6.67 epoch-equivalents (vs. 10.0 for standard training).

## Model Architectures

| Model | Parameters | Input | Batch Size |
|-------|------------|-------|------------|
| BERT-base-uncased | 110M | Text tokens only | 16 |
| LayoutLMv3-base | 126M | Text + bounding boxes + images | 4 |

## Key Results

### Training Efficiency

The progressive schedule reduces training time by ~33% for both architectures:

| Architecture | FUNSD Speedup | CORD Speedup |
|--------------|---------------|--------------|
| BERT | 33.3% | 33.3% |
| LayoutLMv3 | 33.9% | 33.5% |

### Matched-Compute Comparison (Curriculum-10 vs Standard-7)

| Dataset | Architecture | Delta F1 | p-value | Cohen's d_z |
|---------|-------------|----------|---------|-------------|
| FUNSD | BERT | +0.023 | 0.022 | 3.83 |
| FUNSD | LayoutLMv3 | +0.003 | 0.621 | 0.33 |
| CORD | BERT | +0.000 | 0.900 | 0.08 |
| CORD | LayoutLMv3 | -0.006 | 0.496 | -0.48 |

## Reproducibility

All experiments use deterministic settings:

- Fixed random seeds: 42, 123, 456
- Deterministic CUDA operations
- Exact software versions in `requirements.txt`

Training configuration:
- Optimizer: AdamW (beta1=0.9, beta2=0.999, weight decay=0.01)
- Learning rate: 5e-5 with linear warmup (10%) and linear decay
- Gradient clipping: max norm 1.0
- Precision: FP32

Hardware: NVIDIA RTX 2080 Ti (11GB), ~2.1 GB peak memory (BERT), ~2.5 GB peak memory (LayoutLMv3).

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@inproceedings{hamdan2025architecture,
  title={Architecture-Agnostic Curriculum Learning for Document Understanding:
         Empirical Evidence from Text-Only and Multimodal Paradigms},
  author={Hamdan, Mohammed and Dentamaro, Vincenzo and Pirlo, Giuseppe and Cheriet, Mohamed},
  year={2025},
  note={Under review}
}
```

## License

MIT License - See [LICENSE](LICENSE) for details.
