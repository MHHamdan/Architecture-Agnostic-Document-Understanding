# Architecture-Agnostic Document Understanding

A PyTorch implementation of architecture-agnostic curriculum learning for document understanding, validated across BERT and LayoutLMv3 architectures.

## Overview

This repository contains the implementation and experimental validation for Hierarchical Curriculum Meta-Learning (HCML), a three-phase progressive difficulty training schedule that operates independently of model architecture. We demonstrate consistent curriculum effectiveness across text-only and multimodal document understanding models.

## Model Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 ARCHITECTURE-AGNOSTIC DOCUMENT UNDERSTANDING                │
│                   Hierarchical Curriculum Meta-Learning (HCML)              │
└─────────────────────────────────────────────────────────────────────────────┘

                            ┌─────────────────────┐
                            │   INPUT DOCUMENTS   │
                            │   (6 Datasets)      │
                            └──────────┬──────────┘
                                       │
                  ┌────────────────────┼────────────────────┐
                  ▼                    ▼                    ▼
          ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
          │    FUNSD      │    │     CORD      │    │   DocVQA      │
          │   (Forms)     │    │  (Receipts)   │    │    (VQA)      │
          └───────┬───────┘    └───────┬───────┘    └───────┬───────┘
                  │                    │                    │
                  └────────────────────┼────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   CURRICULUM SCHEDULER (Architecture-Agnostic)              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  Phase 1 (Easy)    │  Phase 2 (Medium)  │  Phase 3 (Hard)             │  │
│  │  Epochs 0-3        │  Epochs 3-7        │  Epochs 7-10                │  │
│  │  33% Data          │  67% Data          │  100% Data                  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│  Key: Consistent scaling factors (2.06±0.07, 1.50±0.01) across archs       │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                  ┌────────────────────┴────────────────────┐
                  ▼                                         ▼
┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│       BERT (Text-Only)          │   │    LayoutLMv3 (Multimodal)      │
│  ┌───────────────────────────┐  │   │  ┌───────────────────────────┐  │
│  │  Text Tokenizer           │  │   │  │ Text + Visual Tokenizer   │  │
│  │  (WordPiece, 30K vocab)   │  │   │  │ (BPE + Patch Embeddings)  │  │
│  └─────────────┬─────────────┘  │   │  └─────────────┬─────────────┘  │
│                ▼                │   │                ▼                │
│  ┌───────────────────────────┐  │   │  ┌───────────────────────────┐  │
│  │  Transformer Encoder      │  │   │  │ Multimodal Transformer    │  │
│  │  12 Layers, 768 Hidden    │  │   │  │ 12 Layers, 768 Hidden     │  │
│  │  110M Parameters          │  │   │  │ 126M Parameters           │  │
│  └─────────────┬─────────────┘  │   │  │ + Vision Encoder (ViT)    │  │
│                ▼                │   │  └─────────────┬─────────────┘  │
│  ┌───────────────────────────┐  │   │                ▼                │
│  │  [CLS] + Task Head        │  │   │  ┌───────────────────────────┐  │
│  └───────────────────────────┘  │   │  │ Cross-Modal Attention     │  │
│                                 │   │  │ + Layout Embeddings       │  │
│  Memory: 2.1 GB                 │   │  └───────────────────────────┘  │
│  Speed: 1,250 samples/sec       │   │                                 │
└─────────────────────────────────┘   │  Memory: 4.9 GB                 │
                  │                   │  Speed: 500 samples/sec         │
                  │                   └─────────────────────────────────┘
                  │                                         │
                  └────────────────────┬────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            EVALUATION & METRICS                             │
│  ┌────────────────────────────┬──────────────────────────────────────────┐  │
│  │  Task Metrics              │  Cross-Architecture Comparison           │  │
│  │  • Entity Recognition (F1) │  • 380× Mean Improvement (LMv3 vs BERT) │  │
│  │  • Info Extraction (Acc)   │  • 3.9× to 2,197× Task-Dependent Gains  │  │
│  │  • Visual QA (ANLS)        │  • Zero Training Failures (120 sessions)│  │
│  └────────────────────────────┴──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Curriculum Learning Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                 THREE-PHASE CURRICULUM SCHEDULE                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PHASE 1: EASY           PHASE 2: MEDIUM        PHASE 3: HARD    │
│  ─────────────           ──────────────         ─────────────    │
│                                                                  │
│  ┌───────────┐           ┌───────────┐          ┌───────────┐    │
│  │ Short     │           │ Medium    │          │ Full      │    │
│  │ docs      │     →     │ docs      │     →    │ dataset   │    │
│  │ Simple    │           │ Moderate  │          │ Complex   │    │
│  │ layouts   │           │ structure │          │ layouts   │    │
│  └───────────┘           └───────────┘          └───────────┘    │
│                                                                  │
│  33% of data             67% of data            100% of data     │
│  Epochs 0-3              Epochs 3-7             Epochs 7-10      │
│                                                                  │
│  ↓ Scale Factor          ↓ Scale Factor                          │
│     2.06 ± 0.07             1.50 ± 0.01                          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Research Contributions

1. **Architecture-Agnostic Curriculum Learning**: HCML achieves consistent sample scaling factors (2.06±0.07 Easy→Medium, 1.50±0.01 Medium→Hard) across BERT and LayoutLMv3, validating curriculum transferability across fundamentally different architectures.

2. **Quantitative Multimodal Analysis**: Under identical curriculum conditions, LayoutLMv3 demonstrates 380× arithmetic mean improvement in final training loss compared to BERT, with task-dependent gains from 3.9× (structured forms) to 2,197× (visual question answering).

3. **Production-Ready Validation**: Zero failures across 120 training sessions with full FP32 precision training feasible on consumer GPUs (4.9GB peak memory on 11GB hardware).

## Datasets

Six diverse document understanding datasets spanning multiple domains:

| Dataset | Documents | Task | Domain |
|---------|-----------|------|--------|
| FUNSD | 199 | Entity recognition | Forms |
| CORD | 1,000 | Information extraction | Receipts |
| DocVQA | 5,349 | Visual question answering | Mixed documents |
| Financial | 40 | Classification | Financial documents |
| Legal | 1,200 | Information extraction | Legal contracts |
| Technical | 2,400 | Classification | Technical manuals |

## Requirements

- Python 3.8 or higher
- PyTorch 2.0.1 or higher
- CUDA-capable GPU with 11GB+ memory (recommended)
- transformers 4.30.0 or higher

## Installation

```bash
# Clone repository
git clone https://github.com/MHHamdan/Architecture-Agnostic-Document-Understanding-.git
cd Architecture-Agnostic-Document-Understanding-

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Training BERT with HCML

```bash
python scripts/train_bert.py \
    --dataset funsd \
    --epochs 10 \
    --batch_size 16 \
    --curriculum
```

### Training LayoutLMv3 with HCML

```bash
python scripts/train_layoutlmv3.py \
    --dataset cord \
    --epochs 10 \
    --batch_size 8 \
    --curriculum
```

### Reproducing All Experiments

```bash
bash scripts/run_all_experiments.sh
```

## Repository Structure

```
Architecture-Agnostic-Document-Understanding-/
├── src/                          # Source code
│   ├── curriculum/               # HCML curriculum scheduler
│   │   └── scheduler.py          # HierarchicalCurriculumScheduler
│   ├── data/                     # Data loading
│   │   └── loader.py             # UnifiedDataLoader for 6 datasets
│   └── models/                   # Model trainers
│       ├── bert_trainer.py       # BERT training with HCML
│       └── layoutlmv3_trainer.py # LayoutLMv3 training with HCML
├── scripts/                      # Training scripts
│   ├── train_bert.py             # BERT training CLI
│   ├── train_layoutlmv3.py       # LayoutLMv3 training CLI
│   └── run_all_experiments.sh    # Full experiment suite
├── data/                         # Datasets directory
├── configs/                      # Configuration files
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT license
└── README.md                     # This file
```

## HCML Algorithm

Hierarchical Curriculum Meta-Learning implements a three-phase progressive training schedule:

- **Phase 1 (Easy)**: Epochs 0-3, 33% of training data
- **Phase 2 (Medium)**: Epochs 3-7, 67% of training data
- **Phase 3 (Hard)**: Epochs 7-10, 100% of training data

The curriculum operates at the data distribution level, making no assumptions about model architecture, input modality, or optimization strategy.

## Model Architectures

**BERT-base-uncased**: 110M parameters, text-only transformer encoder

**LayoutLMv3-base**: 126M parameters, multimodal transformer with vision encoder

Both models use comparable parameter counts to isolate the effect of multimodal processing under controlled curriculum conditions.

## Experimental Results

### Loss Convergence Comparison

| Dataset | BERT Final Loss | LayoutLMv3 Final Loss | Improvement Factor |
|---------|----------------|----------------------|-------------------|
| FUNSD | 0.445 | 0.089 | 5.0× |
| CORD | 0.267 | 0.034 | 7.9× |
| DocVQA | 0.523 | 0.0024 | 217.9× |
| Financial | 0.389 | 0.052 | 7.5× |
| Legal | 0.412 | 0.067 | 6.1× |
| Technical | 0.501 | 0.045 | 11.1× |

### Training Efficiency

| Architecture | Total Time | Samples/Second | Peak Memory |
|--------------|-----------|----------------|-------------|
| BERT | 0.92 hours | 1,250 | 2.1 GB |
| LayoutLMv3 | 2.30 hours | 500 | 4.9 GB |

## Reproducibility

All experiments use deterministic settings:

- Fixed random seed: 42
- Deterministic CUDA operations enabled
- Exact software versions in requirements.txt

Training configurations:
- Optimizer: AdamW (beta1=0.9, beta2=0.999, epsilon=1e-8)
- Learning rate: 5e-5 with linear warmup and cosine decay
- Gradient clipping: 1.0
- Precision: FP32 (no quantization)

## Hardware Requirements

Minimum specifications for reproducing experiments:
- GPU: NVIDIA GPU with 11GB memory (tested on RTX 2080 Ti)
- CPU: Modern multi-core processor
- RAM: 16GB system memory
- Storage: 50GB for datasets and checkpoints

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@article{hamdan2025architecture,
  title={Architecture-Agnostic Curriculum Learning for Document Understanding:
         A Comparative Study of Text-Only and Multimodal Approaches},
  author={Hamdan, M. H.},
  journal={Under Review},
  year={2025},
  url={https://github.com/MHHamdan/Architecture-Agnostic-Document-Understanding-}
}
```

## License

MIT License - See LICENSE file for details

## Contact

For questions about the implementation or experimental setup, please open an issue on GitHub.
