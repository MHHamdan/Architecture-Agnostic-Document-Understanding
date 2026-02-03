# Data Directory

This directory contains datasets for document understanding experiments.

## Supported Datasets

### Primary Benchmarks

| Dataset | Documents | Task | Domain |
|---------|-----------|------|--------|
| FUNSD | 199 | Entity recognition | Forms |
| CORD | 1,000 | Key-value extraction | Receipts |

### Extended Domain (Synthetic)

| Dataset | Documents | Task | Domain |
|---------|-----------|------|--------|
| DocVQA | 200 | Visual question answering | Mixed |
| Financial | 50 | Classification | Financial |
| Legal | 50 | Information extraction | Legal |
| Technical | 50 | Classification | Technical |

> Extended domain datasets use synthetic data for framework validation. They confirm that the curriculum pipeline functions correctly across document types, but do not represent competitive benchmark performance.

## Directory Structure

```
data/
├── datasets/
│   ├── funsd/
│   │   ├── train.jsonl
│   │   └── test.jsonl
│   ├── cord/
│   │   ├── train.jsonl
│   │   ├── validation.jsonl
│   │   └── test.jsonl
│   ├── docvqa/
│   │   └── validation.jsonl
│   ├── financial/
│   ├── legal/
│   └── technical/
└── samples/
    └── (sample files for testing)
```

## Downloading Datasets

### FUNSD
```bash
# Download from https://guillaumejaume.github.io/FUNSD/
wget https://guillaumejaume.github.io/FUNSD/dataset.zip
unzip dataset.zip -d datasets/funsd/
```

### CORD
```bash
# Available via Hugging Face
python -c "from datasets import load_dataset; ds = load_dataset('naver-clova-ix/cord-v2')"
```

### DocVQA
```bash
# Download from https://www.docvqa.org/
# Requires registration
```

## Data Format

All datasets are converted to JSONL format with the following structure:

```json
{
  "id": "sample_001",
  "text": "Document text content...",
  "words": ["word1", "word2"],
  "labels": [0, 1, 2],
  "metadata": {}
}
```

## Automatic Loading

The `UnifiedDataLoader` in `src/data/loader.py` tries to load data in this order:

1. HuggingFace datasets library (for FUNSD, CORD)
2. Local JSONL files
3. Synthetic fallback (for testing)
