# Data Directory

This directory contains datasets for document understanding experiments.

## Supported Datasets

| Dataset | Documents | Task | Domain |
|---------|-----------|------|--------|
| FUNSD | 199 | Entity recognition | Forms |
| CORD | 1,000 | Information extraction | Receipts |
| DocVQA | 5,349 | Visual question answering | Mixed |
| Financial | 40 | Classification | Financial |
| Legal | 1,200 | Information extraction | Legal |
| Technical | 2,400 | Classification | Technical |

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
  "words": ["word1", "word2", ...],
  "labels": [...],
  "metadata": {}
}
```
