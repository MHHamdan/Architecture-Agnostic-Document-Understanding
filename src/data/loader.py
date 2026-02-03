#!/usr/bin/env python3
"""
Unified Data Loader for Document Understanding Datasets

Supports loading from:
1. HuggingFace datasets library (primary source for FUNSD, CORD)
2. Local JSONL files (fallback)
3. Synthetic data (testing fallback)

Datasets:
- FUNSD: Form understanding (149 train / 50 test) - entity recognition
- CORD: Receipt understanding (800 train / 100 val / 100 test) - key-value extraction
- DocVQA: Visual question answering (5,349 documents)
- Financial: Financial document classification (30 documents)
- Legal: Legal document extraction (1,000 documents)
- Technical: Technical manual classification (2,000 documents)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ============================================================
# Label Mappings
# ============================================================

# FUNSD SER labels (7 classes, IOB2 format)
FUNSD_LABEL_LIST = [
    "O",
    "B-HEADER", "I-HEADER",
    "B-QUESTION", "I-QUESTION",
    "B-ANSWER", "I-ANSWER",
]
FUNSD_ID2LABEL = {i: l for i, l in enumerate(FUNSD_LABEL_LIST)}
FUNSD_LABEL2ID = {l: i for i, l in enumerate(FUNSD_LABEL_LIST)}
FUNSD_NUM_LABELS = len(FUNSD_LABEL_LIST)

# CORD SER entity types (30 classes: entity-type labels, not IOB)
# During evaluation, these are converted to IOB2 for seqeval
CORD_LABEL_LIST = [
    "O",
    "MENU.NM", "MENU.NUM", "MENU.UNITPRICE", "MENU.CNT",
    "MENU.DISCOUNTPRICE", "MENU.PRICE", "MENU.ITEMSUBTOTAL",
    "MENU.VATYN", "MENU.ETC", "MENU.SUB_NM", "MENU.SUB_UNITPRICE",
    "MENU.SUB_CNT", "MENU.SUB_PRICE", "MENU.SUB_ETC",
    "TOTAL.TOTAL_PRICE", "TOTAL.CASHPRICE", "TOTAL.CHANGEPRICE",
    "TOTAL.MENUTYPE_CNT", "TOTAL.MENUQTY_CNT", "TOTAL.TOTAL_ETC",
    "SUB_TOTAL.SUBTOTAL_PRICE", "SUB_TOTAL.DISCOUNT_PRICE",
    "SUB_TOTAL.SERVICE_PRICE", "SUB_TOTAL.OTHERSVC_PRICE",
    "SUB_TOTAL.TAX_PRICE", "SUB_TOTAL.ETC",
    "VOID_MENU.NM", "VOID_MENU.PRICE",
    "ETC",
]
CORD_ID2LABEL = {i: l for i, l in enumerate(CORD_LABEL_LIST)}
CORD_LABEL2ID = {l: i for i, l in enumerate(CORD_LABEL_LIST)}
CORD_NUM_LABELS = len(CORD_LABEL_LIST)


def entity_labels_to_iob2(label_ids: List[int], id2label: Dict[int, str]) -> List[str]:
    """Convert entity-type label IDs to IOB2 string tags for seqeval.

    For FUNSD, labels are already IOB2 so this is a simple id->string mapping.
    For CORD, labels are entity types that need B-/I- prefix conversion.
    """
    iob_tags = []
    prev_label = "O"
    for lid in label_ids:
        label = id2label.get(lid, "O")
        if label == "O":
            iob_tags.append("O")
        elif label.startswith("B-") or label.startswith("I-"):
            # Already IOB2 format (FUNSD)
            iob_tags.append(label)
        else:
            # Entity-type format (CORD) - convert to IOB2
            if label != prev_label:
                iob_tags.append(f"B-{label}")
            else:
                iob_tags.append(f"I-{label}")
        prev_label = label
    return iob_tags


@dataclass
class UnifiedExample:
    """Unified format for all dataset examples"""
    dataset: str
    task_type: str
    text: str
    labels: Any
    metadata: Dict[str, Any]
    # Token-level fields for NER/SER tasks
    words: Optional[List[str]] = None
    bboxes: Optional[List[List[int]]] = None
    ner_tags: Optional[List[int]] = None
    image: Optional[Any] = None  # PIL Image

    def to_dict(self) -> Dict:
        return {
            'dataset': self.dataset,
            'task_type': self.task_type,
            'text': self.text,
            'labels': self.labels,
            'metadata': self.metadata
        }


class UnifiedDataLoader:
    """
    Unified data loader for document understanding benchmarks.

    Loads datasets from HuggingFace or local files and converts to unified representation.
    """

    def __init__(self, data_dir: Path = Path("data/datasets")):
        self.data_dir = Path(data_dir)

        self.dataset_configs = {
            'funsd': {
                'name': 'FUNSD',
                'task_type': 'entity_recognition',
                'splits': ['train', 'test'],
                'loader': self._load_funsd,
                'label_list': FUNSD_LABEL_LIST,
                'id2label': FUNSD_ID2LABEL,
                'label2id': FUNSD_LABEL2ID,
                'num_labels': FUNSD_NUM_LABELS,
            },
            'cord': {
                'name': 'CORD',
                'task_type': 'key_value_extraction',
                'splits': ['train', 'validation', 'test'],
                'loader': self._load_cord,
                'label_list': CORD_LABEL_LIST,
                'id2label': CORD_ID2LABEL,
                'label2id': CORD_LABEL2ID,
                'num_labels': CORD_NUM_LABELS,
            },
            'docvqa': {
                'name': 'DocVQA',
                'task_type': 'visual_qa',
                'splits': ['validation'],
                'loader': self._load_docvqa,
                'num_labels': 1,
            },
            'financial': {
                'name': 'Financial',
                'task_type': 'classification',
                'splits': ['train', 'test'],
                'loader': self._load_generic,
                'num_labels': 10,
            },
            'legal': {
                'name': 'Legal',
                'task_type': 'extraction',
                'splits': ['train', 'test'],
                'loader': self._load_generic,
                'num_labels': 10,
            },
            'technical': {
                'name': 'Technical',
                'task_type': 'classification',
                'splits': ['train', 'validation'],
                'loader': self._load_generic,
                'num_labels': 10,
            }
        }

    def get_label_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get label mapping info for a dataset."""
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        config = self.dataset_configs[dataset_name]
        return {
            'label_list': config.get('label_list', [f"LABEL_{i}" for i in range(10)]),
            'id2label': config.get('id2label', {i: f"LABEL_{i}" for i in range(10)}),
            'label2id': config.get('label2id', {f"LABEL_{i}": i for i in range(10)}),
            'num_labels': config.get('num_labels', 10),
        }

    def load_dataset(
        self,
        dataset_name: str,
        split: str = "train",
        max_examples: Optional[int] = None
    ) -> List[UnifiedExample]:
        """Load a specific dataset split."""
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}. "
                           f"Available: {list(self.dataset_configs.keys())}")

        config = self.dataset_configs[dataset_name]

        if split not in config['splits']:
            logger.warning(f"Split '{split}' not available for {dataset_name}. "
                         f"Using {config['splits'][0]}")
            split = config['splits'][0]

        examples = config['loader'](dataset_name, split)

        if max_examples and len(examples) > max_examples:
            examples = examples[:max_examples]

        logger.info(f"Loaded {len(examples)} examples from {dataset_name}/{split}")
        return examples

    # ============================================================
    # FUNSD Loading
    # ============================================================

    def _load_funsd(self, dataset_name: str, split: str) -> List[UnifiedExample]:
        """Load FUNSD: try HuggingFace first, then JSONL, then synthetic."""
        # Try HuggingFace datasets
        examples = self._load_funsd_hf(split)
        if examples:
            return examples

        # Try local JSONL
        examples = self._load_funsd_jsonl(split)
        if examples:
            return examples

        # Synthetic fallback
        logger.warning("FUNSD: Using synthetic data (install 'datasets' for real data)")
        return self._create_synthetic_ner_examples('FUNSD', 'entity_recognition',
                                                    FUNSD_LABEL_LIST, 50)

    def _load_funsd_hf(self, split: str) -> Optional[List[UnifiedExample]]:
        """Load FUNSD from HuggingFace datasets library."""
        try:
            from datasets import load_dataset as hf_load_dataset
        except ImportError:
            logger.info("HuggingFace 'datasets' not installed, skipping HF loading")
            return None

        # Try known FUNSD dataset names
        for dataset_path in ["nielsr/funsd"]:
            try:
                logger.info(f"Loading FUNSD from HuggingFace: {dataset_path}/{split}")
                ds = hf_load_dataset(dataset_path, split=split)

                examples = []
                for item in ds:
                    # Handle different column names
                    words = item.get('tokens', item.get('words', []))
                    bboxes = item.get('bboxes', item.get('boxes', []))
                    ner_tags = item.get('ner_tags', item.get('labels', []))
                    image = item.get('image', None)
                    doc_id = item.get('id', '')

                    if not words:
                        continue

                    examples.append(UnifiedExample(
                        dataset='FUNSD',
                        task_type='entity_recognition',
                        text=" ".join(words),
                        labels=ner_tags,
                        metadata={'id': doc_id, 'split': split},
                        words=words,
                        bboxes=bboxes,
                        ner_tags=ner_tags,
                        image=image,
                    ))

                logger.info(f"Loaded {len(examples)} FUNSD examples from HuggingFace")
                return examples

            except Exception as e:
                logger.warning(f"Failed to load {dataset_path}: {e}")
                continue

        return None

    def _load_funsd_jsonl(self, split: str) -> Optional[List[UnifiedExample]]:
        """Load FUNSD from local JSONL files."""
        jsonl_path = self.data_dir / 'funsd' / f"{split}.jsonl"
        if not jsonl_path.exists():
            return None

        examples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                words = data.get('words', data.get('tokens', []))
                ner_tags = data.get('ner_tags', data.get('labels', []))
                bboxes = data.get('bboxes', data.get('boxes', []))

                if not words:
                    continue

                # Ensure ner_tags is a list of ints
                if ner_tags and isinstance(ner_tags[0], str):
                    ner_tags = [FUNSD_LABEL2ID.get(t, 0) for t in ner_tags]

                examples.append(UnifiedExample(
                    dataset='FUNSD',
                    task_type='entity_recognition',
                    text=" ".join(words),
                    labels=ner_tags,
                    metadata={'id': data.get('id', ''), 'split': split},
                    words=words,
                    bboxes=bboxes,
                    ner_tags=ner_tags,
                ))

        return examples if examples else None

    # ============================================================
    # CORD Loading
    # ============================================================

    def _load_cord(self, dataset_name: str, split: str) -> List[UnifiedExample]:
        """Load CORD: try HuggingFace first, then JSONL, then synthetic."""
        examples = self._load_cord_hf(split)
        if examples:
            return examples

        examples = self._load_cord_jsonl(split)
        if examples:
            return examples

        logger.warning("CORD: Using synthetic data (install 'datasets' for real data)")
        return self._create_synthetic_ner_examples('CORD', 'key_value_extraction',
                                                    CORD_LABEL_LIST, 100)

    def _load_cord_hf(self, split: str) -> Optional[List[UnifiedExample]]:
        """Load CORD from HuggingFace datasets library."""
        try:
            from datasets import load_dataset as hf_load_dataset
        except ImportError:
            return None

        try:
            logger.info(f"Loading CORD from HuggingFace: naver-clova-ix/cord-v2/{split}")
            ds = hf_load_dataset("naver-clova-ix/cord-v2", split=split)

            examples = []
            for idx, item in enumerate(ds):
                image = item.get('image', None)
                gt_string = item.get('ground_truth', '{}')

                # Parse ground truth
                try:
                    gt = json.loads(gt_string) if isinstance(gt_string, str) else gt_string
                    gt_parse = gt.get('gt_parse', gt)
                except (json.JSONDecodeError, TypeError):
                    continue

                words, ner_tags = self._parse_cord_gt(gt_parse)
                if not words:
                    continue

                # Create synthetic sequential bboxes (CORD v2 doesn't provide word positions)
                bboxes = self._create_sequential_bboxes(len(words))

                examples.append(UnifiedExample(
                    dataset='CORD',
                    task_type='key_value_extraction',
                    text=" ".join(words),
                    labels=ner_tags,
                    metadata={'split': split, 'index': idx},
                    words=words,
                    bboxes=bboxes,
                    ner_tags=ner_tags,
                    image=image,
                ))

            logger.info(f"Loaded {len(examples)} CORD examples from HuggingFace")
            return examples

        except Exception as e:
            logger.warning(f"Failed to load CORD from HuggingFace: {e}")
            return None

    def _parse_cord_gt(self, gt_parse: dict) -> tuple:
        """Parse CORD ground truth to extract words and entity-type labels."""
        words = []
        ner_tags = []

        if not isinstance(gt_parse, dict):
            return words, ner_tags

        for section_name, section_data in gt_parse.items():
            if isinstance(section_data, list):
                # Menu items (list of dicts)
                for item in section_data:
                    if not isinstance(item, dict):
                        continue
                    for key, value in item.items():
                        if isinstance(value, str) and value.strip():
                            entity_type = f"{section_name.upper()}.{key.upper()}"
                            label_id = CORD_LABEL2ID.get(entity_type, 0)
                            item_words = value.strip().split()
                            for w in item_words:
                                words.append(w)
                                ner_tags.append(label_id)
            elif isinstance(section_data, dict):
                # Totals, sub_totals (dict of key-value)
                for key, value in section_data.items():
                    if isinstance(value, str) and value.strip():
                        entity_type = f"{section_name.upper()}.{key.upper()}"
                        label_id = CORD_LABEL2ID.get(entity_type, 0)
                        item_words = value.strip().split()
                        for w in item_words:
                            words.append(w)
                            ner_tags.append(label_id)

        return words, ner_tags

    @staticmethod
    def _create_sequential_bboxes(n: int, width: int = 1000, height: int = 1000) -> List[List[int]]:
        """Create sequential bounding boxes for words without positional data."""
        bboxes = []
        words_per_line = max(1, n // 20 + 1)
        for i in range(n):
            row = i // words_per_line
            col = i % words_per_line
            x0 = int(col * (width / words_per_line))
            y0 = int(row * 50)
            x1 = int((col + 1) * (width / words_per_line))
            y1 = int(row * 50 + 40)
            # Clamp to valid range
            bboxes.append([
                min(x0, width), min(y0, height),
                min(x1, width), min(y1, height)
            ])
        return bboxes

    def _load_cord_jsonl(self, split: str) -> Optional[List[UnifiedExample]]:
        """Load CORD from local JSONL files."""
        jsonl_path = self.data_dir / 'cord' / f"{split}.jsonl"
        if not jsonl_path.exists():
            return None

        examples = []
        with open(jsonl_path, 'r') as f:
            for idx, line in enumerate(f):
                try:
                    data = json.loads(line)
                    ground_truth = data.get('ground_truth', {})
                    if isinstance(ground_truth, str):
                        ground_truth = json.loads(ground_truth)

                    gt_parse = ground_truth.get('gt_parse', ground_truth)
                    words, ner_tags = self._parse_cord_gt(gt_parse)

                    if not words:
                        # Fallback: extract text from the raw data
                        text_parts = []
                        if isinstance(gt_parse, dict) and 'menu' in gt_parse:
                            for item in gt_parse.get('menu', []):
                                if isinstance(item, dict):
                                    text_parts.append(f"{item.get('nm', '')} {item.get('price', '')}")
                        text = " | ".join(text_parts) if text_parts else f"Receipt #{idx}"
                        words = text.split()
                        ner_tags = [0] * len(words)

                    bboxes = self._create_sequential_bboxes(len(words))

                    examples.append(UnifiedExample(
                        dataset='CORD',
                        task_type='key_value_extraction',
                        text=" ".join(words),
                        labels=ner_tags,
                        metadata={'split': split, 'index': idx},
                        words=words,
                        bboxes=bboxes,
                        ner_tags=ner_tags,
                    ))
                except Exception as e:
                    logger.debug(f"Failed to parse CORD line {idx}: {e}")

        return examples if examples else None

    # ============================================================
    # DocVQA Loading
    # ============================================================

    def _load_docvqa(self, dataset_name: str, split: str) -> List[UnifiedExample]:
        """Load DocVQA dataset."""
        jsonl_path = self.data_dir / dataset_name / f"{split}.jsonl"

        if not jsonl_path.exists():
            logger.warning(f"File not found: {jsonl_path}")
            return self._create_synthetic_examples('DocVQA', 'visual_qa', 200)

        examples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                examples.append(UnifiedExample(
                    dataset='DocVQA',
                    task_type='visual_qa',
                    text=data.get('question', ''),
                    labels=data.get('answers', []),
                    metadata={
                        'questionId': data.get('questionId', ''),
                        'split': split
                    }
                ))

        return examples

    # ============================================================
    # Generic Loading (Financial, Legal, Technical)
    # ============================================================

    def _load_generic(self, dataset_name: str, split: str) -> List[UnifiedExample]:
        """Load generic dataset format."""
        config = self.dataset_configs[dataset_name]
        jsonl_path = self.data_dir / dataset_name / f"{split}.jsonl"

        if not jsonl_path.exists():
            logger.warning(f"File not found: {jsonl_path}")
            return self._create_synthetic_examples(
                config['name'], config['task_type'], 50
            )

        examples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                examples.append(UnifiedExample(
                    dataset=config['name'],
                    task_type=config['task_type'],
                    text=data.get('text', data.get('question', '')),
                    labels=data.get('label', data.get('answer', '')),
                    metadata={'split': split}
                ))

        return examples

    # ============================================================
    # Synthetic Data (fallback for testing)
    # ============================================================

    def _create_synthetic_examples(
        self, dataset: str, task_type: str, count: int
    ) -> List[UnifiedExample]:
        """Create synthetic examples for testing."""
        logger.info(f"Creating {count} synthetic examples for {dataset}")
        examples = []
        for i in range(count):
            examples.append(UnifiedExample(
                dataset=dataset,
                task_type=task_type,
                text=f"Sample document text {i} for {dataset} dataset with {task_type} task.",
                labels={'synthetic': True, 'index': i},
                metadata={'synthetic': True}
            ))
        return examples

    def _create_synthetic_ner_examples(
        self, dataset: str, task_type: str,
        label_list: List[str], count: int
    ) -> List[UnifiedExample]:
        """Create synthetic NER examples with proper word-level labels."""
        import random
        logger.info(f"Creating {count} synthetic NER examples for {dataset}")

        sample_texts = [
            "Company Name Invoice Number Date Amount Due Total Payment",
            "Customer Address Phone Email Order Reference Shipping",
            "Item Description Quantity Unit Price Subtotal Tax",
            "Receipt Store Location Date Time Transaction Number",
            "Document Title Section Header Content Body Footer",
        ]

        examples = []
        for i in range(count):
            text = sample_texts[i % len(sample_texts)]
            words = text.split()
            num_labels = len(label_list)
            # Assign random labels (mostly O, some entities)
            ner_tags = []
            for _ in words:
                if random.random() < 0.3:
                    ner_tags.append(random.randint(1, num_labels - 1))
                else:
                    ner_tags.append(0)

            bboxes = self._create_sequential_bboxes(len(words))

            examples.append(UnifiedExample(
                dataset=dataset,
                task_type=task_type,
                text=text,
                labels=ner_tags,
                metadata={'synthetic': True, 'index': i},
                words=words,
                bboxes=bboxes,
                ner_tags=ner_tags,
            ))
        return examples

    def get_available_datasets(self) -> List[str]:
        """Get list of available datasets."""
        return list(self.dataset_configs.keys())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    loader = UnifiedDataLoader()
    print(f"Available datasets: {loader.get_available_datasets()}")

    for ds_name in ['funsd', 'cord']:
        label_info = loader.get_label_info(ds_name)
        print(f"\n{ds_name}: {label_info['num_labels']} labels")
        print(f"  Labels: {label_info['label_list'][:5]}...")

        examples = loader.load_dataset(ds_name, max_examples=5)
        print(f"  Loaded: {len(examples)} examples")
        if examples:
            ex = examples[0]
            print(f"  Sample text: {ex.text[:80]}...")
            print(f"  Has words: {ex.words is not None}")
            print(f"  Has ner_tags: {ex.ner_tags is not None}")
            print(f"  Has bboxes: {ex.bboxes is not None}")
            print(f"  Has image: {ex.image is not None}")
