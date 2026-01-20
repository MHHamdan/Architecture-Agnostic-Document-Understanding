#!/usr/bin/env python3
"""
Unified Data Loader for Document Understanding Datasets

Supports 6 benchmark datasets:
- FUNSD: Form understanding (199 documents)
- CORD: Receipt understanding (1,000 receipts)
- DocVQA: Visual question answering (5,349 documents)
- Financial: Financial document classification (40 documents)
- Legal: Legal document extraction (1,200 documents)
- Technical: Technical manual classification (2,400 documents)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class UnifiedExample:
    """Unified format for all dataset examples"""
    dataset: str
    task_type: str
    text: str
    labels: Any
    metadata: Dict[str, Any]

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

    Loads datasets in different formats and converts to unified representation.
    """

    def __init__(self, data_dir: Path = Path("data/datasets")):
        self.data_dir = Path(data_dir)

        self.dataset_configs = {
            'funsd': {
                'name': 'FUNSD',
                'task_type': 'entity_recognition',
                'splits': ['train', 'test'],
                'loader': self._load_funsd
            },
            'cord': {
                'name': 'CORD',
                'task_type': 'key_value_extraction',
                'splits': ['train', 'validation', 'test'],
                'loader': self._load_cord
            },
            'docvqa': {
                'name': 'DocVQA',
                'task_type': 'visual_qa',
                'splits': ['validation'],
                'loader': self._load_docvqa
            },
            'financial': {
                'name': 'Financial',
                'task_type': 'classification',
                'splits': ['train', 'test'],
                'loader': self._load_generic
            },
            'legal': {
                'name': 'Legal',
                'task_type': 'extraction',
                'splits': ['train', 'test'],
                'loader': self._load_generic
            },
            'technical': {
                'name': 'Technical',
                'task_type': 'classification',
                'splits': ['train', 'validation'],
                'loader': self._load_generic
            }
        }

    def load_dataset(
        self,
        dataset_name: str,
        split: str = "train",
        max_examples: Optional[int] = None
    ) -> List[UnifiedExample]:
        """Load a specific dataset split"""
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

    def _load_funsd(self, dataset_name: str, split: str) -> List[UnifiedExample]:
        """Load FUNSD dataset"""
        jsonl_path = self.data_dir / dataset_name / f"{split}.jsonl"

        if not jsonl_path.exists():
            logger.warning(f"File not found: {jsonl_path}")
            return self._create_synthetic_examples('FUNSD', 'entity_recognition', 50)

        examples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                text = " ".join(data.get('words', []))
                entities = []

                if 'ner_tags' in data:
                    for word, tag in zip(data.get('words', []), data.get('ner_tags', [])):
                        if tag != 0:
                            entities.append({'word': word, 'tag': tag})

                examples.append(UnifiedExample(
                    dataset='FUNSD',
                    task_type='entity_recognition',
                    text=text,
                    labels=entities,
                    metadata={'id': data.get('id', ''), 'split': split}
                ))

        return examples

    def _load_cord(self, dataset_name: str, split: str) -> List[UnifiedExample]:
        """Load CORD dataset"""
        jsonl_path = self.data_dir / dataset_name / f"{split}.jsonl"

        if not jsonl_path.exists():
            logger.warning(f"File not found: {jsonl_path}")
            return self._create_synthetic_examples('CORD', 'key_value_extraction', 100)

        examples = []
        with open(jsonl_path, 'r') as f:
            for idx, line in enumerate(f):
                try:
                    data = json.loads(line)
                    ground_truth = data.get('ground_truth', {})

                    if isinstance(ground_truth, str):
                        ground_truth = json.loads(ground_truth)

                    text_parts = []
                    labels = {'menu': [], 'total': {}}

                    if isinstance(ground_truth, dict) and 'gt_parse' in ground_truth:
                        gt = ground_truth['gt_parse']
                        if 'menu' in gt:
                            for item in gt.get('menu', []):
                                if isinstance(item, dict):
                                    text_parts.append(f"{item.get('nm', '')} {item.get('price', '')}")
                                    labels['menu'].append(item)
                        if 'total' in gt:
                            labels['total'] = gt['total']

                    examples.append(UnifiedExample(
                        dataset='CORD',
                        task_type='key_value_extraction',
                        text=" | ".join(text_parts) if text_parts else f"Receipt #{idx}",
                        labels=labels,
                        metadata={'split': split}
                    ))
                except Exception as e:
                    logger.debug(f"Failed to parse CORD line {idx}: {e}")

        return examples

    def _load_docvqa(self, dataset_name: str, split: str) -> List[UnifiedExample]:
        """Load DocVQA dataset"""
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

    def _load_generic(self, dataset_name: str, split: str) -> List[UnifiedExample]:
        """Load generic dataset format"""
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

    def _create_synthetic_examples(
        self,
        dataset: str,
        task_type: str,
        count: int
    ) -> List[UnifiedExample]:
        """Create synthetic examples for testing"""
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

    def get_available_datasets(self) -> List[str]:
        """Get list of available datasets"""
        return list(self.dataset_configs.keys())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    loader = UnifiedDataLoader()
    print(f"Available datasets: {loader.get_available_datasets()}")

    for ds_name in loader.get_available_datasets():
        examples = loader.load_dataset(ds_name, max_examples=5)
        print(f"\n{ds_name}: {len(examples)} examples")
        if examples:
            print(f"  Sample: {examples[0].text[:80]}...")
