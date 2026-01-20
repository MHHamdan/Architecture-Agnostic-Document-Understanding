#!/usr/bin/env python3
"""
BERT Trainer with HCML Curriculum Learning

BERT-base-uncased: 110M parameters, text-only transformer encoder
- 12 layers, 768 hidden size, 12 attention heads
- WordPiece tokenization (30K vocab)
- Peak memory: ~2.1 GB
- Throughput: ~1,250 samples/sec
"""

import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertForTokenClassification,
    get_linear_schedule_with_warmup
)

from ..curriculum import HierarchicalCurriculumScheduler
from ..data import UnifiedDataLoader, UnifiedExample

logger = logging.getLogger(__name__)


class BERTDataset(Dataset):
    """BERT-compatible dataset for document understanding"""

    def __init__(
        self,
        examples: List[UnifiedExample],
        tokenizer: BertTokenizer,
        max_length: int = 512,
        num_labels: int = 10
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_labels = num_labels

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        encoding = self.tokenizer(
            example.text[:1000],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_type_ids': encoding['token_type_ids'].squeeze(0),
        }

        # Create token classification labels
        labels = torch.zeros(self.max_length, dtype=torch.long)
        item['labels'] = labels

        return item


class BERTTrainer:
    """
    BERT GPU Trainer with HCML Curriculum Learning

    Implements architecture-agnostic curriculum learning for text-only
    document understanding using BERT-base-uncased.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        output_dir: Path = Path("results/bert"),
        device: str = "cuda"
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        logger.info("=" * 70)
        logger.info("BERT Trainer Initialized")
        logger.info(f"Model: {model_name}")
        logger.info(f"Device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU Memory: {gpu_mem:.2f} GB")
        logger.info("=" * 70)

        self.tokenizer = None
        self.model = None

    def load_model(self, num_labels: int = 10):
        """Load BERT model for token classification"""
        logger.info("Loading BERT model...")

        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels
        )
        self.model = self.model.to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Total parameters: {total_params:,}")

        return True

    def train(
        self,
        dataset_name: str,
        split: str = "train",
        epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 5e-5,
        max_samples: Optional[int] = None,
        use_curriculum: bool = True
    ) -> Dict[str, Any]:
        """
        Train BERT with HCML curriculum learning

        Args:
            dataset_name: Name of dataset to train on
            split: Dataset split
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            max_samples: Maximum samples to use
            use_curriculum: Whether to use HCML curriculum

        Returns:
            Training statistics dictionary
        """
        logger.info("=" * 70)
        logger.info(f"TRAINING: {dataset_name.upper()} with BERT")
        logger.info(f"Curriculum Learning: {'ENABLED' if use_curriculum else 'DISABLED'}")
        logger.info("=" * 70)

        # Load data
        loader = UnifiedDataLoader()
        examples = loader.load_dataset(dataset_name, split, max_examples=max_samples)

        if not examples:
            return {"status": "failed", "error": "No examples loaded"}

        # Load model
        if self.model is None:
            self.load_model()

        # Initialize curriculum
        curriculum = HierarchicalCurriculumScheduler(epochs) if use_curriculum else None

        # Training stats
        stats = {
            'dataset': dataset_name,
            'model': 'BERT',
            'architecture': 'bert-base-uncased',
            'parameters': '110M',
            'device': str(self.device),
            'epochs': [],
            'start_time': datetime.now().isoformat(),
            'curriculum_enabled': use_curriculum
        }

        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = (len(examples) // batch_size) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        # Training loop
        self.model.train()

        for epoch in range(epochs):
            epoch_start = time.time()

            # Curriculum learning
            if curriculum:
                info = curriculum.get_phase_info(epoch)
                sample_ratio = info['sample_ratio']
                num_samples = int(len(examples) * sample_ratio)
                epoch_examples = examples[:num_samples]
                logger.info(f"\nEpoch {epoch + 1}/{epochs} - "
                          f"{info['difficulty'].upper()} ({info['data_percentage']})")
            else:
                epoch_examples = examples
                logger.info(f"\nEpoch {epoch + 1}/{epochs}")

            # Create dataloader
            dataset = BERTDataset(epoch_examples, self.tokenizer)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=torch.cuda.is_available()
            )

            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                num_batches += 1

                if batch_idx % max(len(dataloader) // 5, 1) == 0:
                    logger.info(f"  Batch {batch_idx}/{len(dataloader)}: "
                              f"loss={loss.item():.4f}")

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            epoch_time = time.time() - epoch_start

            logger.info(f">>> Epoch {epoch + 1}: loss={avg_loss:.4f}, "
                       f"time={epoch_time:.2f}s")

            stats['epochs'].append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'time': epoch_time,
                'difficulty': curriculum.get_difficulty_level(epoch) if curriculum else 'standard',
                'num_samples': len(epoch_examples),
                'learning_rate': optimizer.param_groups[0]['lr']
            })

            # Save checkpoint
            self._save_checkpoint(dataset_name, epoch + 1, avg_loss)

        # Finalize stats
        stats['end_time'] = datetime.now().isoformat()
        stats['status'] = 'completed'
        stats['final_loss'] = stats['epochs'][-1]['loss']
        stats['total_time'] = sum(e['time'] for e in stats['epochs'])

        # Save results
        results_path = self.output_dir / dataset_name / "training_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETED!")
        logger.info(f"Final loss: {stats['final_loss']:.4f}")
        logger.info(f"Total time: {stats['total_time']:.2f}s")
        logger.info("=" * 70)

        return stats

    def _save_checkpoint(self, dataset_name: str, epoch: int, loss: float):
        """Save model checkpoint"""
        checkpoint_dir = self.output_dir / dataset_name / f"checkpoint-epoch{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
        }, checkpoint_dir / "pytorch_model.bin")

        logger.info(f"Checkpoint saved: {checkpoint_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    trainer = BERTTrainer()
    results = trainer.train(
        dataset_name="funsd",
        epochs=3,
        batch_size=8,
        use_curriculum=True
    )
