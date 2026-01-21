#!/usr/bin/env python3
"""
Training Utilities for Document Understanding

Provides common training utilities:
- Seed setting for reproducibility
- Optimizer and scheduler creation
- Checkpoint management
"""

import os
import random
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Model
    model_name: str = "bert-base-uncased"
    num_labels: int = 10

    # Training
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Curriculum
    use_curriculum: bool = True

    # Hardware
    device: str = "cuda"
    fp16: bool = False

    # Paths
    output_dir: str = "results"
    checkpoint_dir: str = "checkpoints"

    # Reproducibility
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "max_grad_norm": self.max_grad_norm,
            "use_curriculum": self.use_curriculum,
            "device": self.device,
            "fp16": self.fp16,
            "seed": self.seed
        }


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info(f"Random seed set to {seed}")


def get_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    no_decay_params: Optional[list] = None
) -> AdamW:
    """
    Create AdamW optimizer with weight decay.

    Args:
        model: PyTorch model
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        no_decay_params: Parameter names to exclude from weight decay

    Returns:
        AdamW optimizer
    """
    if no_decay_params is None:
        no_decay_params = ["bias", "LayerNorm.weight", "layer_norm.weight"]

    # Separate parameters with and without weight decay
    optimizer_grouped_params = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay_params) and p.requires_grad
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay_params) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_params,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    return optimizer


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_ratio: float = 0.1,
    scheduler_type: str = "linear"
):
    """
    Create learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer
        num_training_steps: Total number of training steps
        warmup_ratio: Fraction of steps for warmup
        scheduler_type: Type of scheduler ('linear' or 'cosine')

    Returns:
        Learning rate scheduler
    """
    num_warmup_steps = int(warmup_ratio * num_training_steps)

    if scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    elif scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    logger.info(f"Created {scheduler_type} scheduler with {num_warmup_steps} warmup steps")

    return scheduler


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: Path,
    config: Optional[TrainingConfig] = None
):
    """
    Save training checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        path: Save path
        config: Training configuration
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    if config:
        checkpoint["config"] = config.to_dict()

    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved: {path}")


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict[str, Any]:
    """
    Load training checkpoint.

    Args:
        path: Checkpoint path
        model: PyTorch model to load weights into
        optimizer: Optional optimizer to load state into

    Returns:
        Checkpoint dict with epoch, loss, and config
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    logger.info(f"Checkpoint loaded: {path}")

    return {
        "epoch": checkpoint.get("epoch", 0),
        "loss": checkpoint.get("loss", 0.0),
        "config": checkpoint.get("config", {})
    }


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dict with total and trainable parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable
    }


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("Training Utilities Demo")
    print("=" * 60)

    # Set seed
    set_seed(42)
    print("Seed set to 42")

    # Config
    config = TrainingConfig(
        model_name="bert-base-uncased",
        epochs=10,
        batch_size=16,
        learning_rate=5e-5
    )
    print(f"\nConfig: {config.to_dict()}")
