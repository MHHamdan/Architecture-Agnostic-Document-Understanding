"""
Architecture-Agnostic Document Understanding
Progressive curriculum learning for document understanding across text-only and multimodal architectures.
"""

__version__ = "1.0.0"
__author__ = "M. H. Hamdan"

from .curriculum import HierarchicalCurriculumScheduler, CurriculumConfig
from .data import UnifiedDataLoader, UnifiedExample
from .models import BERTTrainer, LayoutLMv3Trainer
from .evaluation import (
    Evaluator,
    compute_entity_f1,
    compute_token_f1,
    compute_accuracy,
    compute_anls,
)
from .training import set_seed, get_optimizer, get_scheduler, TrainingConfig

__all__ = [
    # Curriculum
    'HierarchicalCurriculumScheduler',
    'CurriculumConfig',
    # Data
    'UnifiedDataLoader',
    'UnifiedExample',
    # Models
    'BERTTrainer',
    'LayoutLMv3Trainer',
    # Evaluation
    'Evaluator',
    'compute_entity_f1',
    'compute_token_f1',
    'compute_accuracy',
    'compute_anls',
    # Training
    'set_seed',
    'get_optimizer',
    'get_scheduler',
    'TrainingConfig',
]
