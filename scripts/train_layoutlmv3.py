#!/usr/bin/env python3
"""
LayoutLMv3 Training Script with Curriculum Learning and Ablation Support

Usage:
    # Standard-10 baseline
    python scripts/train_layoutlmv3.py --dataset funsd --epochs 10 --no_curriculum

    # Curriculum-10 (progressive 33->67->100)
    python scripts/train_layoutlmv3.py --dataset funsd --epochs 10 --curriculum --schedule progressive

    # Standard-7 matched-compute baseline
    python scripts/train_layoutlmv3.py --dataset funsd --epochs 7 --no_curriculum

    # Schedule ablations
    python scripts/train_layoutlmv3.py --dataset cord --epochs 10 --curriculum --schedule reverse
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import LayoutLMv3Trainer
from src.curriculum import CurriculumConfig
from src.training.utils import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train LayoutLMv3 with curriculum learning")
    parser.add_argument('--dataset', required=True,
                       choices=['funsd', 'cord', 'docvqa', 'financial', 'legal', 'technical'])
    parser.add_argument('--split', default='train')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--output_dir', default='results/layoutlmv3')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)

    # Curriculum arguments
    parser.add_argument('--curriculum', action='store_true', default=False,
                       help='Enable curriculum learning')
    parser.add_argument('--no_curriculum', action='store_true',
                       help='Disable curriculum learning (standard training)')
    parser.add_argument('--schedule', default='progressive',
                       choices=['progressive', 'two_phase', 'reverse', 'random', 'standard'],
                       help='Curriculum schedule type for ablation experiments')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    set_seed(args.seed)

    use_curriculum = args.curriculum and not args.no_curriculum

    condition = f"curriculum-{args.schedule}" if use_curriculum else "standard"
    logging.info(f"Config: {args.dataset} | LayoutLMv3 | {condition} | {args.epochs} epochs | seed={args.seed}")

    trainer = LayoutLMv3Trainer(output_dir=Path(args.output_dir), device=args.device)

    results = trainer.train(
        dataset_name=args.dataset,
        split=args.split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_samples=args.max_samples,
        use_curriculum=use_curriculum,
        schedule_type=args.schedule if use_curriculum else 'standard',
        seed=args.seed,
    )

    print(f"\nTraining completed: {results.get('status')}")
    print(f"Final loss: {results.get('final_loss', 'N/A')}")
    print(f"Total time: {results.get('total_time', 'N/A'):.2f}s")

    return results


if __name__ == "__main__":
    main()
