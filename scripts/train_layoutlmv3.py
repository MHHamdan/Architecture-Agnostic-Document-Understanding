#!/usr/bin/env python3
"""
LayoutLMv3 Training Script with HCML Curriculum Learning

Usage:
    python scripts/train_layoutlmv3.py --dataset funsd --epochs 10 --batch_size 8
    python scripts/train_layoutlmv3.py --dataset cord --epochs 10 --curriculum
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import LayoutLMv3Trainer


def main():
    parser = argparse.ArgumentParser(
        description="Train LayoutLMv3 with HCML Curriculum Learning"
    )
    parser.add_argument(
        '--dataset', required=True,
        choices=['funsd', 'cord', 'docvqa', 'financial', 'legal', 'technical'],
        help='Dataset name'
    )
    parser.add_argument('--split', default='train', help='Dataset split')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples')
    parser.add_argument('--output_dir', default='results/layoutlmv3', help='Output directory')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--curriculum', action='store_true', default=True,
                       help='Use HCML curriculum learning')
    parser.add_argument('--no_curriculum', action='store_true',
                       help='Disable curriculum learning')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Use curriculum unless explicitly disabled
    use_curriculum = not args.no_curriculum

    # Initialize trainer
    trainer = LayoutLMv3Trainer(
        output_dir=Path(args.output_dir),
        device=args.device
    )

    # Train
    results = trainer.train(
        dataset_name=args.dataset,
        split=args.split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_samples=args.max_samples,
        use_curriculum=use_curriculum
    )

    print(f"\nTraining completed with status: {results.get('status')}")
    print(f"Final loss: {results.get('final_loss', 'N/A')}")

    return results


if __name__ == "__main__":
    main()
