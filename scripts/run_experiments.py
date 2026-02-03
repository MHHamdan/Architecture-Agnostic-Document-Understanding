#!/usr/bin/env python3
"""
Comprehensive Experiment Runner for Architecture-Agnostic Curriculum Learning

Runs all experiments needed for the paper revision:
  Phase 1: Primary experiments (Standard-10, Curriculum-10, Standard-7)
  Phase 2: Schedule ablations (two-phase, reverse, random)
  Phase 3: Extended domain evaluation

For each condition: 2 architectures x N datasets x 3 seeds

Results are aggregated into summary JSON and printed tables at the end.

Usage:
    python scripts/run_experiments.py                  # Full suite
    python scripts/run_experiments.py --quick           # Single seed, quick test
    python scripts/run_experiments.py --phase 1         # Only phase 1
    python scripts/run_experiments.py --models bert     # Only BERT
    python scripts/run_experiments.py --datasets funsd  # Only FUNSD
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.models import BERTTrainer, LayoutLMv3Trainer
from src.training.utils import set_seed

logger = logging.getLogger(__name__)


def run_experiment(model_type, dataset_name, epochs, batch_size, use_curriculum,
                   schedule_type, seed, output_dir):
    """Run a single experiment and return results."""
    condition = f"{'curriculum-' + schedule_type if use_curriculum else 'standard'}_ep{epochs}_seed{seed}"
    logger.info(f"\n{'='*70}")
    logger.info(f"EXPERIMENT: {model_type} / {dataset_name} / {condition}")
    logger.info(f"{'='*70}")

    try:
        if model_type == 'bert':
            trainer = BERTTrainer(output_dir=Path(output_dir) / 'bert')
            results = trainer.train(
                dataset_name=dataset_name,
                epochs=epochs,
                batch_size=batch_size,
                use_curriculum=use_curriculum,
                schedule_type=schedule_type if use_curriculum else 'standard',
                seed=seed,
            )
        elif model_type == 'layoutlmv3':
            trainer = LayoutLMv3Trainer(output_dir=Path(output_dir) / 'layoutlmv3')
            results = trainer.train(
                dataset_name=dataset_name,
                epochs=epochs,
                batch_size=batch_size,
                use_curriculum=use_curriculum,
                schedule_type=schedule_type if use_curriculum else 'standard',
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return results

    except Exception as e:
        logger.error(f"EXPERIMENT FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'failed',
            'error': str(e),
            'model': model_type,
            'dataset': dataset_name,
            'condition': condition,
        }


def aggregate_results(all_results, output_dir):
    """Aggregate results across seeds and produce summary tables."""
    import numpy as np

    summary = {}

    for result in all_results:
        if result.get('status') != 'completed':
            continue

        key = f"{result['model']}_{result['dataset']}_{result.get('schedule_type', 'standard')}_ep{len(result.get('epochs', []))}"
        if result.get('curriculum_enabled'):
            key = f"{result['model']}_{result['dataset']}_curriculum-{result.get('schedule_type', 'progressive')}_ep{len(result.get('epochs', []))}"
        else:
            key = f"{result['model']}_{result['dataset']}_standard_ep{len(result.get('epochs', []))}"

        if key not in summary:
            summary[key] = {
                'model': result['model'],
                'dataset': result['dataset'],
                'condition': 'curriculum' if result.get('curriculum_enabled') else 'standard',
                'schedule': result.get('schedule_type', 'standard'),
                'epochs': len(result.get('epochs', [])),
                'f1_scores': [],
                'times': [],
                'final_losses': [],
            }

        summary[key]['f1_scores'].append(result.get('entity_f1', 0.0))
        summary[key]['times'].append(result.get('total_time', 0.0))
        summary[key]['final_losses'].append(result.get('final_loss', 0.0))

    # Compute statistics
    for key, data in summary.items():
        f1s = np.array(data['f1_scores'])
        times = np.array(data['times'])
        losses = np.array(data['final_losses'])

        data['f1_mean'] = float(np.mean(f1s)) if len(f1s) > 0 else 0.0
        data['f1_std'] = float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0
        data['time_mean'] = float(np.mean(times)) if len(times) > 0 else 0.0
        data['time_std'] = float(np.std(times, ddof=1)) if len(times) > 1 else 0.0
        data['loss_mean'] = float(np.mean(losses)) if len(losses) > 0 else 0.0
        data['n_seeds'] = len(f1s)

    # Save summary
    summary_path = Path(output_dir) / "experiment_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print("\n" + "=" * 90)
    print("EXPERIMENT SUMMARY")
    print("=" * 90)
    print(f"{'Key':<55} {'F1 Mean':>8} {'F1 Std':>8} {'Time':>8} {'Seeds':>5}")
    print("-" * 90)
    for key, data in sorted(summary.items()):
        print(f"{key:<55} {data['f1_mean']:>8.4f} {data['f1_std']:>8.4f} "
              f"{data['time_mean']:>8.1f} {data['n_seeds']:>5}")
    print("=" * 90)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run full experiment suite")
    parser.add_argument('--quick', action='store_true', help='Quick test: 1 seed only')
    parser.add_argument('--phase', type=int, default=0,
                       help='Run specific phase (1=primary, 2=ablations, 3=extended, 0=all)')
    parser.add_argument('--models', nargs='+', default=['bert', 'layoutlmv3'],
                       choices=['bert', 'layoutlmv3'])
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Specific datasets to run')
    parser.add_argument('--output_dir', default='results',
                       help='Output directory for results')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{args.output_dir}/experiment_log.txt", mode='a'),
        ]
    )

    seeds = [42] if args.quick else args.seeds
    models = args.models
    primary_datasets = args.datasets or ['funsd', 'cord']

    all_results = []
    start_time = time.time()

    logger.info("=" * 70)
    logger.info("ARCHITECTURE-AGNOSTIC CURRICULUM LEARNING - EXPERIMENT SUITE")
    logger.info(f"Models: {models}")
    logger.info(f"Datasets: {primary_datasets}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 70)

    # ============================================================
    # Phase 1: Primary experiments
    # ============================================================
    if args.phase in [0, 1]:
        logger.info("\n\n=== PHASE 1: Primary Experiments ===\n")

        batch_sizes = {'bert': 16, 'layoutlmv3': 4}

        for ds in primary_datasets:
            for model in models:
                bs = batch_sizes[model]
                for seed in seeds:
                    # Standard-10
                    result = run_experiment(
                        model, ds, epochs=10, batch_size=bs,
                        use_curriculum=False, schedule_type='standard',
                        seed=seed, output_dir=args.output_dir
                    )
                    all_results.append(result)

                    # Curriculum-10 (progressive)
                    result = run_experiment(
                        model, ds, epochs=10, batch_size=bs,
                        use_curriculum=True, schedule_type='progressive',
                        seed=seed, output_dir=args.output_dir
                    )
                    all_results.append(result)

                    # Standard-7 (matched compute)
                    result = run_experiment(
                        model, ds, epochs=7, batch_size=bs,
                        use_curriculum=False, schedule_type='standard',
                        seed=seed, output_dir=args.output_dir
                    )
                    all_results.append(result)

    # ============================================================
    # Phase 2: Schedule ablations (CORD only)
    # ============================================================
    if args.phase in [0, 2]:
        logger.info("\n\n=== PHASE 2: Schedule Ablations (CORD) ===\n")

        ablation_schedules = ['two_phase', 'reverse', 'random']
        batch_sizes = {'bert': 16, 'layoutlmv3': 4}

        for model in models:
            bs = batch_sizes[model]
            for schedule in ablation_schedules:
                for seed in seeds:
                    result = run_experiment(
                        model, 'cord', epochs=10, batch_size=bs,
                        use_curriculum=True, schedule_type=schedule,
                        seed=seed, output_dir=args.output_dir
                    )
                    all_results.append(result)

    # ============================================================
    # Phase 3: Extended domain
    # ============================================================
    if args.phase in [0, 3]:
        logger.info("\n\n=== PHASE 3: Extended Domain ===\n")

        ext_datasets = ['docvqa', 'financial', 'legal', 'technical']
        batch_sizes = {'bert': 16, 'layoutlmv3': 4}

        for ds in ext_datasets:
            for model in models:
                bs = batch_sizes[model]
                for seed in seeds:
                    # Curriculum
                    result = run_experiment(
                        model, ds, epochs=10, batch_size=bs,
                        use_curriculum=True, schedule_type='progressive',
                        seed=seed, output_dir=args.output_dir
                    )
                    all_results.append(result)

                    # Standard-7 matched
                    result = run_experiment(
                        model, ds, epochs=7, batch_size=bs,
                        use_curriculum=False, schedule_type='standard',
                        seed=seed, output_dir=args.output_dir
                    )
                    all_results.append(result)

    # ============================================================
    # Aggregate and summarize
    # ============================================================
    total_time = time.time() - start_time

    # Save all individual results
    all_results_path = Path(args.output_dir) / "all_results.json"
    with open(all_results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    summary = aggregate_results(all_results, args.output_dir)

    logger.info(f"\nTotal experiment time: {total_time:.1f}s ({total_time/3600:.2f}h)")
    logger.info(f"Results saved to: {args.output_dir}")

    completed = sum(1 for r in all_results if r.get('status') == 'completed')
    failed = sum(1 for r in all_results if r.get('status') == 'failed')
    logger.info(f"Completed: {completed}, Failed: {failed}")

    return all_results


if __name__ == "__main__":
    main()
