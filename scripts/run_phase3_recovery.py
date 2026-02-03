#!/usr/bin/env python3
"""Recovery script for Phase 3 failed experiments.
Runs each experiment as a separate subprocess to avoid GPU memory accumulation."""

import subprocess
import sys
import json
from pathlib import Path

OUTPUT_DIR = "results/phase3_extended"

# All Phase 3 experiments
EXPERIMENTS = []
for ds in ['docvqa', 'financial', 'legal', 'technical']:
    for model in ['bert', 'layoutlmv3']:
        for cond in [('curriculum-progressive', True, 'progressive', 10),
                     ('standard', False, 'standard', 7)]:
            for seed in [42, 123, 456]:
                prefix, use_curr, sched, epochs = cond
                result_dir = f"{prefix}_ep{epochs}_seed{seed}"
                result_path = Path(OUTPUT_DIR) / model / ds / result_dir / "training_results.json"
                if not result_path.exists():
                    EXPERIMENTS.append({
                        'model': model, 'dataset': ds, 'use_curriculum': use_curr,
                        'schedule': sched, 'epochs': epochs, 'seed': seed,
                        'label': f"{model}/{ds}/{result_dir}"
                    })

print(f"Missing experiments: {len(EXPERIMENTS)}")
for exp in EXPERIMENTS:
    print(f"  {exp['label']}")
print()

completed = 0
failed = 0

for i, exp in enumerate(EXPERIMENTS):
    print(f"\n{'='*70}")
    print(f"[{i+1}/{len(EXPERIMENTS)}] {exp['label']}")
    print(f"{'='*70}")

    script = f"""
import os, sys, torch, gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, '.')
from src.models import BERTTrainer, LayoutLMv3Trainer
from src.training.utils import set_seed
from pathlib import Path

torch.cuda.empty_cache()
gc.collect()

model_type = '{exp["model"]}'
dataset = '{exp["dataset"]}'
seed = {exp["seed"]}
epochs = {exp["epochs"]}
use_curriculum = {exp["use_curriculum"]}
schedule = '{exp["schedule"]}'
output_dir = '{OUTPUT_DIR}'

batch_sizes = {{'bert': 16, 'layoutlmv3': 4}}
bs = batch_sizes[model_type]

if model_type == 'bert':
    trainer = BERTTrainer(output_dir=Path(output_dir) / 'bert')
    results = trainer.train(dataset_name=dataset, epochs=epochs, batch_size=bs,
                           use_curriculum=use_curriculum, schedule_type=schedule, seed=seed)
else:
    trainer = LayoutLMv3Trainer(output_dir=Path(output_dir) / 'layoutlmv3')
    results = trainer.train(dataset_name=dataset, epochs=epochs, batch_size=bs,
                           use_curriculum=use_curriculum, schedule_type=schedule, seed=seed)

print(f"STATUS: {{results.get('status', 'unknown')}}")
print(f"F1: {{results.get('entity_f1', 0):.4f}}")
"""

    result = subprocess.run(
        [sys.executable, '-W', 'ignore', '-c', script],
        capture_output=True, text=True, timeout=600,
        env={**__import__('os').environ, 'TF_CPP_MIN_LOG_LEVEL': '3',
             'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}
    )

    if result.returncode == 0 and 'STATUS: completed' in result.stdout:
        completed += 1
        for line in result.stdout.strip().split('\n')[-3:]:
            print(f"  {line}")
    else:
        failed += 1
        print(f"  FAILED (rc={result.returncode})")
        if result.stderr:
            # Print last few lines of stderr
            for line in result.stderr.strip().split('\n')[-5:]:
                print(f"  {line}")

print(f"\n{'='*70}")
print(f"Recovery complete: {completed} succeeded, {failed} failed")
print(f"{'='*70}")
