# Figures

This directory contains figures for the Architecture-Agnostic Document Understanding paper.

## Architecture Diagram

The main architecture diagram is included in the README.md as ASCII art for maximum compatibility.

## Generated Figures

After running experiments, the following figures will be generated:

- `loss_convergence.png` - Training loss curves for BERT vs LayoutLMv3
- `curriculum_phases.png` - Curriculum learning phase visualization
- `performance_comparison.png` - Cross-architecture performance comparison
- `training_efficiency.png` - Training time and memory analysis

## Generating Figures

```bash
python scripts/generate_figures.py --results_dir results/
```
