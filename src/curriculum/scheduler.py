#!/usr/bin/env python3
"""
Curriculum Scheduler with Ablation Support

Implements multiple scheduling strategies:
- Progressive (33%->67%->100%): The proposed curriculum
- Two-phase (50%->100%): Simpler progressive schedule
- Reverse (100%->67%->33%): Anti-curriculum for ablation
- Random pacing: Random ratio each epoch for ablation
- Standard: Full data every epoch (baseline)
- Standard-matched: Full data for matched compute epochs

Key design: All schedules operate at the data loading level,
independent of model architecture, optimizer, or input modality.
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning"""

    # Phase boundaries (as fraction of total epochs)
    phase1_end: float = 0.3   # Easy phase ends at 30% of training
    phase2_end: float = 0.7   # Medium phase ends at 70% of training

    # Data sampling ratios
    easy_ratio: float = 0.33
    medium_ratio: float = 0.67
    hard_ratio: float = 1.0

    # Schedule type: 'progressive', 'two_phase', 'reverse', 'random', 'standard'
    schedule_type: str = 'progressive'


class HierarchicalCurriculumScheduler:
    """
    Curriculum scheduler with support for multiple scheduling strategies.

    Supports ablation experiments by providing different pacing functions
    with approximately matched total data exposure.

    Args:
        total_epochs: Total number of training epochs
        config: CurriculumConfig with phase boundaries and ratios
    """

    def __init__(self, total_epochs: int, config: Optional[CurriculumConfig] = None):
        self.total_epochs = total_epochs
        self.config = config or CurriculumConfig()
        self.schedule_type = self.config.schedule_type

        # Phase boundaries in epochs
        self.phase_boundaries = [
            self.config.phase1_end,
            self.config.phase2_end,
            1.0
        ]

        # Pre-compute random schedule if needed
        self._random_ratios = None
        if self.schedule_type == 'random':
            self._random_ratios = [
                random.choice([self.config.easy_ratio, self.config.medium_ratio, self.config.hard_ratio])
                for _ in range(total_epochs)
            ]

    def get_difficulty_level(self, epoch: int) -> str:
        """Get difficulty level for current epoch."""
        progress = (epoch + 1) / self.total_epochs

        if self.schedule_type == 'standard':
            return "standard"
        elif self.schedule_type == 'random':
            return "random"
        elif self.schedule_type == 'reverse':
            # Reversed phase ordering
            if progress <= self.phase_boundaries[0]:
                return "hard"
            elif progress <= self.phase_boundaries[1]:
                return "medium"
            else:
                return "easy"
        elif self.schedule_type == 'two_phase':
            if progress <= 0.5:
                return "easy"
            else:
                return "hard"
        else:
            # Progressive (default)
            if progress <= self.phase_boundaries[0]:
                return "easy"
            elif progress <= self.phase_boundaries[1]:
                return "medium"
            else:
                return "hard"

    def get_sample_ratio(self, epoch: int) -> float:
        """Get data sampling ratio for current epoch."""
        if self.schedule_type == 'standard':
            return 1.0
        elif self.schedule_type == 'random':
            return self._random_ratios[epoch]
        elif self.schedule_type == 'reverse':
            difficulty = self.get_difficulty_level(epoch)
            ratio_map = {
                "hard": self.config.hard_ratio,
                "medium": self.config.medium_ratio,
                "easy": self.config.easy_ratio,
            }
            return ratio_map[difficulty]
        elif self.schedule_type == 'two_phase':
            if (epoch + 1) / self.total_epochs <= 0.5:
                return 0.50
            else:
                return 1.0
        else:
            # Progressive
            difficulty = self.get_difficulty_level(epoch)
            ratio_map = {
                "easy": self.config.easy_ratio,
                "medium": self.config.medium_ratio,
                "hard": self.config.hard_ratio,
            }
            return ratio_map[difficulty]

    def get_phase_info(self, epoch: int) -> dict:
        """Get complete phase information for current epoch."""
        difficulty = self.get_difficulty_level(epoch)
        sample_ratio = self.get_sample_ratio(epoch)
        progress = (epoch + 1) / self.total_epochs

        phase_map = {"easy": 1, "medium": 2, "hard": 3, "standard": 0, "random": -1}

        return {
            "epoch": epoch + 1,
            "difficulty": difficulty,
            "phase": phase_map.get(difficulty, 0),
            "sample_ratio": sample_ratio,
            "progress": progress,
            "data_percentage": f"{sample_ratio * 100:.0f}%",
            "schedule_type": self.schedule_type,
        }

    def get_effective_epochs(self) -> float:
        """Compute total effective epoch-equivalents for this schedule."""
        total = 0.0
        for epoch in range(self.total_epochs):
            total += self.get_sample_ratio(epoch)
        return total

    def __repr__(self) -> str:
        eff = self.get_effective_epochs()
        return (
            f"HierarchicalCurriculumScheduler("
            f"type={self.schedule_type}, "
            f"epochs={self.total_epochs}, "
            f"effective_epochs={eff:.2f})"
        )


if __name__ == "__main__":
    print("=" * 70)
    print("Curriculum Scheduler Demo - All Schedule Types")
    print("=" * 70)

    schedule_types = ['progressive', 'two_phase', 'reverse', 'random', 'standard']

    for stype in schedule_types:
        config = CurriculumConfig(schedule_type=stype)
        scheduler = HierarchicalCurriculumScheduler(total_epochs=10, config=config)
        print(f"\n{scheduler}")
        print(f"  Effective epochs: {scheduler.get_effective_epochs():.2f}")
        print(f"  Epoch | Ratio | Difficulty")
        print(f"  " + "-" * 35)
        for epoch in range(10):
            info = scheduler.get_phase_info(epoch)
            print(f"    {info['epoch']:2d}  |  {info['data_percentage']:4s} | {info['difficulty']}")
