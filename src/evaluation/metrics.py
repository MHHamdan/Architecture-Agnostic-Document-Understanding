#!/usr/bin/env python3
"""
Evaluation Metrics for Document Understanding

Metrics:
- Entity-level F1: Using seqeval for NER/SER tasks (FUNSD, CORD)
- Token-level F1: Per-token classification metrics
- Accuracy: Classification tasks (Financial, Technical)
- ANLS: Average Normalized Levenshtein Similarity for VQA (DocVQA)

Statistical analysis:
- Paired t-tests for same-seed comparisons
- Cohen's d_z for paired effect sizes
"""

import logging
import math
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

# Try to import seqeval for entity-level evaluation
try:
    from seqeval.metrics import f1_score as seqeval_f1
    from seqeval.metrics import precision_score as seqeval_precision
    from seqeval.metrics import recall_score as seqeval_recall
    from seqeval.metrics import classification_report as seqeval_report
    HAS_SEQEVAL = True
except ImportError:
    HAS_SEQEVAL = False
    logger.warning("seqeval not installed. Entity-level F1 will use token-level fallback. "
                   "Install with: pip install seqeval")


# ============================================================
# ENTITY-LEVEL F1 (seqeval-based)
# ============================================================

def compute_entity_f1(
    predictions: List[List[str]],
    labels: List[List[str]],
    scheme: str = "IOB2"
) -> Dict[str, float]:
    """
    Compute entity-level F1 using seqeval.

    This is the standard evaluation for NER/SER tasks (FUNSD, CORD).
    It evaluates complete entity spans, not individual tokens.

    Args:
        predictions: List of sequences of predicted IOB tags
        labels: List of sequences of gold IOB tags
        scheme: Tagging scheme ('IOB2', 'IOB1', 'IOBES')

    Returns:
        Dict with entity-level precision, recall, f1
    """
    if HAS_SEQEVAL:
        precision = seqeval_precision(labels, predictions)
        recall = seqeval_recall(labels, predictions)
        f1 = seqeval_f1(labels, predictions)
        return {
            "entity_precision": precision,
            "entity_recall": recall,
            "entity_f1": f1,
            "evaluation_type": "entity-level (seqeval)"
        }
    else:
        # Fallback to token-level
        logger.warning("Using token-level F1 as fallback (seqeval not available)")
        return compute_token_f1(
            [tag for seq in predictions for tag in seq],
            [tag for seq in labels for tag in seq]
        )


# ============================================================
# TOKEN-LEVEL F1 (fallback)
# ============================================================

def compute_token_f1(predictions: List[Any], labels: List[Any]) -> Dict[str, float]:
    """
    Compute token-level precision, recall, and F1 score.

    This is a fallback when seqeval is not available. Note that
    token-level F1 typically overestimates performance compared to
    entity-level F1.

    Args:
        predictions: Flat list of predicted labels
        labels: Flat list of gold labels

    Returns:
        Dict with token-level precision, recall, f1
    """
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have same length")

    if len(predictions) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "evaluation_type": "token-level"}

    # Exclude 'O' tags for entity-focused evaluation
    pred_entities = [(i, p) for i, p in enumerate(predictions) if str(p) != 'O' and str(p) != '0']
    gold_entities = [(i, l) for i, l in enumerate(labels) if str(l) != 'O' and str(l) != '0']

    pred_set = set(pred_entities)
    gold_set = set(gold_entities)

    if len(pred_set) == 0 and len(gold_set) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "evaluation_type": "token-level"}
    if len(pred_set) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "evaluation_type": "token-level"}
    if len(gold_set) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "evaluation_type": "token-level"}

    true_positives = len(pred_set & gold_set)
    precision = true_positives / len(pred_set)
    recall = true_positives / len(gold_set)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "evaluation_type": "token-level"
    }


# ============================================================
# ACCURACY
# ============================================================

def compute_accuracy(predictions: List[Any], labels: List[Any]) -> float:
    """Compute classification accuracy."""
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have same length")
    if len(predictions) == 0:
        return 0.0
    correct = sum(1 for p, l in zip(predictions, labels) if str(p) == str(l))
    return correct / len(predictions)


# ============================================================
# ANLS (for DocVQA)
# ============================================================

def normalized_levenshtein(s1: str, s2: str) -> float:
    """Compute normalized Levenshtein similarity (0 to 1)."""
    if len(s1) == 0 and len(s2) == 0:
        return 1.0
    if len(s1) == 0 or len(s2) == 0:
        return 0.0

    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    distance = dp[m][n]
    max_len = max(m, n)
    return 1.0 - (distance / max_len)


def compute_anls(predictions: List[str], labels: List[List[str]], threshold: float = 0.5) -> float:
    """
    Compute Average Normalized Levenshtein Similarity (ANLS).
    Used for Document VQA evaluation with multiple valid answers.
    """
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have same length")
    if len(predictions) == 0:
        return 0.0

    scores = []
    for pred, valid_answers in zip(predictions, labels):
        pred_str = str(pred).lower().strip()
        max_score = 0.0
        for answer in valid_answers:
            answer_str = str(answer).lower().strip()
            score = normalized_levenshtein(pred_str, answer_str)
            max_score = max(max_score, score)
        if max_score < threshold:
            max_score = 0.0
        scores.append(max_score)

    return float(np.mean(scores))


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

def paired_ttest(values_a: List[float], values_b: List[float]) -> Dict[str, float]:
    """
    Compute paired t-test and Cohen's d_z for paired designs.

    For experiments sharing random seeds, paired analysis is more
    appropriate and powerful than independent t-tests.

    Cohen's d_z = mean(differences) / std(differences) = t / sqrt(n)

    Args:
        values_a: Measurements from condition A (e.g., curriculum times)
        values_b: Measurements from condition B (e.g., standard times)

    Returns:
        Dict with mean_diff, std_diff, t_statistic, p_value, cohens_dz
    """
    if len(values_a) != len(values_b):
        raise ValueError("Both conditions must have same number of observations")

    n = len(values_a)
    if n < 2:
        return {
            "mean_diff": float(np.mean(np.array(values_a) - np.array(values_b))),
            "std_diff": 0.0,
            "t_statistic": float('inf'),
            "p_value": 0.0,
            "cohens_dz": float('inf'),
            "n": n,
            "note": "n < 2, statistics unreliable"
        }

    differences = np.array(values_a) - np.array(values_b)
    mean_diff = float(np.mean(differences))
    std_diff = float(np.std(differences, ddof=1))

    if std_diff == 0:
        t_stat = float('inf') if mean_diff != 0 else 0.0
        p_value = 0.0 if mean_diff != 0 else 1.0
    else:
        se = std_diff / math.sqrt(n)
        t_stat = mean_diff / se
        # Two-tailed p-value approximation using t-distribution
        # For n=3 (df=2), use conservative approximation
        df = n - 1
        # Approximate p-value (for exact, use scipy.stats.t.sf)
        try:
            from scipy import stats
            p_value = float(stats.t.sf(abs(t_stat), df) * 2)
        except ImportError:
            # Conservative approximation for small df
            p_value = 0.001 if abs(t_stat) > 4.303 else 0.05  # df=2 thresholds

    # Cohen's d_z for paired designs: d_z = t / sqrt(n)
    if n > 0 and not math.isinf(t_stat):
        cohens_dz = t_stat / math.sqrt(n)
    else:
        cohens_dz = float('inf')

    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_dz": cohens_dz,
        "n": n,
    }


def compute_speedup(curriculum_times: List[float], standard_times: List[float]) -> Dict[str, float]:
    """
    Compute speedup percentage with statistical analysis.

    Args:
        curriculum_times: Training times under curriculum condition
        standard_times: Training times under standard condition

    Returns:
        Dict with speedup_pct, mean/std for both conditions, and paired statistics
    """
    curr_mean = float(np.mean(curriculum_times))
    curr_std = float(np.std(curriculum_times, ddof=1)) if len(curriculum_times) > 1 else 0.0
    std_mean = float(np.mean(standard_times))
    std_std = float(np.std(standard_times, ddof=1)) if len(standard_times) > 1 else 0.0

    speedup_pct = (std_mean - curr_mean) / std_mean * 100.0

    stats = paired_ttest(curriculum_times, standard_times)

    return {
        "curriculum_mean": curr_mean,
        "curriculum_std": curr_std,
        "standard_mean": std_mean,
        "standard_std": std_std,
        "speedup_pct": speedup_pct,
        **stats
    }


# ============================================================
# UNIFIED EVALUATOR
# ============================================================

class Evaluator:
    """
    Unified evaluator for document understanding tasks.

    Supports:
    - entity_recognition: Entity-level F1 (seqeval)
    - classification: Accuracy
    - visual_qa: ANLS
    - key_value_extraction: Entity-level F1
    """

    def __init__(self):
        self.results = defaultdict(list)

    def evaluate(
        self,
        predictions: List[Any],
        labels: List[Any],
        task_type: str
    ) -> Dict[str, float]:
        """Evaluate predictions based on task type."""
        if task_type in ["entity_recognition", "key_value_extraction"]:
            # Check if inputs are sequences of tag sequences (for seqeval)
            if predictions and isinstance(predictions[0], list):
                metrics = compute_entity_f1(predictions, labels)
            else:
                metrics = compute_token_f1(predictions, labels)
        elif task_type == "classification":
            metrics = {"accuracy": compute_accuracy(predictions, labels)}
        elif task_type == "visual_qa":
            if labels and isinstance(labels[0], list):
                metrics = {"anls": compute_anls(predictions, labels)}
            else:
                labels_wrapped = [[l] for l in labels]
                metrics = {"anls": compute_anls(predictions, labels_wrapped)}
        else:
            metrics = {"accuracy": compute_accuracy(predictions, labels)}

        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.results[k].append(v)

        return metrics

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics of all evaluations."""
        summary = {}
        for metric, values in self.results.items():
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values))
        return summary

    def reset(self):
        """Reset stored results."""
        self.results = defaultdict(list)


if __name__ == "__main__":
    print("=" * 60)
    print("Evaluation Metrics Demo")
    print("=" * 60)

    # Entity-level F1 (seqeval)
    if HAS_SEQEVAL:
        preds = [["B-PER", "I-PER", "O", "B-LOC"]]
        labels = [["B-PER", "I-PER", "O", "B-LOC"]]
        result = compute_entity_f1(preds, labels)
        print(f"\nEntity F1: {result}")
    else:
        print("\nseqeval not installed, skipping entity-level demo")

    # Token-level F1
    preds = ["B-PER", "I-PER", "O", "B-LOC"]
    labels = ["B-PER", "I-PER", "O", "B-ORG"]
    result = compute_token_f1(preds, labels)
    print(f"Token F1: {result}")

    # Statistical analysis
    curr_times = [16.8, 16.9, 17.0]
    std_times = [26.0, 26.0, 26.0]
    stats = compute_speedup(curr_times, std_times)
    print(f"\nSpeedup analysis:")
    print(f"  Speedup: {stats['speedup_pct']:.1f}%")
    print(f"  t-stat: {stats['t_statistic']:.2f}")
    print(f"  Cohen's d_z: {stats['cohens_dz']:.2f}")
    print(f"  p-value: {stats['p_value']}")
