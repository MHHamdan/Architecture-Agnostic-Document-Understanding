#!/usr/bin/env python3
"""
Evaluation Metrics for Document Understanding

Metrics:
- F1 Score: Entity recognition tasks (FUNSD)
- Accuracy: Classification tasks (Financial, Technical)
- ANLS: Average Normalized Levenshtein Similarity for VQA (DocVQA)
"""

import logging
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


def compute_f1(predictions: List[Any], labels: List[Any]) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 score.

    Args:
        predictions: Model predictions
        labels: Ground truth labels

    Returns:
        Dict with precision, recall, f1
    """
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have same length")

    if len(predictions) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Convert to sets for comparison
    pred_set = set(str(p) for p in predictions if p)
    label_set = set(str(l) for l in labels if l)

    if len(pred_set) == 0 and len(label_set) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    if len(pred_set) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if len(label_set) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    true_positives = len(pred_set & label_set)
    precision = true_positives / len(pred_set)
    recall = true_positives / len(label_set)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def compute_accuracy(predictions: List[Any], labels: List[Any]) -> float:
    """
    Compute accuracy score.

    Args:
        predictions: Model predictions
        labels: Ground truth labels

    Returns:
        Accuracy as float between 0 and 1
    """
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have same length")

    if len(predictions) == 0:
        return 0.0

    correct = sum(1 for p, l in zip(predictions, labels) if str(p) == str(l))
    return correct / len(predictions)


def normalized_levenshtein(s1: str, s2: str) -> float:
    """
    Compute normalized Levenshtein distance.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Normalized similarity (0 to 1, higher is more similar)
    """
    if len(s1) == 0 and len(s2) == 0:
        return 1.0

    if len(s1) == 0 or len(s2) == 0:
        return 0.0

    # Create distance matrix
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

    Used for Document VQA evaluation where multiple answers may be valid.

    Args:
        predictions: List of predicted answers
        labels: List of lists of valid answers
        threshold: Minimum similarity threshold (default 0.5)

    Returns:
        ANLS score (0 to 1)
    """
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have same length")

    if len(predictions) == 0:
        return 0.0

    scores = []

    for pred, valid_answers in zip(predictions, labels):
        pred_str = str(pred).lower().strip()

        # Find best match among valid answers
        max_score = 0.0
        for answer in valid_answers:
            answer_str = str(answer).lower().strip()
            score = normalized_levenshtein(pred_str, answer_str)
            max_score = max(max_score, score)

        # Apply threshold
        if max_score < threshold:
            max_score = 0.0

        scores.append(max_score)

    return np.mean(scores)


class Evaluator:
    """
    Unified evaluator for document understanding tasks.

    Supports multiple task types:
    - entity_recognition: F1 score
    - classification: Accuracy
    - visual_qa: ANLS
    - key_value_extraction: F1 score
    """

    def __init__(self):
        self.results = defaultdict(list)

    def evaluate(
        self,
        predictions: List[Any],
        labels: List[Any],
        task_type: str
    ) -> Dict[str, float]:
        """
        Evaluate predictions based on task type.

        Args:
            predictions: Model predictions
            labels: Ground truth labels
            task_type: Type of task (entity_recognition, classification, visual_qa)

        Returns:
            Dict with metric scores
        """
        if task_type in ["entity_recognition", "key_value_extraction"]:
            metrics = compute_f1(predictions, labels)
        elif task_type == "classification":
            metrics = {"accuracy": compute_accuracy(predictions, labels)}
        elif task_type == "visual_qa":
            # For VQA, labels should be list of valid answers
            if labels and isinstance(labels[0], list):
                metrics = {"anls": compute_anls(predictions, labels)}
            else:
                # Single answer, wrap in list
                labels_wrapped = [[l] for l in labels]
                metrics = {"anls": compute_anls(predictions, labels_wrapped)}
        else:
            # Default to accuracy
            metrics = {"accuracy": compute_accuracy(predictions, labels)}

        # Store results
        for k, v in metrics.items():
            self.results[k].append(v)

        return metrics

    def get_summary(self) -> Dict[str, float]:
        """Get summary of all evaluations"""
        summary = {}
        for metric, values in self.results.items():
            summary[f"{metric}_mean"] = np.mean(values)
            summary[f"{metric}_std"] = np.std(values)
        return summary

    def reset(self):
        """Reset stored results"""
        self.results = defaultdict(list)


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("Evaluation Metrics Demo")
    print("=" * 60)

    # F1 Score
    preds = ["entity1", "entity2", "entity3"]
    labels = ["entity1", "entity2", "entity4"]
    f1_result = compute_f1(preds, labels)
    print(f"\nF1 Score: {f1_result}")

    # Accuracy
    preds = [0, 1, 1, 0, 1]
    labels = [0, 1, 0, 0, 1]
    acc = compute_accuracy(preds, labels)
    print(f"Accuracy: {acc:.2%}")

    # ANLS
    preds = ["hello world", "test answer"]
    labels = [["hello world", "hello"], ["test", "answer"]]
    anls = compute_anls(preds, labels)
    print(f"ANLS: {anls:.4f}")

    # Evaluator
    evaluator = Evaluator()
    evaluator.evaluate(["A", "B", "C"], ["A", "B", "D"], "entity_recognition")
    print(f"\nEvaluator Summary: {evaluator.get_summary()}")
