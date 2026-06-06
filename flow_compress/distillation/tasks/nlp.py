"""
Natural language understanding task support.
"""

# fad/tasks/nlp.py
from typing import Dict, Optional, Callable, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class NLPTask:
    """
    Task handler for natural language understanding.
    """

    def __init__(self, task_type: str = "classification", num_labels: int = 2):
        self.task_type = task_type
        self.num_labels = num_labels

        if task_type == "classification":
            self.loss_fn = nn.CrossEntropyLoss()
        elif task_type == "regression":
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def compute_metric(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        metric_type: str = "accuracy",
    ) -> float:
        """
        Computes NLP metric.
        """

        if metric_type == "accuracy":

            if self.task_type == "classification":
                preds = predictions.argmax(dim=1)
                correct = (preds == labels).float()
                return correct.mean().item()

            else:
                # For regression, use threshold-based accuracy
                preds = (predictions > 0.5).long()
                correct = (preds == labels).float()
                return correct.mean().item()

        elif metric_type == "f1":
            # F1 score for classification
            preds = predictions.argmax(dim=1)
            tp = ((preds == 1) & (labels == 1)).sum().float()
            fp = ((preds == 1) & (labels == 0)).sum().float()
            fn = ((preds == 0) & (labels == 1)).sum().float()
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            return f1.item()

        else:
            raise ValueError(f"Unknown metric type: {metric_type}")

    def compute_loss(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes NLP loss.
        """

        return self.loss_fn(predictions, labels)

    def get_metric_fn(self, metric_type: str = "accuracy") -> Callable:
        """
        Returns a metric function for optimization.
        """

        def metric_fn(predictions, labels):
            return self.compute_metric(predictions, labels, metric_type)

        return metric_fn
