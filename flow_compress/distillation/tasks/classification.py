"""
Image classification task support.
"""

from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationTask:
    """
    Task handler for image classification.
    """

    def __init__(self, num_classes: int):
        """
        Args:
            num_classes: Number of classes
        """
        self.num_classes = num_classes
        self.loss_fn = nn.CrossEntropyLoss()

    def compute_metric(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        metric_type: str = "accuracy",
    ) -> float:
        """
        Computes classification metric.
        """

        if metric_type == "accuracy":
            preds = logits.argmax(dim=1)
            correct = (preds == labels).float()
            return correct.mean().item()

        elif metric_type == "top5_accuracy":
            _, top5_preds = logits.topk(5, dim=1)
            correct = top5_preds.eq(labels.view(-1, 1).expand_as(top5_preds))
            return correct.any(dim=1).float().mean().item()

        else:
            raise ValueError(f"Unknown metric type: {metric_type}")

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes classification loss.
        """

        return self.loss_fn(logits, labels)

    def get_metric_fn(self, metric_type: str = "accuracy") -> Callable:
        """
        Returns a metric function for optimization.
        """

        def metric_fn(logits, labels):
            return self.compute_metric(logits, labels, metric_type)

        return metric_fn
