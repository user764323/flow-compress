"""
Object detection task support.
"""

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionTask:
    """
    Task handler for object detection.
    """

    def __init__(self, num_classes: int):
        """
        Args:
            num_classes: Number of object classes
        """
        self.num_classes = num_classes

    def compute_metric(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        metric_type: str = "mAP",
    ) -> float:
        """
        Computes detection metric.
        """
        if metric_type == "mAP":
            if "boxes" in predictions and "boxes" in targets:
                iou = self._compute_iou(predictions["boxes"], targets["boxes"])
                return iou.mean().item() if iou.numel() > 0 else 0.0
            return 0.0
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")

    def _compute_iou(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes IoU between boxes.
        """

        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        # Compute intersection
        inter_x1 = torch.max(boxes1[:, 0].unsqueeze(
            1), boxes2[:, 0].unsqueeze(0))
        inter_y1 = torch.max(boxes1[:, 1].unsqueeze(
            1), boxes2[:, 1].unsqueeze(0))
        inter_x2 = torch.min(boxes1[:, 2].unsqueeze(
            1), boxes2[:, 2].unsqueeze(0))
        inter_y2 = torch.min(boxes1[:, 3].unsqueeze(
            1), boxes2[:, 3].unsqueeze(0))

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(
            inter_y2 - inter_y1, min=0
        )
        union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area

        iou = inter_area / (union_area + 1e-8)
        return iou

    def get_metric_fn(self, metric_type: str = "mAP") -> Callable:
        """
        Returns a metric function for optimization.
        """

        def metric_fn(predictions, targets):
            return self.compute_metric(predictions, targets, metric_type)

        return metric_fn
