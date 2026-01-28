"""
Task-specific modules for different domains.
"""

from flow_compress.distillation.tasks.classification import ClassificationTask
from flow_compress.distillation.tasks.detection import DetectionTask
from flow_compress.distillation.tasks.nlp import NLPTask

__all__ = [
    "ClassificationTask",
    "DetectionTask",
    "NLPTask",
]
