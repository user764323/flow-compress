"""
Training modules for FAD framework.
"""

from flow_compress.distillation.trainer.attention_transfer_trainer import AttentionTransferFADTrainer
from flow_compress.distillation.trainer.contrastive_distillation_trainer import (
    ContrastiveDistillationFADTrainer,
)
from flow_compress.distillation.trainer.curriculum_fad_trainer import CurriculumFADTrainer
from flow_compress.distillation.trainer.fad_trainer import FADFeatureDistiller, FADTrainer, distill
from flow_compress.distillation.trainer.generative_distillation_trainer import GenerativeDistillationFADTrainer
from flow_compress.distillation.trainer.graph_distillation import GraphDistillationTrainer
from flow_compress.distillation.trainer.joint_distillation import JointDistillationTrainer

try:
    from flow_compress.distillation.trainer.lightning_callbacks import (
        CurriculumLearningScheduler,
        FADMetricsCallback,
    )

    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False

__all__ = [
    "FADTrainer",
    "FADFeatureDistiller",
    "distill",
    "JointDistillationTrainer",
    "GraphDistillationTrainer",
    "CurriculumFADTrainer",
    "AttentionTransferFADTrainer",
    "ContrastiveDistillationFADTrainer",
    "GenerativeDistillationFADTrainer",
]

if LIGHTNING_AVAILABLE:
    __all__.extend(
        [
            "CurriculumLearningScheduler",
            "FADMetricsCallback",
        ]
    )
