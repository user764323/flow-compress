"""
FAD: Flow-Aligned Knowledge Distillation
"""

__version__ = "1.0.0"

from flow_compress.distillation.flows import (
    compute_dalign,
    compute_fad_loss_integrated,
    compute_layerwise_flow_divergence,
    selective_flow_alignment,
)
from flow_compress.distillation.losses.fad_loss import FADLoss
from flow_compress.distillation.losses.kd_logits import LogitsKDLoss
from flow_compress.distillation.models.student_wrapper import StudentWrapper
from flow_compress.distillation.models.teacher_wrapper import TeacherWrapper
from flow_compress.distillation.trainer.curriculum_fad_trainer import CurriculumFADTrainer
from flow_compress.distillation.trainer.fad_distillation import fad_distillation
from flow_compress.distillation.trainer.fad_trainer import FADTrainer
from flow_compress.distillation.trainer.joint_distillation import JointDistillationTrainer
from flow_compress.utils import (
    adapt_distillation_targets,
    calibrate_distillation_weights,
    compute_capacity_alignment_score,
    compute_information_sensitivity,
    compute_task_driven_fidelity,
    select_best_student_architecture,
)

__all__ = [
    "__version__",
    "FADTrainer",
    "fad_distillation",
    "JointDistillationTrainer",
    "CurriculumFADTrainer",
    "FADLoss",
    "LogitsKDLoss",
    "TeacherWrapper",
    "StudentWrapper",
    "compute_dalign",
    "compute_fad_loss_integrated",
    "compute_layerwise_flow_divergence",
    "selective_flow_alignment",
    "select_best_student_architecture",
    "compute_information_sensitivity",
    "compute_capacity_alignment_score",
    "calibrate_distillation_weights",
    "compute_task_driven_fidelity",
    "adapt_distillation_targets",
]
