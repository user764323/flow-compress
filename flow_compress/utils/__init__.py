"""
Utility functions for flow_compress.
"""

from . import pruning_utils, quantization_utils
from flow_compress.utils.architecture_selection import (
    compute_capacity_alignment_score,
    compute_information_sensitivity,
    evaluate_student_candidate,
    select_best_student_architecture,
)
from flow_compress.utils.config import (
    CheckpointConfig,
    CurriculumConfig,
    DatasetConfig,
    ExperimentConfig,
    FADTrainerConfig,
    LoggingConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    get_cifar10_config,
    get_default_config,
    get_imagenet_config,
)
from flow_compress.utils.find_optimal_candidate import (
    find_optimal_student,
    find_optimal_student_with_constraints,
    rank_distillation_candidates,
    search_best_distillation_candidate,
    select_optimal_candidate_for_distillation,
)
from flow_compress.utils.functional_geometry import FunctionalGeometryAnalyzer
from flow_compress.utils.gradient_calibration import (
    adaptive_temperature_calibration,
    calibrate_distillation_weights,
    compute_gradient_magnitude,
    compute_gradient_sensitivity,
)
from flow_compress.utils.optimization_criteria import StudentSelectionOptimizer
from flow_compress.utils.representational_fidelity import (
    adapt_distillation_targets,
    compute_information_preservation,
    compute_representational_similarity,
    compute_task_driven_fidelity,
)
from flow_compress.utils.representational_geometry import RepresentationalGeometryAnalyzer

try:
    from flow_compress.utils.wandb_logger import WandBLogger, create_wandb_logger

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from flow_compress.utils.hydra_config import (
        HydraConfigManager,
        create_default_configs,
    )

    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False



__all__ = [
    "pruning_utils",
    "quantization_utils",
    "compute_capacity_alignment_score",
    "compute_information_sensitivity",
    "evaluate_student_candidate",
    "select_best_student_architecture",
    "compute_gradient_magnitude",
    "compute_gradient_sensitivity",
    "calibrate_distillation_weights",
    "adaptive_temperature_calibration",
    "compute_representational_similarity",
    "compute_task_driven_fidelity",
    "compute_information_preservation",
    "adapt_distillation_targets",
    "FunctionalGeometryAnalyzer",
    "RepresentationalGeometryAnalyzer",
    "StudentSelectionOptimizer",
    "find_optimal_student",
    "search_best_distillation_candidate",
    "select_optimal_candidate_for_distillation",
    "rank_distillation_candidates",
    "find_optimal_student_with_constraints",
    "ExperimentConfig",
    "ModelConfig",
    "DatasetConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "FADTrainerConfig",
    "CurriculumConfig",
    "LoggingConfig",
    "CheckpointConfig",
    "get_default_config",
    "get_cifar10_config",
    "get_imagenet_config",
]

if WANDB_AVAILABLE:
    __all__.extend(
        [
            "WandBLogger",
            "create_wandb_logger",
        ]
    )

if HYDRA_AVAILABLE:
    __all__.extend(
        [
            "HydraConfigManager",
            "create_default_configs",
        ]
    )
