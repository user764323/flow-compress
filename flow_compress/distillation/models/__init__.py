"""
Model wrappers and adapters for FAD framework.
"""

from flow_compress.distillation.models.student_wrapper import StudentWrapper
from flow_compress.distillation.models.teacher_wrapper import TeacherWrapper

try:
    from flow_compress.distillation.models.hf_adapters import (
        TrajectoryAlignmentAdapter,
        ViTAdapter,
        ResNetAdapter,
        CLIPViTAdapter,
        LLaVAAdapter,
        ImageBindAdapter,
        create_hf_adapter,
    )

    HF_ADAPTERS_AVAILABLE = True
except ImportError:
    HF_ADAPTERS_AVAILABLE = False

__all__ = [
    "TeacherWrapper",
    "StudentWrapper",
]

if HF_ADAPTERS_AVAILABLE:
    __all__.extend(
        [
            "TrajectoryAlignmentAdapter",
            "ViTAdapter",
            "ResNetAdapter",
            "CLIPViTAdapter",
            "LLaVAAdapter",
            "ImageBindAdapter",
            "create_hf_adapter",
        ]
    )
