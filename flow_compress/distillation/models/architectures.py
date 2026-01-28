"""
Architecture-specific wrappers for different model types.
"""

from typing import Dict, List, Optional

from flow_compress.distillation.models.student_wrapper import StudentWrapper
from flow_compress.distillation.models.teacher_wrapper import TeacherWrapper
import torch
import torch.nn as nn


def get_vit_layers(model: nn.Module) -> List[str]:
    """
    Extract layer names from Vision Transformer (ViT).
    """

    layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.LayerNorm, nn.MultiheadAttention)):
            layer_names.append(name)
    return layer_names


def get_mobilenet_layers(model: nn.Module) -> List[str]:
    """
    Extract layer names from MobileNet.
    """

    layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
            layer_names.append(name)
    return layer_names


def get_bert_layers(model: nn.Module) -> List[str]:
    """
    Extract layer names from BERT.
    """

    layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.LayerNorm, nn.MultiheadAttention)):
            # Focus on transformer layers
            if "encoder.layer" in name or "pooler" in name:
                layer_names.append(name)
    return layer_names


def get_bilstm_layers(model: nn.Module) -> List[str]:
    """
    Extract layer names from BiLSTM.
    """

    layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.LSTM, nn.Linear, nn.Embedding)):
            layer_names.append(name)
    return layer_names


def create_vit_wrapper(
    model: nn.Module,
    is_teacher: bool = True,
) -> TeacherWrapper | StudentWrapper:
    """
    Create wrapper for Vision Transformer.
    """

    layer_names = get_vit_layers(model)
    if is_teacher:
        return TeacherWrapper(model, layer_names)
    else:
        return StudentWrapper(model, layer_names)


def create_mobilenet_wrapper(
    model: nn.Module,
    is_teacher: bool = True,
) -> TeacherWrapper | StudentWrapper:
    """
    Create wrapper for MobileNet.
    """

    layer_names = get_mobilenet_layers(model)
    if is_teacher:
        return TeacherWrapper(model, layer_names)
    else:
        return StudentWrapper(model, layer_names)


def create_bert_wrapper(
    model: nn.Module,
    is_teacher: bool = True,
) -> TeacherWrapper | StudentWrapper:
    """
    Create wrapper for BERT.
    """

    layer_names = get_bert_layers(model)
    if is_teacher:
        return TeacherWrapper(model, layer_names)
    else:
        return StudentWrapper(model, layer_names)


def create_bilstm_wrapper(
    model: nn.Module,
    is_teacher: bool = True,
) -> TeacherWrapper | StudentWrapper:
    """
    Create wrapper for BiLSTM.
    """

    layer_names = get_bilstm_layers(model)
    if is_teacher:
        return TeacherWrapper(model, layer_names)
    else:
        return StudentWrapper(model, layer_names)
