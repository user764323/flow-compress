"""
Utility functions for flow-based distillation.
"""

import logging
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary


def set_seed(seed=42):
    """
    Set random seed for reproducibility.

    This function sets the random seed for all random operations (PyTorch, NumPy, Python).

    Args:
        seed (int): Random seed value. Default is 42.

    Returns:
        None
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def freeze_and_unfreeze_model(model, freeze=True):
    """Freeze and unfreeze all parameters in the model."""

    for param in model.parameters():
        param.requires_grad = not freeze


def get_attention_layers(model):
    """
    Retrieve all attention layers from the model.

    Args:
        model: Model to inspect.

    Returns:
        list: List of attention layers.
    """

    attention_layers = nn.ModuleList()

    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            attention_layers.append(module)

    return attention_layers


def model_summary(model, device="cuda", input_size=(1, 3, 224, 224)):
    """Generate a summary of the model."""

    # Generate summary
    cur_summary = summary(model, input_size, device=device, verbose=0)

    summary_dict = {
        "total_params": cur_summary.total_params,
        "trainable_params": cur_summary.trainable_params,
        "gflops": cur_summary.total_mult_adds / 1e9,  # Convert to GFLOPs
        "forward_backward_pass_size_MB": cur_summary.total_output_bytes / 1024**2,
        "params_size_MB": cur_summary.total_param_bytes / 1024**2,
        "estimated_total_memory_MB": (
            cur_summary.total_input
            + cur_summary.total_output_bytes
            + cur_summary.total_param_bytes
        )
        / 1024**2,
    }

    return summary_dict


def get_module(model, module_name):
    """
    Retrieve a submodule from a model.
    """

    parts = module_name.split(".")
    module = model

    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = module._modules.get(part)
            if module is None:
                raise ValueError(f"Module {part} not found in the model.")

    return module


def evaluate_topk(model, test_loader, device, k=5):
    """
    Evaluate the top-k accuracy of the model on the given test dataset.

    Args:
        model: Model to evaluate.
        test_loader: DataLoader for the test set.
        device: Device to use (CPU/GPU).
        k: Number of top predictions to consider. Default is 5.

    Returns:
        float: Top-k accuracy as a ratio between 0 and 1.
    """

    model.eval()

    correct = 0  # Counter for correct predictions
    total = 0  # Counter for total samples
    start_time = time.time()

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.topk(k, 1, True, True)
            # Count correct predictions (true label in top-k)
            correct += (predicted == y.view(-1, 1)).sum().item()
            total += y.size(0)

    elapsed_time = time.time() - start_time
    logging.info(
        f"[evaluate_topk] Top-{k} accuracy computed in {elapsed_time:.2f} seconds.")

    return correct / total


def count_attention_heads(model):
    """
    Count the total number of attention heads across all MultiheadAttention layers.

    Args:
        model: Model to inspect.

    Returns:
        int: Total number of attention heads in the model.
    """

    total_heads = 0

    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            total_heads += module.num_heads

    return total_heads
