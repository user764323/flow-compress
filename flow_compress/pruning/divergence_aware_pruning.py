"""IDAP++ (Iterative Dataset-Aware Pruning) Implementation.

This module implements the IDAP++ algorithm for neural network pruning, which:
1. Iteratively removes less important neurons while maintaining model performance
2. Uses dataset-aware importance metrics for pruning decisions
3. Applies knowledge distillation to recover performance after pruning
4. Monitors and logs pruning progress and model statistics
"""

import os
import copy
import logging
import torch
from torch import nn
import torch_pruning as tp
from tqdm import tqdm

from flow_compress.pruning.distillation import distill
from flow_compress.utils.pruning_utils import (
    evaluate_topk,
    get_module_by_name,
    is_prunable_layer,
    get_module_by_name,
    get_model_summary,
)


def compute_neurons_importance_from_loader(
    model, data_loader, prunable_layers=None, max_batches=10, device="cuda", writer=None, iteration=0
):
    """Compute neuron importance from data loader.
    Args:
        model: The neural network model to analyze
        data_loader: DataLoader providing input samples
        prunable_layers: List of (name, module) tuples for layers that can be pruned
        max_batches: Maximum number of batches to process for importance calculation
        device: Device to run computations on ('cuda' or 'cpu')
        writer: TensorBoard writer for logging importance scores
        iteration: Current pruning iteration for logging
        
    Returns:
        list: List of (importance_score, layer_name, neuron_index) tuples
    """

    model.eval()

    # Initialize accumulator for importance scores
    importance_accumulator = {name: [] for name, _ in prunable_layers}

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(data_loader):
            if batch_idx >= max_batches:
                break

            x = x.to(device)

            hooks = []

            def make_hook(name, module):
                def save_divergence(module, input, output):
                    output = output.detach()
                    output_activated = torch.relu(output)
                    weights = module.weight.detach()

                    if len(output_activated.shape) == 4:
                        b, c, h, w = output_activated.shape

                        # Calculate activation norm across spatial dimensions
                        div = output_activated.view(b, c, -1)
                        div = torch.norm(div, dim=2).mean(dim=0)

                        # Calculate weight norm
                        weights = weights.view(c, -1)
                        weights = torch.norm(weights, dim=1)

                        normalization_coef = 1 / (c * h * w)
                    else:
                        b, n = output_activated.shape

                        # Calculate activation norm
                        div = output_activated.view(b, n)
                        div = torch.norm(div, dim=0).mean(dim=0)

                        # Calculate weight norm
                        weights = torch.norm(weights, dim=1)
                        normalization_coef = 1 / n

                    # Combine activation and weight importance
                    div = normalization_coef * div * weights

                    importance_accumulator[name].append(div.cpu())

                return save_divergence

            # Register hooks for all prunable layers
            for name, module in prunable_layers:
                hooks.append(module.register_forward_hook(make_hook(name, module)))

            # Forward pass to compute importance scores
            _ = model(x)

            # Remove hooks after computation
            for h in hooks:
                h.remove()

    # Compute average importance scores across batches
    result = []
    for name, list_of_divs in importance_accumulator.items():
        if list_of_divs:
            stacked = torch.stack(list_of_divs)
            avg = stacked.mean(dim=0)

            # Log neuron importance to TensorBoard
            if writer is not None:
                writer.add_histogram(f'Neuron_Importance/{name}', avg, iteration)

            # Store importance scores with layer and neuron information
            for i, score in enumerate(avg):
                result.append((score.item(), name, i))

    return result


def compute_layer_importance(importance_scores, aggregation_method="mean"):
    """
    Compute layer importance based on neuron divergence scores.

    Args:
        importance_scores (list): List of (importance_score, layer_name, neuron_index) tuples.
        aggregation_method (str): Method to aggregate neuron scores for a layer ('mean', 'sum', 'max').

    Returns:
        list: List of (layer_importance_score, layer_name) tuples.
    """
    from collections import defaultdict
    import numpy as np

    # Group neuron importance scores by layer
    layer_scores = defaultdict(list)
    for score, layer_name, _ in importance_scores:
        layer_scores[layer_name].append(score)

    # Compute layer importance based on the aggregation method
    layer_importance = []
    for layer_name, scores in layer_scores.items():
        if aggregation_method == "mean":
            importance = np.mean(scores)
        elif aggregation_method == "sum":
            importance = np.sum(scores)
        elif aggregation_method == "max":
            importance = np.max(scores)
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation_method}")
        layer_importance.append((importance, layer_name))

    # Sort layers by importance in descending order
    layer_importance.sort(key=lambda x: x[0], reverse=True)

    return layer_importance


def divergence_aware_pruning(
    model_path,
    pretrained_model,
    test_loader,
    train_loader,
    num_classes=10,
    performance_metric_function=evaluate_topk,
    max_performance_metric_degradation_th=0.1,
    number_of_pruning_iterations=10,
    base_pruning_percentage=0.05,
    pruning_alpha=1.5,
    device="cuda",
    save_checkpoints=True,
    writer=None,
):
    """Implement the IDAP++ (Iterative Dataset-Aware Pruning) algorithm.
    
    This function performs iterative pruning of a neural network while:
    1. Maintaining model performance within specified degradation threshold
    2. Using dataset-aware importance metrics for pruning decisions
    3. Applying knowledge distillation to recover performance
    4. Monitoring and logging pruning progress
    
    Args:
        model_path: Path to save the pruned model
        pretrained_model: Initial model to prune
        test_loader: DataLoader for evaluation
        train_loader: DataLoader for training/distillation
        num_classes: Number of output classes
        performance_metric_function: Function to evaluate model performance
        max_performance_metric_degradation_th: Maximum allowed performance degradation
        number_of_pruning_iterations: Number of pruning iterations to perform
        base_pruning_percentage: Base percentage of neurons to prune per iteration
        pruning_alpha: Scaling factor for pruning percentage
        device: Device to run computations on
        save_checkpoints: Whether to save model checkpoints
        writer: TensorBoard writer for logging
        
    Returns:
        The best performing pruned model
    """

    # Ensure output directory exists
    dir_name = os.path.dirname(model_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    # Prepare dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # Pretrained model to device
    pretrained_model = pretrained_model.to(device)
    pretrained_model.eval()

    # Evaluate initial model performance
    initial_accuracy_top1 = performance_metric_function(
        pretrained_model, test_loader, device, k=1
    )
    initial_accuracy_top5 = performance_metric_function(
        pretrained_model, test_loader, device, k=5
    )

    # Log initial performance metrics
    if writer is not None:
        writer.add_scalar('Accuracy/Initial_Top1', initial_accuracy_top1, 0)
        writer.add_scalar('Accuracy/Initial_Top5', initial_accuracy_top5, 0)

    logging.debug("Initial TOP-1 accuracy: %s%%", initial_accuracy_top1)
    logging.debug("Initial TOP-5 accuracy: %s%%", initial_accuracy_top5)

    # Get initial model statistics
    initial_model_summary = get_model_summary(pretrained_model, device=device)
    logging.debug("Initial model summary: %s", initial_model_summary)

    # Initialize teacher model for knowledge distillation
    teacher_model = copy.deepcopy(pretrained_model)

    # Initialize tracking of best model
    best_model = copy.deepcopy(pretrained_model)
    best_accuracy = initial_accuracy_top1

    # Main pruning loop
    for t in range(number_of_pruning_iterations):
        # Log current model statistics
        model_summary = get_model_summary(pretrained_model, device=device)
        logging.debug("Model summary, step %s: %s", model_summary, t)

        if writer is not None:
            writer.add_scalar('Model/Parameters', model_summary['total_params'], t)
            writer.add_scalar('Model/FLOPs', model_summary['gflops'], t)

        # Identify prunable layers
        prunable_layers = [
            (name, m)
            for name, m in pretrained_model.named_modules()
            if is_prunable_layer(m, num_classes)
        ]
        if not prunable_layers:
            logging.info("No prunable layers found. Stopping pruning.")
            break

        # Compute importance scores for neurons
        neurons_importance = compute_neurons_importance_from_loader(
            pretrained_model,
            test_loader,
            prunable_layers,
            max_batches=100,
            device=device,
            writer=writer,
            iteration=t,
        )
        neurons_importance.sort(key=lambda x: x[0])

        # Calculate pruning percentage for this iteration
        pruning_percentage = (
            base_pruning_percentage
            * (1 + t / number_of_pruning_iterations) ** pruning_alpha
        )
        num_prune = int(pruning_percentage * len(neurons_importance))
        
        # Log pruning statistics
        if writer is not None:
            writer.add_scalar('Pruning/Percentage', pruning_percentage * 100, t)
            writer.add_scalar('Pruning/Neurons_Removed', num_prune, t)

        logging.debug(
            "Pruning %s neurons; total prunable elements: %s",
            num_prune,
            len(neurons_importance),
        )

        if num_prune == 0:
            logging.debug("No more filters to prune.")
            break

        # Select neurons to prune based on importance scores
        prune_targets = neurons_importance[:num_prune]

        prune_dict = {}
        # Group neurons by layer
        for _, name, neuron_idx in tqdm(prune_targets):
            if name not in prune_dict:
                prune_dict[name] = [neuron_idx]
            else:
                prune_dict[name].append(neuron_idx)

        # Prune the model
        for name, neuron_idxs in prune_dict.items():
            # Build dependency graph for the model
            dep_graph = tp.DependencyGraph().build_dependency(
                pretrained_model,
                example_inputs=dummy_input
            )
            module = get_module_by_name(pretrained_model, name)

            # Create pruning group based on layer type
            if isinstance(module, nn.Conv2d):
                group = dep_graph.get_pruning_group(
                    module,
                    tp.prune_conv_out_channels,
                    idxs=neuron_idxs
                )
            elif isinstance(module, nn.Linear):
                group = dep_graph.get_pruning_group(
                    module,
                    tp.prune_linear_out_channels,
                    idxs=neuron_idxs
                )
            elif isinstance(module, nn.MultiheadAttention):
                # For MultiheadAttention, we prune the output projection
                group = dep_graph.get_pruning_group(
                    module,
                    tp.prune_multihead_attention_out_channels,
                    idxs=neuron_idxs
                )
            else:
                logging.warning("Layer %s is not Conv2d, Linear, or MultiheadAttention, skipping pruning.", name)
                continue

            # Execute pruning if valid
            if dep_graph.check_pruning_group(group):
                group.prune()
                torch.cuda.empty_cache()

        # Apply knowledge distillation to recover performance
        distill(
            pretrained_model,
            teacher_model,
            dummy_input,
            train_loader,
            test_loader,
            performance_metric_function,
            initial_accuracy_top1,
            max_performance_metric_degradation_th,
            device,
            writer=writer,
        )
        teacher_model = copy.deepcopy(pretrained_model)

        # Evaluate current model performance
        current_accuracy_top1 = performance_metric_function(
            pretrained_model, test_loader, device, k=1
        )
        current_accuracy_top5 = performance_metric_function(
            pretrained_model, test_loader, device, k=5
        )

        # Log current performance metrics
        if writer is not None:
            writer.add_scalar('Accuracy/Current_Top1', current_accuracy_top1, t)
            writer.add_scalar('Accuracy/Current_Top5', current_accuracy_top5, t)
            writer.add_scalar('Accuracy/Degradation', initial_accuracy_top1 - current_accuracy_top1, t)

        logging.debug("Current Top-1 Accuracy: %s%%", current_accuracy_top1)
        logging.debug("Current Top-5 Accuracy: %s%%", current_accuracy_top5)

        # Check if performance degradation exceeds threshold
        if (
            initial_accuracy_top1 - current_accuracy_top1
            > max_performance_metric_degradation_th
        ):
            logging.info("Accuracy degradation threshold exceeded. Stopping pruning.")
            break

        # Update best model if current performance is better
        if current_accuracy_top1 > best_accuracy:
            best_accuracy = current_accuracy_top1
            best_model = copy.deepcopy(pretrained_model)
            if writer is not None:
                writer.add_scalar('Accuracy/Best', best_accuracy, t)

    best_model = idap_plus_plus(
        pruned_network=best_model, \
        train_loader=train_loader, \
        validation_loader=test_loader, \
        target_error_reduction_ratio=0.05,  # Example value, adjust as needed
        max_layer_removal_budget=5,  # Example value, adjust as needed
        metric_function=performance_metric_function,
        fine_tune_function=distill,
        final_fine_tune_function=distill,
        device=device,
        teacher_model=teacher_model,
        writer=writer,
        t=t,
        initial_accuracy_top1=initial_accuracy_top1,
        max_performance_metric_degradation_th=max_performance_metric_degradation_th
    )

    return best_model


def sort_layers_by_flow(pretrained_model, test_loader, prunable_layers, t, device="cuda", writer=None):
    """
    Sort layers by their information flow (divergence) in descending order.

    Returns:
        List of layers sorted by their flow in descending order.
    """

    # Compute importance scores for neurons
    neurons_importance = compute_neurons_importance_from_loader(
        pretrained_model,
        test_loader,
        prunable_layers,
        max_batches=100,
        device=device,
        writer=writer,
        iteration=t,
    )

    # Compute layer importance from neuron importance scores
    layer_importance = compute_layer_importance(neurons_importance, aggregation_method="mean")

    # Sort layers by importance in descending order
    sorted_layers = sorted(layer_importance, key=lambda x: x[0], reverse=True)
    sorted_layer_names = [name for _, name in sorted_layers]
    logging.debug("Layers sorted by flow: %s", sorted_layer_names)

    return sorted_layers


def idap_plus_plus(
    pruned_network,
    train_loader,
    validation_loader,
    target_error_reduction_ratio,
    max_layer_removal_budget,
    metric_function,
    fine_tune_function,
    final_fine_tune_function,
    device,
    teacher_model,
    writer,
    t,
    initial_accuracy_top1,
    max_performance_metric_degradation_th
):
    """
    Layer Removal via Information Flow Optimization (IDAP++).

    Returns:
        Optimally compressed network and the set of removed layers.
    """

    # Prepare dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # Initialize removal candidate set by sorting layers by flow
    candidate_layers = sort_layers_by_flow(pruned_network, validation_loader, pruned_network.named_modules(), 0, device=device)
    removed_layers = set()

    # Initialize error reduction tracker and removal counter
    delta_error = 0
    removal_count = 0

    # Evaluate initial error
    pruned_network = pruned_network.to(device)
    pruned_network.eval()
    initial_error = metric_function(pruned_network, validation_loader, device)

    while removal_count < max_layer_removal_budget and delta_error < target_error_reduction_ratio:
        # Select layer with minimal flow
        layer_to_remove = candidate_layers.pop(0)

        # Build dependency graph for the model
        dep_graph = tp.DependencyGraph().build_dependency(
            pruned_network,
            example_inputs=dummy_input
        )
        group = dep_graph.get_pruning_group(
            layer_to_remove,
            tp.prune_conv_out_channels
        )

        if dep_graph.check_pruning_group(group):
            group.prune()
            torch.cuda.empty_cache()

        # Apply knowledge distillation to recover performance
        fine_tune_function(
            pruned_network,
            teacher_model,
            dummy_input,
            train_loader,
            validation_loader,
            metric_function,
            initial_accuracy_top1,
            max_performance_metric_degradation_th,
            device,
            writer=writer,
        )
        teacher_model = copy.deepcopy(pruned_network)

        # Evaluate current model performance
        current_accuracy_top1 = metric_function(
            pruned_network, validation_loader, device, k=1
        )
        current_accuracy_top5 = metric_function(
            pruned_network, validation_loader, device, k=5
        )

        # Log current performance metrics
        if writer is not None:
            writer.add_scalar('Accuracy/Current_Top1', current_accuracy_top1, t)
            writer.add_scalar('Accuracy/Current_Top5', current_accuracy_top5, t)
            writer.add_scalar('Accuracy/Degradation', initial_accuracy_top1 - current_accuracy_top1, t)

        logging.debug("Current Top-1 Accuracy: %s%%", current_accuracy_top1)
        logging.debug("Current Top-5 Accuracy: %s%%", current_accuracy_top5)

        # Check if performance degradation exceeds threshold
        if (
            initial_accuracy_top1 - current_accuracy_top1
            > max_performance_metric_degradation_th
        ):
            logging.info("Accuracy degradation threshold exceeded. Stopping pruning.")
            break

        # Update best model if current performance is better
        if current_accuracy_top1 > best_accuracy:
            best_accuracy = current_accuracy_top1
            best_model = copy.deepcopy(pruned_network)
            if writer is not None:
                writer.add_scalar('Accuracy/Best', best_accuracy, t)

        # Evaluate Impact
        temp_error = metric_function(pruned_network, validation_loader, device)
        error_reduction = initial_error - temp_error

        if error_reduction > 0:
            # Accept removal
            pruned_network = pruned_network
            removed_layers.add(layer_to_remove)
            delta_error += error_reduction
            removal_count += 1
        else:
            # Mark layer as essential
            continue

    # Finalize Network
    final_network = final_fine_tune_function(pruned_network, validation_loader, device)

    return final_network, removed_layers
