"""
FAD Distillation Implementation.
"""

import os
import copy
import logging
import torch
from torch import nn
from tqdm import tqdm

from utils import (
    evaluate_topk,
    get_module_by_name,
    get_module_by_name,
    get_model_summary,
)

from collections import defaultdict
import numpy as np


def make_a_hook(name, module, importance_accumulator):
    """
    Make a hook to save the divergence of the module.
    """

    def save_divergence_of_the_module(module, importance_accumulator, input, output):
        """
        Save the divergence of the module.
        """

        output = output.detach()
        output_A = torch.relu(output)
        weights = module.weight.detach()

        if len(output_A.shape) == 4:
            b, c, h, w = output_A.shape

            # Calculate activation norm
            diverg = output_A.view(b, c, -1)
            diverg = torch.norm(diverg, dim=2).mean(dim=0)

            # Calculate weight norm
            weights = weights.view(c, -1)
            weights = torch.norm(weights, dim=1)

            norm_coef = 1 / (c * h * w)

        else:
            b, n = output_A.shape

            # Calculate activation norm
            diverg = output_A.view(b, n)
            diverg = torch.norm(diverg, dim=0).mean(dim=0)

            # Calculate weight norm
            weights = torch.norm(weights, dim=1)
            norm_coef = 1 / n

        # Combine activation and weight importance
        diverg = (norm_coef * diverg * weights).cpu()

        importance_accumulator[name].append(diverg)

    return save_divergence_of_the_module


@torch.no_grad()
def calc_neurons_importance(
    current_model, dataloader, distillation_layers=None, max_batches=10, device="cuda:0", writer=None, iteration=0
):
    """
    Calculate neurons importance from data loader.
    """

    current_model.eval()

    # Initialize accumulator for importance scores
    importance_accumulator = {name: [] for name, _ in distillation_layers}

    for batch_index, (x, _) in enumerate(dataloader):
        if batch_index >= max_batches:
            break

        x = x.to(device)

        hooks = []

        # Register hooks for all distillation layers
        for name, module in distillation_layers:
            hooks.append(module.register_forward_hook(make_a_hook(name, module, importance_accumulator)))

        # Forward pass to calculate importance scores
        _ = current_model(x)

        # Remove hooks after computation
        for h in hooks:
            h.remove()

    # Calculate average importance scores across batches
    result = []
    for name, list_of_divergences in importance_accumulator.items():
        if list_of_divergences:
            stacked_divergences = torch.stack(list_of_divergences)
            average_divergences = stacked_divergences.mean(dim=0)

            # Log neuron importance to TensorBoard
            if writer is not None:
                writer.add_histogram(f'Neuron_Importance/{name}', average_divergences, iteration)

            # Store importance scores with layer and neuron information
            for i, score in enumerate(average_divergences):
                result.append((score.item(), name, i))

    return result


def calc_layer_importance(importance_scores, aggregation_method="mean"):
    """
    Calculate layer importance based on neuron divergence scores.
    """

    # Group neuron importance scores by layer
    layer_scores = defaultdict(list)
    for score, layer_name, _ in importance_scores:
        layer_scores[layer_name].append(score)

    # Calculate layer importance based on the aggregation method
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


def divergence_aware_distillation(
    model_path,
    pretrained_model,
    test_loader,
    train_loader,
    num_classes=10,
    performance_metric_function=evaluate_topk,
    max_performance_metric_degradation_th=0.1,
    number_of_iterations=10,
    base_percentage=0.05,
    alpha=1.5,
    device="cuda",
    save_checkpoints=True,
    writer=None,
):
    """
    divergence_aware_distillation
    """

    # Ensure output directory exists
    directory_name = os.path.dirname(model_path)
    if directory_name and not os.path.exists(directory_name):
        os.makedirs(directory_name, exist_ok=True)

    # Pretrained model to device
    pretrained_model = pretrained_model.to(device)
    pretrained_model.eval()

    # Prepare dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # Evaluate initial model performance
    initial_accuracy1 = performance_metric_function(
        pretrained_model, test_loader, device, k=1
    )
    initial_accuracy5 = performance_metric_function(
        pretrained_model, test_loader, device, k=5
    )

    # Log initial performance metrics
    if writer is not None:
        writer.add_scalar('Acc/Initial_Top1', initial_accuracy1, 0)
        writer.add_scalar('Acc/Initial_Top5', initial_accuracy5, 0)

    logging.debug("Initial Acc@1: %s%%", initial_accuracy1)
    logging.debug("Initial Acc@5: %s%%", initial_accuracy5)

    # Get initial model statistics
    initial_model_summary = current_model_summary(pretrained_model, device=device)
    logging.debug("Initial model summary: %s", initial_model_summary)

    # Initialize teacher model for knowledge distillation
    teacher_model = copy.deepcopy(pretrained_model)

    # Initialize tracking of best model
    best_model = copy.deepcopy(pretrained_model)
    best_accuracy = initial_accuracy1

    # Main loop
    for t in range(number_of_iterations):
        # Log current model statistics
        current_model_summary = model_summary(pretrained_model, device=device)
        logging.debug("Model summary %s: %s", current_model_summary, t)

        if writer is not None:
            writer.add_scalar('Model/Parameters', current_model_summary['total_params'], t)
            writer.add_scalar('Model/FLOPs', current_model_summary['gflops'], t)

        # Identify layers
        distillation_layers = [
            (name, m)
            for name, m in pretrained_model.named_modules()
            if is_distillation_layer(m, num_classes)
        ]
        if not distillation_layers:
            logging.info("No distillation layers found. Stopping distillation.")
            break

        # Compute importance scores for neurons
        neurons_importance = calc_neurons_importance(
            pretrained_model,
            test_loader,
            distillation_layers,
            max_batches=100,
            device=device,
            writer=writer,
            iteration=t,
        )
        neurons_importance.sort(key=lambda x: x[0])

        # Calculate distillation percentage for this iteration
        distillation_percentage = (
            base_percentage
            * (1 + t / number_of_iterations) ** alpha
        )
        num_distillation = int(distillation_percentage * len(neurons_importance))
        
        # Log distillation statistics
        if writer is not None:
            writer.add_scalar('Distillation/Percentage', distillation_percentage * 100, t)
            writer.add_scalar('Distillation/Neurons_Removed', num_distillation, t)

        logging.debug(
            "Distillation %s neurons; total distillation elements: %s",
            num_distillation,
            len(neurons_importance),
        )

        if num_distillation == 0:
            logging.debug("No more filters to distillation.")
            break

        # Select neurons to distillation based on importance scores
        distillation_targets = neurons_importance[:num_distillation]

        distillation_dict = {}
        # Group neurons by layer
        for _, name, neuron_idx in tqdm(distillation_targets):
            if name not in distillation_dict:
                distillation_dict[name] = [neuron_idx]
            else:
                distillation_dict[name].append(neuron_idx)

        # Distillation the model
        for name, neuron_idxs in distillation_dict.items():
            # Build dependency graph for the model
            dep_graph = tp.DependencyGraph().build_dependency(
                pretrained_model,
                example_inputs=dummy_input
            )
            module = get_module_by_name(pretrained_model, name)

            # Create distillation group based on layer type
            if isinstance(module, nn.Conv2d):
                group = dep_graph.get_distillation_group(
                    module,
                    tp.distill_conv_out_channels,
                    idxs=neuron_idxs
                )
            elif isinstance(module, nn.Linear):
                group = dep_graph.get_distillation_group(
                    module,
                    tp.distill_linear_out_channels,
                    idxs=neuron_idxs
                )
            elif isinstance(module, nn.MultiheadAttention):
                # For MultiheadAttention, we distillation the output projection
                group = dep_graph.get_distillation_group(
                    module,
                    tp.distill_multihead_attention_out_channels,
                    idxs=neuron_idxs
                )
            else:
                logging.warning("Layer %s is not Conv2d, Linear, or MultiheadAttention, skipping distillation.", name)
                continue

            # Execute distillation if valid
            if dep_graph.check_distillation_group(group):
                group.distill()
                torch.cuda.empty_cache()

        # Apply knowledge distillation to recover performance
        distill(
            pretrained_model,
            teacher_model,
            dummy_input,
            train_loader,
            test_loader,
            performance_metric_function,
            initial_accuracy1,
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
            writer.add_scalar('Accuracy/Degradation', initial_accuracy1 - current_accuracy_top1, t)

        logging.debug("Current Top-1 Accuracy: %s%%", current_accuracy_top1)
        logging.debug("Current Top-5 Accuracy: %s%%", current_accuracy_top5)

        # Check if performance degradation exceeds threshold
        if (
            initial_accuracy1 - current_accuracy_top1
            > max_performance_metric_degradation_th
        ):
            logging.info("Accuracy degradation threshold exceeded. Stopping distillation.")
            break

        # Update best model if current performance is better
        if current_accuracy_top1 > best_accuracy:
            best_accuracy = current_accuracy_top1
            best_model = copy.deepcopy(pretrained_model)
            if writer is not None:
                writer.add_scalar('Accuracy/Best', best_accuracy, t)

    best_model = fad_distillation(
        network=best_model,
        train_loader=train_loader,
        validation_loader=test_loader,
        target_error_reduction_ratio=0.05,  # Example value, adjust as needed
        max_layer_removal_budget=5,  # Example value, adjust as needed
        metric_function=performance_metric_function,
        fine_tune_function=distill,
        final_fine_tune_function=distill,
        device=device,
        teacher_model=teacher_model,
        writer=writer,
        t=t,
        initial_accuracy_top1=initial_accuracy1,
        max_performance_metric_degradation_th=max_performance_metric_degradation_th
    )

    return best_model


def sort_layers_by_flow(pretrained_model, test_loader, distillation_layers, t, device="cuda", writer=None):
    """
    Sort layers by their information flow (divergence) in descending order based on neurons importance.
    """

    # Calculate importance scores for neurons
    neurons_importance = calc_neurons_importance(
        pretrained_model,
        test_loader,
        distillation_layers,
        max_batches=100,
        device=device,
        writer=writer,
        iteration=t,
    )

    # Calculate layer importance from neuron importance scores
    layer_importance = calc_layer_importance(neurons_importance, aggregation_method="mean")

    # Sort layers by importance in descending order
    sorted_layers = sorted(layer_importance, key=lambda x: x[0], reverse=True)
    sorted_layer_names = [name for _, name in sorted_layers]
    logging.debug("Layers sorted by flow: %s", sorted_layer_names)

    return sorted_layers


def fad_distillation(
    network,
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
    Layer Removal via Information Flow Optimization.
    """

    # Prepare dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # Initialize removal candidate set by sorting layers by flow
    candidate_layers = sort_layers_by_flow(network, validation_loader, network.named_modules(), 0, device=device)
    removed_layers = set()

    # Initialize error reduction tracker and removal counter
    delta_error = 0
    removal_count = 0

    # Evaluate initial error
    network = network.to(device)
    network.eval()
    initial_error = metric_function(network, validation_loader, device)

    while removal_count < max_layer_removal_budget and delta_error < target_error_reduction_ratio:
        # Select layer with minimal flow
        layer_to_remove = candidate_layers.pop(0)

        # Build dependency graph for the model
        dep_graph = tp.DependencyGraph().build_dependency(
            network,
            example_inputs=dummy_input
        )
        group = dep_graph.get_distillation_group(
            layer_to_remove,
            tp.distill_conv_out_channels
        )

        if dep_graph.check_distillation_group(group):
            group.distill()
            torch.cuda.empty_cache()

        # Apply knowledge distillation to recover performance
        fine_tune_function(
            network,
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
        teacher_model = copy.deepcopy(network)

        # Evaluate current model performance
        current_accuracy_top1 = metric_function(
            network, validation_loader, device, k=1
        )
        current_accuracy_top5 = metric_function(
            network, validation_loader, device, k=5
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
            logging.info("Accuracy degradation threshold exceeded. Stopping distillation.")
            break

        # Update best model if current performance is better
        if current_accuracy_top1 > best_accuracy:
            best_accuracy = current_accuracy_top1
            best_model = copy.deepcopy(network)
            if writer is not None:
                writer.add_scalar('Accuracy/Best', best_accuracy, t)

        # Evaluate Impact
        temp_error = metric_function(network, validation_loader, device)
        error_reduction = initial_error - temp_error

        if error_reduction > 0:
            # Accept removal
            network = network
            removed_layers.add(layer_to_remove)
            delta_error += error_reduction
            removal_count += 1
        else:
            # Mark layer as essential
            continue

    return network, removed_layers
