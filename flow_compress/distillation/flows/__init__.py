"""
Flow-based modules for knowledge distillation.
"""

from flow_compress.distillation.flows.alignment import (
    compute_dalign,
    compute_dalign_at_depth,
    compute_fad_loss_integrated,
    compute_fad_loss_with_weights,
    compute_layer_weights,
    depth_weighting,
)
from flow_compress.distillation.flows.divergence import compute_layerwise_flow_divergence, flow_divergence_pair
from flow_compress.distillation.flows.graph_matching import (
    DynamicGraphMatcher,
    GraphMatchingNetwork,
    compute_graph_matching,
)
from flow_compress.distillation.flows.graph_representation import ComputationalGraph, GraphNode
from flow_compress.distillation.flows.hooks import ActivationHook
from flow_compress.distillation.flows.selective import (
    compute_alignment_error,
    curriculum_mask,
    detect_critical_points,
    prioritized_alignment_mask,
    select_critical_layers,
    selective_flow_alignment,
)

__all__ = [
    "compute_dalign",
    "compute_dalign_at_depth",
    "compute_fad_loss_integrated",
    "compute_fad_loss_with_weights",
    "compute_layer_weights",
    "depth_weighting",
    "flow_divergence_pair",
    "compute_layerwise_flow_divergence",
    "ActivationHook",
    "ComputationalGraph",
    "GraphNode",
    "DynamicGraphMatcher",
    "GraphMatchingNetwork",
    "compute_graph_matching",
    "detect_critical_points",
    "compute_alignment_error",
    "prioritized_alignment_mask",
    "selective_flow_alignment",
    "select_critical_layers",
    "curriculum_mask",
]
