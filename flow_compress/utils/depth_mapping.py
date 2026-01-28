"""
Depth mapping for flow-based distillation.
"""

from typing import Dict, List, Tuple


def normalize_depth_indices(num_layers: int) -> Dict[int, float]:
    """
    Returns normalized depth index for a network with num_layers layers.
    """

    if num_layers == 1:
        return {0: 0.0}

    return {i: i / (num_layers - 1) for i in range(num_layers)}


def bipartite_depth_mapping(
    teacher_layers: List[str],
    student_layers: List[str],
) -> Dict[str, str]:
    """
    Builds mapping of layers teacher→student by normalized depth.
    """

    T = len(teacher_layers)
    S = len(student_layers)

    t_depth = normalize_depth_indices(T)  # {i: s^T_i}
    s_depth = normalize_depth_indices(S)  # {j: s^S_j}

    mapping: Dict[str, str] = {}
    for j, s_name in enumerate(student_layers):
        s_pos = s_depth[j]
        # Find the nearest teacher layer by depth.
        best_i = min(range(T), key=lambda i: abs(t_depth[i] - s_pos))
        t_name = teacher_layers[best_i]
        mapping[t_name] = s_name

    return mapping


bipartite_depth_matching = bipartite_depth_mapping
