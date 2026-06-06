"""
Optimization-driven criteria for student architecture selection.
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple

from flow_compress.distillation.models.student_wrapper import StudentWrapper
from flow_compress.distillation.models.teacher_wrapper import TeacherWrapper
from flow_compress.utils.functional_geometry import FunctionalGeometryAnalyzer
from flow_compress.utils.representational_geometry import RepresentationalGeometryAnalyzer
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class StudentSelectionOptimizer:
    """
    Optimization-driven student architecture selection.
    """

    def __init__(
        self,
        teacher: TeacherWrapper,
        device: str = "cuda",
        alpha_functional: float = 0.4,
        alpha_representational: float = 0.4,
        alpha_performance: float = 0.2,
        num_analysis_samples: int = 1000,
    ):
        self.teacher = teacher
        self.device = device
        self.alpha_functional = alpha_functional
        self.alpha_representational = alpha_representational
        self.alpha_performance = alpha_performance
        self.num_analysis_samples = num_analysis_samples

        # Initialize analyzers
        self.functional_analyzer = FunctionalGeometryAnalyzer(
            self.teacher.backbone,
            list(self.teacher.layer_names),
            device=device,
        )

        self.representational_analyzer = RepresentationalGeometryAnalyzer(
            self.teacher.backbone,
            list(self.teacher.layer_names),
            device=device,
        )

        # Cache teacher analysis
        self._teacher_functional_analysis = None
        self._teacher_representational_analysis = None

    def analyze_teacher(
        self,
        data_loader: DataLoader,
    ):
        """
        Analyze teacher's functional and representational geometry.
        """

        logging.info("Analyzing teacher functional geometry...")
        self._teacher_functional_analysis = self.functional_analyzer.analyze(
            data_loader,
            num_samples=self.num_analysis_samples,
        )

        logging.info("Analyzing teacher representational geometry...")
        self._teacher_representational_analysis = (
            self.representational_analyzer.analyze(
                data_loader,
                num_samples=self.num_analysis_samples,
            )
        )

    def compute_functional_distance(
        self,
        student: StudentWrapper,
        data_loader: DataLoader,
        layer_mapping: Optional[Dict[str, str]] = None,
    ) -> float:
        """
        Computes functional geometry distance between teacher and student.
        """

        if self._teacher_functional_analysis is None:
            raise ValueError(
                "Teacher analysis not performed. Call analyze_teacher() first."
            )

        # Analyze student
        student_analyzer = FunctionalGeometryAnalyzer(
            student.backbone,
            list(student.layer_names),
            device=self.device,
        )
        student_analysis = student_analyzer.analyze(
            data_loader,
            num_samples=self.num_analysis_samples,
        )

        # Compute alignment
        alignment = self.functional_analyzer.compute_functional_alignment(
            self._teacher_functional_analysis["manifolds"],
            student_analysis["manifolds"],
            layer_mapping=layer_mapping,
        )

        # Distance = 1 - alignment
        distance = 1.0 - alignment

        return distance

    def compute_representational_distance(
        self,
        student: StudentWrapper,
        data_loader: DataLoader,
        layer_mapping: Optional[Dict[str, str]] = None,
    ) -> float:
        """
        Computes representational geometry distance between teacher and student.
        """

        if self._teacher_representational_analysis is None:
            raise ValueError(
                "Teacher analysis not performed. Call analyze_teacher() first."
            )

        # Analyze student
        student_analyzer = RepresentationalGeometryAnalyzer(
            student.backbone,
            list(student.layer_names),
            device=self.device,
        )
        student_analysis = student_analyzer.analyze(
            data_loader,
            num_samples=self.num_analysis_samples,
        )

        # Compute alignment
        alignment = self.representational_analyzer.compute_representational_alignment(
            self._teacher_representational_analysis["spaces"],
            student_analysis["spaces"],
            layer_mapping=layer_mapping,
        )

        # Distance = 1 - alignment
        distance = 1.0 - alignment

        return distance

    def compute_performance_gap(
        self,
        student: StudentWrapper,
        data_loader: DataLoader,
        task_metric_fn: Optional[Callable] = None,
    ) -> float:
        """
        Computes performance gap between teacher and student.
        """

        self.teacher.eval()
        student.eval()

        # Default metric: accuracy
        if task_metric_fn is None:

            def task_metric_fn(logits, labels):
                preds = logits.argmax(dim=1)
                return (preds == labels).float().mean()

        teacher_metrics = []
        student_metrics = []

        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0], batch[1]
                else:
                    x = batch
                    y = None

                x = x.to(self.device)
                if y is not None:
                    y = y.to(self.device)

                # Teacher
                t_logits = self.teacher(x)
                if y is not None:
                    t_metric = task_metric_fn(t_logits, y)
                    teacher_metrics.append(t_metric.item())

                # Student
                s_logits = student(x)
                if y is not None:
                    s_metric = task_metric_fn(s_logits, y)
                    student_metrics.append(s_metric.item())

        if not teacher_metrics or not student_metrics:
            return 1.0  # Maximum gap if no metrics

        avg_teacher_metric = np.mean(teacher_metrics)
        avg_student_metric = np.mean(student_metrics)

        # Normalized gap
        if avg_teacher_metric > 0:
            gap = abs(avg_teacher_metric - avg_student_metric) / \
                avg_teacher_metric
        else:
            gap = 1.0

        return gap

    def compute_objective(
        self,
        student: StudentWrapper,
        data_loader: DataLoader,
        layer_mapping: Optional[Dict[str, str]] = None,
        task_metric_fn: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """
        Computes the optimization objective for a student candidate.
        """

        # Compute distances
        d_functional = self.compute_functional_distance(
            student, data_loader, layer_mapping
        )
        d_representational = self.compute_representational_distance(
            student, data_loader, layer_mapping
        )
        d_performance = self.compute_performance_gap(
            student, data_loader, task_metric_fn
        )

        # Compute weighted objective
        objective = (
            self.alpha_functional * d_functional
            + self.alpha_representational * d_representational
            + self.alpha_performance * d_performance
        )

        return {
            "objective": objective,
            "functional_distance": d_functional,
            "representational_distance": d_representational,
            "performance_gap": d_performance,
        }

    def select_optimal_student(
        self,
        student_candidates: List[Tuple[StudentWrapper, Dict]],
        data_loader: DataLoader,
        task_metric_fn: Optional[Callable] = None,
        use_graph_matching: bool = True,
    ) -> Tuple[StudentWrapper, Dict]:
        """
        Selects the optimal student architecture using optimization criteria.
        """

        if self._teacher_functional_analysis is None:
            self.analyze_teacher(data_loader)

        best_objective = float("inf")
        best_student = None
        best_metrics = None

        for student, metadata in student_candidates:
            # Compute layer mapping
            if use_graph_matching:
                try:
                    from flow_compress.distillation.flows.graph_representation import ComputationalGraph
                    from flow_compress.distillation.flows.graph_matching import DynamicGraphMatcher

                    teacher_graph = ComputationalGraph(
                        self.teacher.backbone,
                        list(self.teacher.layer_names),
                    )
                    student_graph = ComputationalGraph(
                        student.backbone,
                        list(student.layer_names),
                    )

                    matcher = DynamicGraphMatcher()
                    # Get activations for matching
                    sample_batch = next(iter(data_loader))
                    x = (
                        sample_batch[0]
                        if isinstance(sample_batch, (list, tuple))
                        else sample_batch
                    )
                    x = x.to(self.device)

                    with torch.no_grad():
                        t_logits, _ = self.teacher.forward_with_flows(x)
                        s_logits, _ = student.forward_with_flows(x)
                        t_features = self.teacher.activation_hook.activations.copy()
                        s_features = student.activation_hook.activations.copy()

                    layer_mapping = matcher.compute_matching(
                        teacher_graph,
                        student_graph,
                        t_features,
                        s_features,
                    )
                except Exception:
                    # Fall back to depth-based mapping
                    from flow_compress.utils.depth_mapping import bipartite_depth_mapping

                    layer_mapping = bipartite_depth_mapping(
                        list(self.teacher.layer_names),
                        list(student.layer_names),
                    )
            else:
                from flow_compress.utils.depth_mapping import bipartite_depth_mapping

                layer_mapping = bipartite_depth_mapping(
                    list(self.teacher.layer_names),
                    list(student.layer_names),
                )

            # Compute objective
            metrics = self.compute_objective(
                student,
                data_loader,
                layer_mapping,
                task_metric_fn,
            )

            metrics["metadata"] = metadata
            metrics["layer_mapping"] = layer_mapping

            if metrics["objective"] < best_objective:
                best_objective = metrics["objective"]
                best_student = student
                best_metrics = metrics

        return best_student, best_metrics
