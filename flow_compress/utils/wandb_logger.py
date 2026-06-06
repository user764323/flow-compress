"""
Weights & Biases (wandb) integration for FAD training monitoring.
"""

import logging
from typing import Any, Dict, List, Optional

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("Warning: wandb not available. Install with: pip install wandb")


class WandBLogger:
    """
    Weights & Biases logger for FAD training.
    """

    def __init__(
        self,
        project: str = "fad-training",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        resume: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        if not WANDB_AVAILABLE:
            raise ImportError(
                "wandb is required. Install with: pip install wandb")

        self.project = project
        self.entity = entity
        self.name = name
        self.config = config or {}
        self.resume = resume
        self.tags = tags or []

        # Initialize wandb
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config,
            resume=resume,
            tags=tags,
        )

        logging.info(
            f"Initialized WandB logger: project={project}, name={name}")

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
    ) -> None:
        """
        Log metrics to W&B.
        """

        if step is not None:
            metrics["step"] = step
        wandb.log(metrics, commit=commit)

    def log_flow_divergence_trajectories(
        self,
        teacher_div: Dict[str, float],
        student_div: Dict[str, float],
        step: int,
    ) -> None:
        """
        Log per-layer flow divergence trajectories.
        """

        metrics = {}

        # Log individual layer divergences
        for layer_name, div in teacher_div.items():
            metrics[f"flow_divergence/teacher/{layer_name}"] = float(div)

        for layer_name, div in student_div.items():
            metrics[f"flow_divergence/student/{layer_name}"] = float(div)

        # Log alignment metrics
        if teacher_div and student_div:
            # Compute mean divergences
            mean_teacher_div = sum(teacher_div.values()) / len(teacher_div)
            mean_student_div = sum(student_div.values()) / len(student_div)

            metrics["flow_divergence/teacher/mean"] = float(mean_teacher_div)
            metrics["flow_divergence/student/mean"] = float(mean_student_div)

            # Compute alignment gap
            alignment_gap = abs(mean_teacher_div - mean_student_div)
            metrics["flow_divergence/alignment_gap"] = float(alignment_gap)

        self.log_metrics(metrics, step=step)

    def log_gradient_norms(
        self,
        gradient_norms: Dict[str, float],
        step: int,
    ) -> None:
        """
        Log gradient norms per layer.
        """

        metrics = {}
        for layer_name, norm in gradient_norms.items():
            metrics[f"gradient_norm/{layer_name}"] = float(norm)

        if gradient_norms:
            mean_norm = sum(gradient_norms.values()) / len(gradient_norms)
            max_norm = max(gradient_norms.values())
            min_norm = min(gradient_norms.values())

            metrics["gradient_norm/mean"] = float(mean_norm)
            metrics["gradient_norm/max"] = float(max_norm)
            metrics["gradient_norm/min"] = float(min_norm)

        self.log_metrics(metrics, step=step)

    def log_alignment_metrics(
        self,
        alignment_metrics: Dict[str, float],
        step: int,
    ) -> None:
        """
        Log alignment metrics.
        """

        metrics = {}
        for key, value in alignment_metrics.items():
            metrics[f"alignment/{key}"] = float(value)

        self.log_metrics(metrics, step=step)

    def log_training_metrics(
        self,
        loss: float,
        l_task: float,
        l_kd: float,
        l_fad: float,
        val_acc: Optional[float] = None,
        step: int = 0,
        epoch: Optional[int] = None,
    ) -> None:
        """
        Log standard training metrics.
        """

        metrics = {
            "loss/total": float(loss),
            "loss/task": float(l_task),
            "loss/kd": float(l_kd),
            "loss/fad": float(l_fad),
        }

        if val_acc is not None:
            metrics["metrics/val_acc"] = float(val_acc)

        if epoch is not None:
            metrics["epoch"] = epoch

        self.log_metrics(metrics, step=step)

    def log_curriculum_stage(
        self,
        stage: int,
        num_stages: int,
        gamma_k: float,
        step: int,
    ) -> None:
        """
        Log curriculum learning stage information.
        """

        metrics = {
            "curriculum/stage": stage,
            "curriculum/num_stages": num_stages,
            "curriculum/gamma_k": float(gamma_k),
            "curriculum/stage_progress": stage / num_stages,
        }
        self.log_metrics(metrics, step=step)

    def watch(
        self,
        model: Any,
        log: str = "gradients",
        log_freq: int = 100,
    ) -> None:
        """
        Watch model for logging gradients and parameters.
        """

        wandb.watch(model, log=log, log_freq=log_freq)

    def finish(self) -> None:
        """Finish the W&B run."""
        wandb.finish()
        logging.info("WandB run finished")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()


def create_wandb_logger(
    project: str = "fad-training",
    **kwargs,
) -> Optional[WandBLogger]:
    """
    Factory function to create WandB logger.
    """

    if not WANDB_AVAILABLE:
        logging.warning("wandb not available, skipping WandB logging")
        return None

    return WandBLogger(project=project, **kwargs)
