from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from flow_compress.distillation.models.student_wrapper import StudentWrapper
from flow_compress.distillation.models.teacher_wrapper import TeacherWrapper
from flow_compress.distillation.trainer.fad_trainer import FADTrainer

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Results from a single distillation experiment."""

    task: str
    dataset: str
    teacher_model: str
    student_model: str
    method: str
    metric_name: str
    metric_value: float
    baseline_metric: Optional[float] = None
    teacher_size_mb: Optional[float] = None
    student_size_mb: Optional[float] = None
    compression_ratio: Optional[float] = None
    epochs: int = 0
    config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""

        return asdict(self)


class BaseExperiment(ABC):
    """Base class for distillation experiments."""

    def __init__(
        self,
        task: str,
        dataset: str,
        teacher_model_name: str,
        student_model_name: str,
        device: str = "cuda",
        data_root: str = "./data",
        output_dir: str = "./experiments/results",
        log_dir: Optional[str] = None,
    ):
        self.task = task
        self.dataset = dataset
        self.teacher_model_name = teacher_model_name
        self.student_model_name = student_model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if log_dir is None:
            log_dir = (
                self.output_dir
                / "tensorboard"
                / f"{task}_{dataset}_{teacher_model_name}_to_{student_model_name}"
            )
        else:
            log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(log_dir))

    @abstractmethod
    def load_teacher_model(self) -> nn.Module:
        """Load the teacher model architecture."""
        pass

    @abstractmethod
    def load_student_model(self) -> nn.Module:
        """Load the student model architecture."""
        pass

    @abstractmethod
    def load_dataset(self) -> Tuple[Any, Any]:
        """Load train and validation datasets."""
        pass

    @abstractmethod
    def get_metric_name(self) -> str:
        """Return the name of the evaluation metric."""
        pass

    def evaluate(self, model: nn.Module, val_loader: Any) -> float:
        """Evaluate model accuracy on validation set."""

        model.eval()
        model.to(self.device)
        total = 0
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits = model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        return correct / max(1, total)

    def get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB."""

        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        total_size = param_size + buffer_size
        return total_size / (1024 * 1024)

    def _infer_layer_names(self, model: nn.Module) -> list[str]:
        """Infer layer names for flow alignment."""

        names = [name for name, _ in model.named_modules() if "layer" in name and "bn" not in name]
        if names:
            return names
        return [name for name, _ in model.named_modules() if name]

    def run_fad_experiment(
        self,
        num_epochs: int = 1,
        temperature: float = 4.0,
        lambda_weight: float = 0.1,
        use_weights: bool = True,
        precompute_teacher_divs: bool = True,
    ) -> ExperimentResult:
        """Run experiment with FAD distillation."""

        logger.info(f"\n{'='*80}")
        logger.info(
            f"Running FAD experiment: {self.dataset} | {self.teacher_model_name} -> {self.student_model_name}"
        )
        logger.info(f"{'='*80}")

        teacher = self.load_teacher_model()
        student = self.load_student_model()
        train_loader, val_loader = self.load_dataset()

        logger.info("Evaluating teacher baseline...")
        baseline_metric = self.evaluate(teacher, val_loader)

        teacher_wrapper = TeacherWrapper(teacher, layer_names=self._infer_layer_names(teacher))
        student_wrapper = StudentWrapper(student, layer_names=self._infer_layer_names(student))

        optimizer = torch.optim.AdamW(student_wrapper.parameters(), lr=1e-3, weight_decay=1e-4)
        trainer = FADTrainer(
            teacher=teacher_wrapper,
            student=student_wrapper,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=str(self.device),
            temperature=temperature,
            lambda_weight=lambda_weight,
            use_weights=use_weights,
            precompute_teacher_divs=precompute_teacher_divs,
        )

        for epoch in range(num_epochs):
            trainer.train_epoch(epoch)
            val_metrics = trainer.evaluate()
            self.writer.add_scalar("val_acc", val_metrics["val_acc"], epoch)

        student_metric = val_metrics["val_acc"]
        teacher_size = self.get_model_size(teacher)
        student_size = self.get_model_size(student)
        compression_ratio = teacher_size / max(1e-8, student_size)

        result = ExperimentResult(
            task=self.task,
            dataset=self.dataset,
            teacher_model=self.teacher_model_name,
            student_model=self.student_model_name,
            method="fad",
            metric_name=self.get_metric_name(),
            metric_value=student_metric,
            baseline_metric=baseline_metric,
            teacher_size_mb=teacher_size,
            student_size_mb=student_size,
            compression_ratio=compression_ratio,
            epochs=num_epochs,
            config={
                "temperature": temperature,
                "lambda_weight": lambda_weight,
                "use_weights": use_weights,
                "precompute_teacher_divs": precompute_teacher_divs,
            },
        )

        logger.info(f"Baseline {self.get_metric_name()}: {baseline_metric:.4f}")
        logger.info(f"Student {self.get_metric_name()}: {student_metric:.4f}")
        logger.info(f"Compression: {compression_ratio:.2f}x")

        return result

    def save_result(self, result: ExperimentResult, filename: Optional[str] = None):
        """Save experiment result to JSON file."""

        if filename is None:
            filename = (
                f"{self.task}_{self.dataset}_{self.teacher_model_name}_to_"
                f"{self.student_model_name}_{result.method}.json"
            )

        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"Result saved to {filepath}")

    def __del__(self):
        if hasattr(self, "writer"):
            self.writer.close()
