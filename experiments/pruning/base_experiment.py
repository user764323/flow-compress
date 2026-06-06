from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from flow_compress.pruning.divergence_aware_pruning import divergence_aware_pruning
from flow_compress.utils.pruning_utils import evaluate_topk

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Results from a single pruning experiment."""

    task: str
    dataset: str
    model: str
    method: str
    metric_name: str
    metric_value: float
    baseline_metric: Optional[float] = None
    model_size_mb: Optional[float] = None
    pruned_size_mb: Optional[float] = None
    compression_ratio: Optional[float] = None
    config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseExperiment(ABC):
    """Base class for pruning experiments."""

    def __init__(
        self,
        task: str,
        dataset: str,
        model_name: str,
        device: str = "cuda",
        data_root: str = "./data",
        output_dir: str = "./experiments/results",
        log_dir: Optional[str] = None,
    ):
        self.task = task
        self.dataset = dataset
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if log_dir is None:
            log_dir = self.output_dir / "tensorboard" / f"{task}_{dataset}_{model_name}"
        else:
            log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(log_dir))

    @abstractmethod
    def load_model(self) -> nn.Module:
        """Load the model architecture."""
        pass

    @abstractmethod
    def load_dataset(self) -> Tuple[Any, Any]:
        """Load train and test datasets."""
        pass

    @abstractmethod
    def get_metric_name(self) -> str:
        """Return the name of the evaluation metric."""
        pass

    def get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB."""

        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        total_size = param_size + buffer_size
        return total_size / (1024 * 1024)

    def run_idap_experiment(
        self,
        max_performance_metric_degradation_th: float = 0.1,
        number_of_pruning_iterations: int = 10,
        base_pruning_percentage: float = 0.05,
    ) -> ExperimentResult:
        """Run experiment with IDAP++ pruning."""

        logger.info(f"\n{'='*80}")
        logger.info(f"Running IDAP++ experiment: {self.task} - {self.dataset} - {self.model_name}")
        logger.info(f"{'='*80}")

        model = self.load_model()
        train_loader, test_loader = self.load_dataset()

        logger.info("Evaluating baseline (FP32)...")
        baseline_metric = evaluate_topk(model, test_loader, self.device, k=1)
        logger.info(f"Baseline {self.get_metric_name()}: {baseline_metric:.4f}")

        model_path = self.output_dir / f"{self.task}_{self.dataset}_{self.model_name}_pruned.pth"
        pruned_model = divergence_aware_pruning(
            model_path=str(model_path),
            pretrained_model=model,
            test_loader=test_loader,
            train_loader=train_loader,
            num_classes=self._get_num_classes(),
            max_performance_metric_degradation_th=max_performance_metric_degradation_th,
            number_of_pruning_iterations=number_of_pruning_iterations,
            base_pruning_percentage=base_pruning_percentage,
            device=str(self.device),
            save_checkpoints=False,
            writer=self.writer,
        )

        pruned_metric = evaluate_topk(pruned_model, test_loader, self.device, k=1)
        model_size = self.get_model_size(model)
        pruned_size = self.get_model_size(pruned_model)
        compression_ratio = model_size / max(1e-8, pruned_size)

        result = ExperimentResult(
            task=self.task,
            dataset=self.dataset,
            model=self.model_name,
            method="idap",
            metric_name=self.get_metric_name(),
            metric_value=pruned_metric,
            baseline_metric=baseline_metric,
            model_size_mb=model_size,
            pruned_size_mb=pruned_size,
            compression_ratio=compression_ratio,
            config={
                "max_performance_metric_degradation_th": max_performance_metric_degradation_th,
                "number_of_pruning_iterations": number_of_pruning_iterations,
                "base_pruning_percentage": base_pruning_percentage,
            },
        )

        logger.info(f"Pruned {self.get_metric_name()}: {pruned_metric:.4f}")
        logger.info(f"Compression: {compression_ratio:.2f}x")

        return result

    def save_result(self, result: ExperimentResult, filename: Optional[str] = None):
        """Save experiment result to JSON file."""

        if filename is None:
            filename = f"{self.task}_{self.dataset}_{self.model_name}_{result.method}.json"

        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"Result saved to {filepath}")

    def _get_num_classes(self) -> int:
        dataset_map = {
            "cifar10": 10,
            "cifar100": 100,
            "imagenet": 1000,
        }
        return dataset_map.get(self.dataset.lower(), 1000)

    def __del__(self):
        if hasattr(self, "writer"):
            self.writer.close()
