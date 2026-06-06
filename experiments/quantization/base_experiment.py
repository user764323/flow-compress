from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from flow_compress.quantization.faaq import FAAQQuantizer, FAAQReport
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""

    task: str
    dataset: str
    model: str
    method: str
    precision: float
    hardware: str
    metric_name: str
    metric_value: float
    baseline_metric: Optional[float] = None
    model_size_mb: float
    compression_ratio: float
    latency_ms: Optional[float] = None
    throughput: Optional[float] = None
    config: Optional[Dict[str, Any]] = None
    report: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""

        result = asdict(self)

        # Handle non-serializable objects
        if self.report:
            result['report'] = self._serialize_report(self.report)
        return result

    @staticmethod
    def _serialize_report(report: Any) -> Dict[str, Any]:
        """Serialize FAAQReport or other complex objects."""

        if isinstance(report, FAAQReport):
            return {
                'avg_bits_param_weighted': report.avg_bits_param_weighted,
                'total_params': report.total_params,
                'num_layers': len(report.layer_profiles),
            }

        return report if isinstance(report, dict) else str(report)


class BaseExperiment(ABC):
    """Base class for all experiments."""

    def __init__(
        self,
        task: str,
        dataset: str,
        model_name: str,
        device: Union[str, torch.device] = 'cuda',
        data_root: str = './data',
        output_dir: str = './experiments/results',
        log_dir: Optional[str] = None,
    ):
        self.task = task
        self.dataset = dataset
        self.model_name = model_name
        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu')
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize TensorBoard writer
        if log_dir is None:
            log_dir = self.output_dir / 'tensorboard' / f"{task}_{dataset}_{model_name}"
        else:
            log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(log_dir))
        self.global_step = 0

    @abstractmethod
    def load_model(self) -> nn.Module:
        """Load the model architecture."""
        pass

    @abstractmethod
    def load_dataset(self) -> Tuple[Any, Any]:
        """Load train and test datasets."""
        pass

    @abstractmethod
    def evaluate(self, model: nn.Module, test_loader: Any) -> float:
        """Evaluate model and return metric value."""
        pass

    @abstractmethod
    def get_metric_name(self) -> str:
        """Return the name of the evaluation metric."""
        pass

    def measure_latency(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        num_warmup: int = 10,
        num_runs: int = 100,
    ) -> float:
        """Measure inference latency in milliseconds."""

        model.eval()
        model.to(self.device)

        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(dummy_input)

        # Synchronize if using GPU
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        # Measure
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        end_time = time.time()
        avg_latency_ms = (end_time - start_time) / num_runs * 1000.0

        return avg_latency_ms

    def get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB."""

        param_size = sum(p.numel() * p.element_size()
                         for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size()
                          for b in model.buffers())
        total_size = param_size + buffer_size
        return total_size / (1024 * 1024)  # Convert to MB

    def get_quantized_model_size(self, report: FAAQReport) -> float:
        """Get quantized model size in MB based on bit allocation."""

        total_bits = sum(
            p.bits * p.num_params for p in report.layer_profiles.values()
        )
        # Add activation overhead (rough estimate: same as weights (2x))
        total_bits *= 2 

        # Convert to MB
        return total_bits / (8 * 1024 * 1024)

    def run_faaq_experiment(
        self,
        target_bits: float = 4.0,
        bmin: int = 2,
        bmax: int = 8,
        gamma: float = 5.0,
        num_calib_batches: int = 32,
    ) -> ExperimentResult:
        """Run experiment with FAAQ quantization."""

        logger.info(f"\n{'='*80}")
        logger.info(
            f"Running FAAQ experiment: {self.task} - {self.dataset} - {self.model_name}")
        logger.info(f"Target bits: {target_bits}, Range: [{bmin}, {bmax}]")
        logger.info(f"{'='*80}")

        # Load model and dataset
        model = self.load_model()
        train_loader, test_loader = self.load_dataset()

        # Evaluate baseline
        logger.info("Evaluating baseline (FP32)...")
        baseline_metric = self.evaluate(model, test_loader)
        logger.info(f"Baseline {self.get_metric_name()}: {baseline_metric:.4f}")

        # Measure baseline latency
        input_shape = self._get_input_shape(test_loader)
        baseline_latency = self.measure_latency(model, input_shape)
        baseline_size = self.get_model_size(model)

        # Log baseline metrics to TensorBoard
        self.writer.add_scalar('baseline/metric', baseline_metric, self.global_step)
        self.writer.add_scalar('baseline/latency_ms', baseline_latency, self.global_step)
        self.writer.add_scalar('baseline/model_size_mb', baseline_size, self.global_step)
        self.writer.add_scalar('config/target_bits', target_bits, self.global_step)
        self.writer.add_scalar('config/bmin', bmin, self.global_step)
        self.writer.add_scalar('config/bmax', bmax, self.global_step)
        self.writer.add_scalar('config/gamma', gamma, self.global_step)

        # Quantize with FAAQ
        logger.info(f"\nQuantizing with FAAQ (target_bits={target_bits})...")
        quantizer = FAAQQuantizer(
            model=model,
            b_target=target_bits,
            bmin=bmin,
            bmax=bmax,
            gamma=gamma,
            device=self.device,
            writer=self.writer,
            global_step=self.global_step,
        )

        quantized_model, report = quantizer.quantize(
            calib_loader=train_loader,
            num_calib_batches=num_calib_batches,
        )

        # Update global step after quantization
        self.global_step += 1

        # Evaluate quantized model
        logger.info(f"Evaluating quantized model...")
        quantized_metric = self.evaluate(quantized_model, test_loader)
        logger.info(f"Quantized {self.get_metric_name()}: {quantized_metric:.4f}")

        # Measure quantized latency
        quantized_latency = self.measure_latency(quantized_model, input_shape)
        quantized_size = self.get_quantized_model_size(report)
        compression_ratio = baseline_size / quantized_size if quantized_size > 0 else 1.0

        # Log quantized metrics to TensorBoard
        self.writer.add_scalar('quantized/metric', quantized_metric, self.global_step)
        self.writer.add_scalar('quantized/latency_ms', quantized_latency, self.global_step)
        self.writer.add_scalar('quantized/model_size_mb', quantized_size, self.global_step)
        self.writer.add_scalar('quantized/compression_ratio', compression_ratio, self.global_step)
        self.writer.add_scalar('quantized/avg_bits', report.avg_bits_param_weighted, self.global_step)
        self.writer.add_scalar('quantized/metric_drop', baseline_metric - quantized_metric, self.global_step)
        self.writer.add_scalar('quantized/latency_speedup', baseline_latency / quantized_latency if quantized_latency > 0 else 1.0, self.global_step)

        # Log bit allocation distribution
        bits_distribution = {}
        for name, profile in report.layer_profiles.items():
            bits = profile.bits
            if bits not in bits_distribution:
                bits_distribution[bits] = 0
            bits_distribution[bits] += profile.num_params
        for bits, count in bits_distribution.items():
            self.writer.add_scalar(f'bit_allocation/bits_{bits}', count, self.global_step)

        result = ExperimentResult(
            task=self.task,
            dataset=self.dataset,
            model=self.model_name,
            method='faaq',
            precision=report.avg_bits_param_weighted,
            hardware='gpu' if self.device.type == 'cuda' else 'cpu',
            metric_name=self.get_metric_name(),
            metric_value=quantized_metric,
            baseline_metric=baseline_metric,
            model_size_mb=quantized_size,
            compression_ratio=compression_ratio,
            latency_ms=quantized_latency,
            config={
                'target_bits': target_bits,
                'bmin': bmin,
                'bmax': bmax,
                'gamma': gamma,
            },
            report=report,
        )

        logger.info(f"\nResults:")
        logger.info(
            f"  Metric: {quantized_metric:.4f} (baseline: {baseline_metric:.4f})")
        logger.info(f"  Drop: {baseline_metric - quantized_metric:.4f}")
        logger.info(f"  Avg bits: {report.avg_bits_param_weighted:.2f}")
        logger.info(f"  Compression: {compression_ratio:.2f}x")
        logger.info(
            f"  Latency: {quantized_latency:.2f} ms (baseline: {baseline_latency:.2f} ms)")

        return result

    def _get_input_shape(self, test_loader: Any) -> Tuple[int, ...]:
        """Get input shape from test loader."""

        for batch in test_loader:
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            elif isinstance(batch, dict):
                x = next(iter(batch.values()))
            else:
                x = batch

            if torch.is_tensor(x):
                return x.shape
            break

        raise ValueError("Could not determine input shape")

    def save_result(self, result: ExperimentResult, filename: Optional[str] = None):
        """Save experiment result to JSON file."""

        if filename is None:
            filename = f"{self.task}_{self.dataset}_{self.model_name}_{result.method}_{result.precision}bits.json"

        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"Result saved to {filepath}")

    def __del__(self):
        """Close TensorBoard writer on cleanup."""

        if hasattr(self, 'writer'):
            self.writer.close()
