import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

from base_experiment import ExperimentResult
from image_classification import ImageClassificationExperiment
import torch
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


EXPERIMENT_CONFIGS = {
    "image_classification": [
        {"dataset": "cifar10", "model": "resnet50"},
        {"dataset": "cifar100", "model": "resnet50"},
        {"dataset": "imagenet", "model": "resnet50"},
    ]
}


def create_experiment(task: str, dataset: str, model: str, device: str, **kwargs):
    if task == "image_classification":
        return ImageClassificationExperiment(
            dataset=dataset,
            model_name=model,
            device=device,
            **kwargs,
        )
    raise ValueError(f"Unknown task: {task}")


def run_idap_experiments(
    tasks: List[str],
    device: str = "cuda",
    output_dir: str = "./experiments/results",
    number_of_pruning_iterations: int = 5,
    base_pruning_percentage: float = 0.05,
    max_performance_metric_degradation_th: float = 0.1,
) -> List[ExperimentResult]:
    results = []

    log_dir = Path(output_dir) / "tensorboard" / "all_pruning_experiments"
    log_dir.mkdir(parents=True, exist_ok=True)
    main_writer = SummaryWriter(str(log_dir))
    experiment_count = 0

    for task in tasks:
        if task not in EXPERIMENT_CONFIGS:
            logger.info(f"Warning: No config for task {task}")
            continue

        for config in EXPERIMENT_CONFIGS[task]:
            dataset = config["dataset"]
            model = config["model"]

            logger.info(f"\n{'='*100}")
            logger.info(f"Task: {task}, Dataset: {dataset}, Model: {model}")
            logger.info(f"{'='*100}")

            try:
                exp = create_experiment(
                    task=task,
                    dataset=dataset,
                    model=model,
                    device=device,
                    output_dir=output_dir,
                )

                result = exp.run_idap_experiment(
                    max_performance_metric_degradation_th=max_performance_metric_degradation_th,
                    number_of_pruning_iterations=number_of_pruning_iterations,
                    base_pruning_percentage=base_pruning_percentage,
                )
                results.append(result)
                exp.save_result(result)

                main_writer.add_scalar(
                    f"{task}/{dataset}/{model}/metric",
                    result.metric_value,
                    experiment_count,
                )
                main_writer.add_scalar(
                    f"{task}/{dataset}/{model}/compression_ratio",
                    result.compression_ratio or 0.0,
                    experiment_count,
                )
                experiment_count += 1
            except Exception as e:
                logger.info(f"Error in IDAP experiment: {e}")
                import traceback

                traceback.print_exc()

    main_writer.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Run IDAP++ pruning experiments")
    parser.add_argument("--tasks", type=str, nargs="+", default=["image_classification"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="./experiments/results")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--base-pruning-percentage", type=float, default=0.05)
    parser.add_argument("--max-metric-drop", type=float, default=0.1)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_idap_experiments(
        tasks=args.tasks,
        device=args.device,
        output_dir=args.output_dir,
        number_of_pruning_iterations=args.iterations,
        base_pruning_percentage=args.base_pruning_percentage,
        max_performance_metric_degradation_th=args.max_metric_drop,
    )


if __name__ == "__main__":
    main()
