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
        {"dataset": "cifar10", "teacher": "resnet50", "student": "resnet18"},
        {"dataset": "cifar100", "teacher": "resnet50", "student": "resnet18"},
        {"dataset": "imagenet", "teacher": "resnet50", "student": "resnet18"},
    ]
}


def create_experiment(task: str, dataset: str, teacher: str, student: str, device: str, **kwargs):
    if task == "image_classification":
        return ImageClassificationExperiment(
            dataset=dataset,
            teacher_model_name=teacher,
            student_model_name=student,
            device=device,
            **kwargs,
        )
    raise ValueError(f"Unknown task: {task}")


def run_fad_experiments(
    tasks: List[str],
    device: str = "cuda",
    output_dir: str = "./experiments/results",
    num_epochs: int = 1,
) -> List[ExperimentResult]:
    results = []

    log_dir = Path(output_dir) / "tensorboard" / "all_distillation_experiments"
    log_dir.mkdir(parents=True, exist_ok=True)
    main_writer = SummaryWriter(str(log_dir))
    experiment_count = 0

    for task in tasks:
        if task not in EXPERIMENT_CONFIGS:
            logger.info(f"Warning: No config for task {task}")
            continue

        for config in EXPERIMENT_CONFIGS[task]:
            dataset = config["dataset"]
            teacher = config["teacher"]
            student = config["student"]

            logger.info(f"\n{'='*100}")
            logger.info(f"Task: {task}, Dataset: {dataset}, Teacher: {teacher}, Student: {student}")
            logger.info(f"{'='*100}")

            try:
                exp = create_experiment(
                    task=task,
                    dataset=dataset,
                    teacher=teacher,
                    student=student,
                    device=device,
                    output_dir=output_dir,
                )

                result = exp.run_fad_experiment(num_epochs=num_epochs)
                results.append(result)
                exp.save_result(result)

                main_writer.add_scalar(
                    f"{task}/{dataset}/{teacher}_to_{student}/metric",
                    result.metric_value,
                    experiment_count,
                )
                main_writer.add_scalar(
                    f"{task}/{dataset}/{teacher}_to_{student}/compression_ratio",
                    result.compression_ratio or 0.0,
                    experiment_count,
                )
                experiment_count += 1
            except Exception as e:
                logger.info(f"Error in FAD experiment: {e}")
                import traceback

                traceback.print_exc()

    main_writer.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Run FAD distillation experiments")
    parser.add_argument("--tasks", type=str, nargs="+", default=["image_classification"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="./experiments/results")
    parser.add_argument("--num-epochs", type=int, default=1)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_fad_experiments(
        tasks=args.tasks,
        device=args.device,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
    )


if __name__ == "__main__":
    main()
