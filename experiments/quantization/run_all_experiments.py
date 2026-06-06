import argparse
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List

from base_experiment import ExperimentResult
from image_classification import ImageClassificationExperiment
from image_generation import ImageGenerationExperiment
from image_segmentation import ImageSegmentationExperiment
from object_detection import ObjectDetectionExperiment
from text_classification import TextClassificationExperiment
import torch
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


# Experiment configurations
EXPERIMENT_CONFIGS = {
    "image_classification": [
        {"dataset": "cifar10", "model": "resnet50"},
        {"dataset": "cifar10", "model": "resnet50"},
        {"dataset": "cifar100", "model": "resnet50"},
        {"dataset": "imagenet", "model": "resnet50"},
        {"dataset": "stanford_cars", "model": "convnext_small"},
        {"dataset": "food101", "model": "efficientnet_b4"},
    ],
    "image_segmentation": [
        {"dataset": "cityscapes", "model": "deeplabv3_resnet50"},
        {"dataset": "pascal_voc", "model": "deeplabv3_resnet50"},
        {"dataset": "coco", "model": "deeplabv3_resnet101"},
        {"dataset": "ade20k", "model": "deeplabv3_resnet50"},
    ],
    "object_detection": [
        {"dataset": "pascal_voc", "model": "fasterrcnn_resnet50"},
        {"dataset": "coco", "model": "fasterrcnn_resnet50"},
        {"dataset": "open_images", "model": "fasterrcnn_resnet50"},
        {"dataset": "objects365", "model": "fasterrcnn_resnet50"},
    ],
    "image_generation": [
        {"dataset": "cifar10", "model": "dcgan"},
        {"dataset": "coco_stuff", "model": "dcgan"},
        {"dataset": "coco", "model": "dcgan"},
    ],
    "text_classification": [
        {"dataset": "imdb", "model": "bert-base-uncased"},
        {"dataset": "ag_news", "model": "bert-base-uncased"},
        {"dataset": "yelp", "model": "roberta-base"},
        {"dataset": "amazon", "model": "distilbert-base-uncased"},
    ],
}

# Precision configurations
PRECISION_CONFIGS = [
    {"target_bits": 2.0, "bmin": 2, "bmax": 4},
    {"target_bits": 3.0, "bmin": 2, "bmax": 6},
    {"target_bits": 4.0, "bmin": 2, "bmax": 8},
    {"target_bits": 6.0, "bmin": 4, "bmax": 8},
    {"target_bits": 8.0, "bmin": 6, "bmax": 8},
]

# methods for comparison
METHODS = [
    "uniform",
    "adaround",
    "brecq",
    "smoothquant",
    "ptq4vit",  # For ViT models only
    "lsq",
]


def create_experiment(task: str, dataset: str, model: str, device: str, **kwargs):
    """Create experiment instance based on task."""

    if task == "image_classification":
        return ImageClassificationExperiment(
            dataset=dataset, model_name=model, device=device, **kwargs
        )
    elif task == "image_segmentation":
        return ImageSegmentationExperiment(
            dataset=dataset, model_name=model, device=device, **kwargs
        )
    elif task == "object_detection":
        return ObjectDetectionExperiment(
            dataset=dataset, model_name=model, device=device, **kwargs
        )
    elif task == "image_generation":
        return ImageGenerationExperiment(
            dataset=dataset, model_name=model, device=device, **kwargs
        )
    elif task == "text_classification":
        return TextClassificationExperiment(
            dataset=dataset, model_name=model, device=device, **kwargs
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def run_faaq_experiments(
    tasks: List[str],
    precisions: List[Dict[str, Any]],
    device: str = "cuda",
    output_dir: str = "./experiments/results",
    num_calib_batches: int = 32,
) -> List[ExperimentResult]:
    """Run all FAAQ experiments."""

    results = []

    # Initialize TensorBoard writer for overall experiment tracking
    log_dir = Path(output_dir) / 'tensorboard' / 'all_experiments'
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

                for prec_config in precisions:
                    target_bits = prec_config["target_bits"]
                    bmin = prec_config["bmin"]
                    bmax = prec_config["bmax"]

                    try:
                        result = exp.run_faaq_experiment(
                            target_bits=target_bits,
                            bmin=bmin,
                            bmax=bmax,
                            num_calib_batches=num_calib_batches,
                        )
                        results.append(result)
                        exp.save_result(result)

                        # Log to main TensorBoard writer
                        main_writer.add_scalar(f'{task}/{dataset}/{model}/metric', result.metric_value, experiment_count)
                        main_writer.add_scalar(f'{task}/{dataset}/{model}/compression_ratio', result.compression_ratio, experiment_count)
                        main_writer.add_scalar(f'{task}/{dataset}/{model}/avg_bits', result.precision, experiment_count)
                        main_writer.add_scalar(f'{task}/{dataset}/{model}/metric_drop', result.baseline_metric - result.metric_value if result.baseline_metric else 0, experiment_count)
                        experiment_count += 1
                    except Exception as e:
                        logger.info(f"Error in FAAQ experiment: {e}")
                        import traceback

                        traceback.print_exc()

            except Exception as e:
                logger.info(f"Error creating experiment: {e}")
                import traceback

                traceback.print_exc()

    main_writer.close()
    return results


def run_method_comparisons(
    tasks: List[str],
    device: str = "cuda",
    output_dir: str = "./experiments/results",
    num_calib_batches: int = 32,
) -> List[ExperimentResult]:
    """Run method comparisons."""

    results = []

    # Run on a subset of experiments for comparison
    comparison_configs = [
        {"task": "image_classification", "dataset": "cifar10", "model": "resnet50"},
        {"task": "image_classification", "dataset": "imagenet", "model": "resnet50"},
    ]

    for config in comparison_configs:
        task = config["task"]
        dataset = config["dataset"]
        model = config["model"]

        logger.info(f"\n{'='*100}")
        logger.info(f"Method Comparison: {task} - {dataset} - {model}")
        logger.info(f"{'='*100}")

        try:
            exp = create_experiment(
                task=task,
                dataset=dataset,
                model=model,
                device=device,
                output_dir=output_dir,
            )

            # Run FAAQ for comparison
            faaq_result = exp.run_faaq_experiment(
                target_bits=4.0,
                bmin=2,
                bmax=8,
                num_calib_batches=num_calib_batches,
            )
            results.append(faaq_result)

            # Run method baselines
            for method_name in METHODS:
                # Skip PTQ4ViT for non-ViT models
                if method_name == "ptq4vit" and "vit" not in model.lower():
                    continue

                try:
                    baseline = get_method(method_name)
                    logger.info(
                        f"method_name: {method_name}")
                except Exception as e:
                    logger.info(f"  Error with {method_name}: {e}")

        except Exception as e:
            logger.info(f"Error in method comparison: {e}")
            import traceback

            traceback.print_exc()

    return results


def run_hardware_comparisons(
    tasks: List[str],
    precisions: List[Dict[str, Any]],
    output_dir: str = "./experiments/results",
    num_calib_batches: int = 32,
) -> List[ExperimentResult]:
    """Run experiments on different hardware."""

    results = []

    hardware_configs = ["cpu", "cuda"]

    for hardware in hardware_configs:
        if hardware == "cuda" and not torch.cuda.is_available():
            logger.info(f"Skipping {hardware} - not available")
            continue

        logger.info(f"\n{'='*100}")
        logger.info(f"Hardware: {hardware}")
        logger.info(f"{'='*100}")

        # Run a subset of experiments for hardware comparison
        test_configs = [
            {"task": "image_classification",
                "dataset": "cifar10", "model": "resnet50"},
        ]

        for config in test_configs:
            task = config["task"]
            dataset = config["dataset"]
            model = config["model"]

            try:
                exp = create_experiment(
                    task=task,
                    dataset=dataset,
                    model=model,
                    device=hardware,
                    output_dir=output_dir,
                )

                # Test with first 2 precisions
                for prec_config in precisions[:2]:
                    target_bits = prec_config["target_bits"]
                    bmin = prec_config["bmin"]
                    bmax = prec_config["bmax"]

                    try:
                        result = exp.run_faaq_experiment(
                            target_bits=target_bits,
                            bmin=bmin,
                            bmax=bmax,
                            num_calib_batches=num_calib_batches,
                        )
                        result.hardware = hardware
                        results.append(result)
                        exp.save_result(result)
                    except Exception as e:
                        logger.info(f"Error: {e}")

            except Exception as e:
                logger.info(f"Error: {e}")

    return results


def generate_summary_report(results: List[ExperimentResult], output_dir: str):
    """Generate summary report from all results."""

    output_path = Path(output_dir) / "summary_report.json"

    summary = {
        "total_experiments": len(results),
        "by_task": {},
        "by_precision": {},
        "by_method": {},
        "by_hardware": {},
        "results": [r.to_dict() for r in results],
    }

    # Group by task
    for result in results:
        task = result.task
        if task not in summary["by_task"]:
            summary["by_task"][task] = []
        summary["by_task"][task].append(result.to_dict())

    # Group by precision
    for result in results:
        prec = f"{result.precision:.1f}bits"
        if prec not in summary["by_precision"]:
            summary["by_precision"][prec] = []
        summary["by_precision"][prec].append(result.to_dict())

    # Group by method
    for result in results:
        method = result.method
        if method not in summary["by_method"]:
            summary["by_method"][method] = []
        summary["by_method"][method].append(result.to_dict())

    # Group by hardware
    for result in results:
        hw = result.hardware
        if hw not in summary["by_hardware"]:
            summary["by_hardware"][hw] = []
        summary["by_hardware"][hw].append(result.to_dict())

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nSummary report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive FAAQ experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["image_classification"],
        choices=[
            "image_classification",
            "image_segmentation",
            "object_detection",
            "image_generation",
            "text_classification"
        ],
        help="Tasks to run",
    )

    parser.add_argument(
        "--precisions",
        nargs="+",
        type=float,
        default=[2.0, 3.0, 4.0, 6.0, 8.0],
        help="Target precisions (bits)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiments/results",
        help="Output directory for results",
    )

    parser.add_argument(
        "--num-calib-batches",
        type=int,
        default=32,
        help="Number of calibration batches",
    )

    parser.add_argument("--run-method", action="store_true",
                        help="Run method comparisons")

    parser.add_argument(
        "--run-hardware", action="store_true", help="Run hardware comparisons"
    )

    parser.add_argument(
        "--quick", action="store_true", help="Run quick test (fewer experiments)"
    )

    args = parser.parse_args()

    # Prepare tasks
    if "all" in args.tasks:
        tasks = [
            "image_classification",
            "image_segmentation",
            "object_detection",
            "image_generation",
            "text_classification",
        ]
    else:
        tasks = args.tasks

    # Prepare precisions
    precisions = []
    for target_bits in args.precisions:
        if target_bits <= 2:
            precisions.append(
                {"target_bits": target_bits, "bmin": 2, "bmax": 4})
        elif target_bits <= 4:
            precisions.append(
                {"target_bits": target_bits, "bmin": 2, "bmax": 8})
        else:
            precisions.append(
                {"target_bits": target_bits, "bmin": 4, "bmax": 8})

    if args.quick:
        # Run only first precision and first task
        precisions = precisions[:1]
        tasks = tasks[:1]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running experiments:")
    logger.info(f"  Tasks: {tasks}")
    logger.info(f"  Precisions: {[p['target_bits'] for p in precisions]}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Output: {output_dir}")

    all_results = []

    # Run main FAAQ experiments
    logger.info("\n" + "=" * 100)
    logger.info("Running FAAQ Experiments")
    logger.info("=" * 100)
    faaq_results = run_faaq_experiments(
        tasks=tasks,
        precisions=precisions,
        device=args.device,
        output_dir=str(output_dir),
        num_calib_batches=args.num_calib_batches,
    )
    all_results.extend(faaq_results)

    # Run method comparisons
    if args.run_method:
        logger.info("\n" + "=" * 100)
        logger.info("Running Method Comparisons")
        logger.info("=" * 100)
        method_results = run_method_comparisons(
            tasks=tasks,
            device=args.device,
            output_dir=str(output_dir),
            num_calib_batches=args.num_calib_batches,
        )
        all_results.extend(method_results)

    # Run hardware comparisons
    if args.run_hardware:
        logger.info("\n" + "=" * 100)
        logger.info("Running Hardware Comparisons")
        logger.info("=" * 100)
        hw_results = run_hardware_comparisons(
            tasks=tasks,
            precisions=precisions,
            output_dir=str(output_dir),
            num_calib_batches=args.num_calib_batches,
        )
        all_results.extend(hw_results)

    # Generate summary
    logger.info("\n" + "=" * 100)
    logger.info("Generating Summary Report")
    logger.info("=" * 100)
    generate_summary_report(all_results, str(output_dir))

    logger.info(f"\n{'='*100}")
    logger.info(f"All experiments completed!")
    logger.info(f"Total experiments: {len(all_results)}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"{'='*100}")


if __name__ == "__main__":
    main()
