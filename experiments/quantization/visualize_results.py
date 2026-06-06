import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def load_results(results_dir: str) -> List[Dict[str, Any]]:
    """Load all result JSON files."""

    results_path = Path(results_dir)
    results = []

    for json_file in results_path.glob("*.json"):
        if json_file.name == "summary_report.json":
            continue

        try:
            with open(json_file, "r") as f:
                result = json.load(f)
                results.append(result)
        except Exception as e:
            logger.info(f"Error loading {json_file}: {e}")

    return results


def create_comparison_table(results: List[Dict[str, Any]], output_path: str):
    """Create comparison table of results."""

    df_data = []

    for r in results:
        df_data.append(
            {
                "Task": r.get("task", "unknown"),
                "Dataset": r.get("dataset", "unknown"),
                "Model": r.get("model", "unknown"),
                "Method": r.get("method", "unknown"),
                "Precision (bits)": f"{r.get('precision', 0):.1f}",
                "Metric": f"{r.get('metric_value', 0):.2f}",
                "Baseline": f"{r.get('baseline_metric', 0):.2f}",
                "Drop": f"{r.get('baseline_metric', 0) - r.get('metric_value', 0):.2f}",
                "Compression": f"{r.get('compression_ratio', 0):.2f}x",
                "Latency (ms)": (
                    f"{r.get('latency_ms', 0):.2f}" if r.get(
                        "latency_ms") else "N/A"
                ),
            }
        )

    df = pd.DataFrame(df_data)

    # Save as CSV
    csv_path = Path(output_path).parent / "comparison_table.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Comparison table saved to {csv_path}")

    # Print markdown table
    logger.info("\n" + "=" * 100)
    logger.info("Results Summary (Markdown)")
    logger.info("=" * 100)
    logger.info(df.to_markdown(index=False))


def plot_accuracy_vs_precision(results: List[Dict[str, Any]], output_path: str):
    """Plot accuracy vs precision."""

    faaq_results = [r for r in results if r.get("method") == "faaq"]

    if not faaq_results:
        logger.info("No FAAQ results to plot")
        return

    # Group by task and dataset
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    tasks = list(set(r.get("task", "unknown") for r in faaq_results))

    for idx, task in enumerate(tasks[:4]):
        ax = axes[idx]
        task_results = [r for r in faaq_results if r.get("task") == task]

        if not task_results:
            continue

        precisions = [r.get("precision", 0) for r in task_results]
        metrics = [r.get("metric_value", 0) for r in task_results]
        baselines = [r.get("baseline_metric", 0) for r in task_results]

        ax.plot(precisions, metrics, "o-",
                label="FAAQ", linewidth=2, markersize=8)
        if baselines and baselines[0]:
            ax.axhline(y=baselines[0], color="r",
                       linestyle="--", label="FP32 Baseline")

        ax.set_xlabel("Precision (bits)")
        ax.set_ylabel("Metric Value")
        ax.set_title(f"{task}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = Path(output_path).parent / "accuracy_vs_precision.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved to {plot_path}")


def plot_compression_vs_accuracy(results: List[Dict[str, Any]], output_path: str):
    """Plot compression ratio vs accuracy."""

    faaq_results = [r for r in results if r.get("method") == "faaq"]

    if not faaq_results:
        logger.info("No FAAQ results to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    compressions = [r.get("compression_ratio", 0) for r in faaq_results]
    metrics = [r.get("metric_value", 0) for r in faaq_results]
    baselines = [r.get("baseline_metric", 0) for r in faaq_results]
    drops = [b - m for b, m in zip(baselines, metrics) if b and m]

    ax.scatter(compressions, metrics, s=100, alpha=0.6, label="FAAQ")

    ax.set_xlabel("Compression Ratio")
    ax.set_ylabel("Metric Value")
    ax.set_title("Compression vs Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = Path(output_path).parent / "compression_vs_accuracy.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize experiment results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./experiments/results",
        help="Directory containing result JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./experiments/results/visualizations",
        help="Output directory for visualizations",
    )

    args = parser.parse_args()

    # Load results
    logger.info(f"Loading results from {args.results_dir}...")
    results = load_results(args.results_dir)
    logger.info(f"Loaded {len(results)} results")

    if not results:
        logger.info("No results found!")
        return

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create visualizations
    logger.info("\nGenerating visualizations...")
    create_comparison_table(results, str(output_dir / "comparison.md"))
    plot_accuracy_vs_precision(results, str(output_dir / "plot.png"))
    plot_compression_vs_accuracy(results, str(output_dir / "plot.png"))

    logger.info("\nVisualization complete!")


if __name__ == "__main__":
    main()
