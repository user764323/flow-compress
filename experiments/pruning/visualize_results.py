import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def load_results(results_dir: str) -> List[Dict[str, Any]]:
    results_path = Path(results_dir)
    results = []
    for json_file in results_path.glob("*.json"):
        if json_file.name == "summary_report.json":
            continue
        try:
            with open(json_file, "r") as f:
                results.append(json.load(f))
        except Exception as e:
            logger.info(f"Error loading {json_file}: {e}")
    return results


def create_comparison_table(results: List[Dict[str, Any]], output_dir: Path):
    df_data = []
    for r in results:
        df_data.append(
            {
                "Task": r.get("task", "unknown"),
                "Dataset": r.get("dataset", "unknown"),
                "Model": r.get("model", "unknown"),
                "Metric": f"{r.get('metric_value', 0):.2f}",
                "Baseline": f"{r.get('baseline_metric', 0):.2f}",
                "Drop": f"{r.get('baseline_metric', 0) - r.get('metric_value', 0):.2f}",
                "Compression": f"{r.get('compression_ratio', 0):.2f}x",
            }
        )

    df = pd.DataFrame(df_data)
    csv_path = output_dir / "comparison_table.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Comparison table saved to {csv_path}")
    logger.info("\n" + df.to_markdown(index=False))


def plot_compression_vs_accuracy(results: List[Dict[str, Any]], output_dir: Path):
    if not results:
        return
    compressions = [r.get("compression_ratio", 0) for r in results]
    metrics = [r.get("metric_value", 0) for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(compressions, metrics, s=100, alpha=0.6, label="IDAP++")
    ax.set_xlabel("Compression Ratio")
    ax.set_ylabel("Metric Value")
    ax.set_title("Compression vs Accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plot_path = output_dir / "compression_vs_accuracy.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize pruning results")
    parser.add_argument("--results-dir", type=str, default="./experiments/results")
    parser.add_argument("--output", type=str, default="./experiments/results/pruning_viz")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    results = load_results(args.results_dir)
    if not results:
        logger.info("No results found!")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    create_comparison_table(results, output_dir)
    plot_compression_vs_accuracy(results, output_dir)


if __name__ == "__main__":
    main()
