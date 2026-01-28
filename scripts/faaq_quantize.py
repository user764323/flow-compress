import argparse
import logging
import os
from pathlib import Path
import sys

from flow_compress.quantization.faaq import FAAQQuantizer
from flow_compress.utils.quantization_utils import format_faaq_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger(__name__)


def load_model(model_path_or_name: str, num_classes: int = None) -> nn.Module:
    """Load a model from file or create a standard architecture."""

    # Try loading from file first
    if os.path.exists(model_path_or_name):
        logger.info(f"Loading model from {model_path_or_name}")
        checkpoint = torch.load(model_path_or_name, map_location="cpu")

        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                model = checkpoint["model"]
            elif "state_dict" in checkpoint:
                model = torchvision.models.resnet50(pretrained=False)
                if num_classes:
                    model.fc = nn.Linear(model.fc.in_features, num_classes)
                model.load_state_dict(checkpoint["state_dict"])
            else:
                raise ValueError("Unknown checkpoint format")
        else:
            model = checkpoint
        return model

    model_name = model_path_or_name.lower()

    if "resnet" in model_name:
        if "18" in model_name:
            model = torchvision.models.resnet18(pretrained=True)
        elif "34" in model_name:
            model = torchvision.models.resnet34(pretrained=True)
        elif "50" in model_name:
            model = torchvision.models.resnet50(pretrained=True)
        elif "101" in model_name:
            model = torchvision.models.resnet101(pretrained=True)
        else:
            model = torchvision.models.resnet50(pretrained=True)

        if num_classes and num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        logger.info(f"Created {model_name} model")
        return model

    raise ValueError(f"Could not load or create model: {model_path_or_name}")


def get_dataset(dataset_name: str, root: str = "./data", batch_size: int = 32):
    """Get dataset and dataloader."""

    dataset_name = dataset_name.lower()

    if dataset_name == "cifar10":
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        train_set = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True, transform=transform_train
        )
        test_set = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=transform_test
        )
        num_classes = 10

    elif dataset_name == "cifar100":
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )

        train_set = torchvision.datasets.CIFAR100(
            root=root, train=True, download=True, transform=transform_train
        )
        test_set = torchvision.datasets.CIFAR100(
            root=root, train=False, download=True, transform=transform_test
        )
        num_classes = 100

    elif dataset_name == "imagenet":
        try:
            from torchvision.datasets import ImageNet

            transform_train = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            train_set = ImageNet(root=root, split="train",
                                 transform=transform_train)
            test_set = ImageNet(root=root, split="val",
                                transform=transform_test)
            num_classes = 1000

        except (ImportError, AttributeError):
            from torchvision.datasets import ImageFolder

            train_dir = os.path.join(root, "imagenet", "train")
            val_dir = os.path.join(root, "imagenet", "val")

            if not os.path.exists(train_dir) or not os.path.exists(val_dir):
                raise ValueError(
                    f"ImageNet dataset not found. Please download ImageNet and place it in:\n"
                    f"  Train: {train_dir}\n"
                    f"  Val: {val_dir}\n"
                    f"Or use ImageFolder structure with these directories."
                )

            transform_train = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            train_set = ImageFolder(train_dir, transform=transform_train)
            test_set = ImageFolder(val_dir, transform=transform_test)
            num_classes = len(train_set.classes)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, test_loader, num_classes


def evaluate_model(
    model: nn.Module, test_loader: DataLoader, device: torch.device
) -> float:
    """Evaluate model accuracy on test set."""

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


def main():
    parser = argparse.ArgumentParser(
        description="FAAQ: Flow-Aware Adaptive Quantization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model file or model name (e.g., resnet50)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100", "imagenet"],
        help="Dataset name",
    )
    parser.add_argument(
        "--target-bits", type=float, default=4.0, help="Target average bit-width"
    )
    parser.add_argument("--bmin", type=int, default=2,
                        help="Minimum bit-width")
    parser.add_argument("--bmax", type=int, default=8,
                        help="Maximum bit-width")
    parser.add_argument(
        "--gamma", type=float, default=5.0, help="Gradient sensitivity parameter"
    )
    parser.add_argument(
        "--num-calib-batches",
        type=int,
        default=32,
        help="Number of calibration batches",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output path for quantized model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--data-root", type=str, default="./data", help="Root directory for datasets"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for calibration"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate model before and after quantization",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    train_loader, test_loader, num_classes = get_dataset(
        args.dataset, root=args.data_root, batch_size=args.batch_size
    )

    # Load model
    model = load_model(args.model, num_classes=num_classes)
    model = model.to(device)
    model.eval()

    # Evaluate original model if requested
    if args.eval:
        logger.info("Evaluating original model...")
        orig_accuracy = evaluate_model(model, test_loader, device)
        logger.info(f"Original model accuracy: {orig_accuracy:.2f}%")

    # Create quantizer
    logger.info(f"\nInitializing FAAQ quantizer...")
    logger.info(f"  Target bits: {args.target_bits}")
    logger.info(f"  Bit range: [{args.bmin}, {args.bmax}]")
    logger.info(f"  Gamma: {args.gamma}")
    logger.info(f"  Calibration batches: {args.num_calib_batches}")

    quantizer = FAAQQuantizer(
        model=model,
        b_target=args.target_bits,
        bmin=args.bmin,
        bmax=args.bmax,
        gamma=args.gamma,
        device=device,
    )

    # Quantize
    logger.info("\nStarting quantization...")
    quantized_model, report = quantizer.quantize(
        calib_loader=train_loader,
        num_calib_batches=args.num_calib_batches,
    )

    # Print report
    logger.info("\n" + "=" * 80)
    logger.info("FAAQ Quantization Report")
    logger.info("=" * 80)
    logger.info(format_faaq_report(report, topk=30))
    logger.info("=" * 80)

    # Evaluate quantized model if requested
    if args.eval:
        logger.info("\nEvaluating quantized model...")
        quant_accuracy = evaluate_model(quantized_model, test_loader, device)
        logger.info(f"Quantized model accuracy: {quant_accuracy:.2f}%")
        if args.eval:
            logger.info(f"Accuracy drop: {orig_accuracy - quant_accuracy:.2f}%")

    # Save quantized model
    if args.output:
        logger.info(f"\nSaving quantized model to {args.output}")
        torch.save(
            {
                "model": quantized_model,
                "report": report,
                "config": {
                    "target_bits": args.target_bits,
                    "bmin": args.bmin,
                    "bmax": args.bmax,
                    "gamma": args.gamma,
                },
            },
            args.output,
        )
        logger.info("Model saved successfully!")

    logger.info("\nQuantization complete!")


if __name__ == "__main__":
    main()
