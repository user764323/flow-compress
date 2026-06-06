from typing import Tuple

from base_experiment import BaseExperiment
import timm
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


class ImageClassificationExperiment(BaseExperiment):
    """Pruning experiment for image classification tasks."""

    def __init__(
        self,
        dataset: str,
        model_name: str,
        device: str = "cuda",
        data_root: str = "./data",
        output_dir: str = "./experiments/results",
        batch_size: int = 32,
    ):
        super().__init__(
            task="image_classification",
            dataset=dataset,
            model_name=model_name,
            device=device,
            data_root=data_root,
            output_dir=output_dir,
        )
        self.batch_size = batch_size

    def load_model(self) -> nn.Module:
        """Load classification model."""
        try:
            if any(keyword in self.model_name.lower() for keyword in ["vit", "efficientnet", "convnext"]):
                model = timm.create_model(self.model_name, pretrained=True)
                model.reset_classifier(self._get_num_classes())
                return model
        except Exception:
            pass

        model_name_lower = self.model_name.lower()
        if "resnet" in model_name_lower:
            model = torchvision.models.resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, self._get_num_classes())
        elif "vgg" in model_name_lower:
            model = torchvision.models.vgg16(pretrained=True)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, self._get_num_classes())
        elif "densenet" in model_name_lower:
            model = torchvision.models.densenet121(pretrained=True)
            model.classifier = nn.Linear(model.classifier.in_features, self._get_num_classes())
        elif "mobilenet" in model_name_lower:
            model = torchvision.models.mobilenet_v2(pretrained=True)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, self._get_num_classes())
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        return model

    def load_dataset(self) -> Tuple[DataLoader, DataLoader]:
        dataset_name = self.dataset.lower()
        if dataset_name == "cifar10":
            return self._load_cifar10()
        if dataset_name == "cifar100":
            return self._load_cifar100()
        if dataset_name == "imagenet":
            return self._load_imagenet()
        raise ValueError(f"Unknown dataset: {dataset_name}")

    def _load_cifar10(self) -> Tuple[DataLoader, DataLoader]:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

        train_set = torchvision.datasets.CIFAR10(
            root=str(self.data_root), train=True, download=True, transform=transform_train
        )
        test_set = torchvision.datasets.CIFAR10(
            root=str(self.data_root), train=False, download=True, transform=transform_test
        )

        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

        return train_loader, test_loader

    def _load_cifar100(self) -> Tuple[DataLoader, DataLoader]:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )

        train_set = torchvision.datasets.CIFAR100(
            root=str(self.data_root), train=True, download=True, transform=transform_train
        )
        test_set = torchvision.datasets.CIFAR100(
            root=str(self.data_root), train=False, download=True, transform=transform_test
        )

        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

        return train_loader, test_loader

    def _load_imagenet(self) -> Tuple[DataLoader, DataLoader]:
        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        train_set = torchvision.datasets.ImageNet(
            root=str(self.data_root), split="train", download=True, transform=transform
        )
        test_set = torchvision.datasets.ImageNet(
            root=str(self.data_root), split="val", download=True, transform=transform
        )

        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

        return train_loader, test_loader

    def get_metric_name(self) -> str:
        return "Acc@1"
