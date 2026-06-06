from typing import Any, Tuple

from experiments.base_experiment import BaseExperiment, ExperimentResult
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


class ImageClassificationExperiment(BaseExperiment):
    """Experiment for image classification tasks."""

    def __init__(
        self,
        dataset: str,
        model_name: str,
        device: str = 'cuda',
        data_root: str = './data',
        output_dir: str = './experiments/results',
        batch_size: int = 32,
    ):
        super().__init__(
            task='image_classification',
            dataset=dataset,
            model_name=model_name,
            device=device,
            data_root=data_root,
            output_dir=output_dir,
        )
        self.batch_size = batch_size

    def load_model(self) -> nn.Module:
        """Load classification model."""

        # Try timm models first (for efficientnet, vit, convnext, shufflenet, etc.)
        try:
            if any(keyword in self.model_name.lower() for keyword in 
                   ['vit', 'efficientnet', 'convnext', 'shufflenet']):
                model = timm.create_model(self.model_name, pretrained=True)
                num_classes = self._get_num_classes()
                # Adjust output layer for timm models
                if hasattr(model, 'get_classifier'):
                    in_features = model.get_classifier().in_features if hasattr(model.get_classifier(), 'in_features') else model.num_features
                    model.reset_classifier(num_classes)
                elif hasattr(model, 'classifier'):
                    if isinstance(model.classifier, nn.Sequential):
                        model.classifier[-1] = nn.Linear(
                            model.classifier[-1].in_features, num_classes)
                    else:
                        model.classifier = nn.Linear(
                            model.classifier.in_features, num_classes)
                elif hasattr(model, 'head'):
                    if isinstance(model.head, nn.Sequential):
                        model.head[-1] = nn.Linear(
                            model.head[-1].in_features, num_classes)
                    else:
                        model.head = nn.Linear(
                            model.head.in_features, num_classes)
                return model
        except:
            pass

        model_name_lower = self.model_name.lower()

        # Load torchvision models
        if 'resnet' in model_name_lower:
            model = torchvision.models.resnet50(pretrained=True)
        elif 'vgg' in model_name_lower:
            model = torchvision.models.vgg16(pretrained=True)
        elif 'densenet' in model_name_lower:
            model = torchvision.models.densenet121(pretrained=True)
        elif 'mobilenet' in model_name_lower or 'mobilenetv2' in model_name_lower:
            model = torchvision.models.mobilenet_v2(pretrained=True)
        elif 'inception' in model_name_lower or 'inceptionv3' in model_name_lower:
            model = torchvision.models.inception_v3(pretrained=True, aux_logits=False)
        elif 'shufflenet' in model_name_lower or 'shufflenetv2' in model_name_lower:
            model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        num_classes = self._get_num_classes()

        # Adjust output layer based on model architecture
        if hasattr(model, 'fc'):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif hasattr(model, 'classifier'):
            if isinstance(model.classifier, nn.Sequential):
                model.classifier[-1] = nn.Linear(
                    model.classifier[-1].in_features, num_classes)
            else:
                model.classifier = nn.Linear(
                    model.classifier.in_features, num_classes)
        elif hasattr(model, 'head'):
            if isinstance(model.head, nn.Sequential):
                model.head[-1] = nn.Linear(
                    model.head[-1].in_features, num_classes)
            else:
                model.head = nn.Linear(
                    model.head.in_features, num_classes)

        return model

    def _get_num_classes(self) -> int:
        """Get number of classes for dataset."""

        dataset_map = {
            'cifar10': 10,
            'cifar100': 100,
            'imagenet': 1000,
            'stanford_cars': 196,
            'food101': 101,
            'mnist': 10,
            'inaturalist': 10000,
        }
        return dataset_map.get(self.dataset.lower(), 1000)

    def load_dataset(self) -> Tuple[DataLoader, DataLoader]:
        """Load classification dataset."""

        dataset_name = self.dataset.lower()

        if dataset_name == 'cifar10':
            return self._load_cifar10()
        elif dataset_name == 'cifar100':
            return self._load_cifar100()
        elif dataset_name == 'imagenet':
            return self._load_imagenet()
        elif dataset_name == 'stanford_cars':
            return self._load_stanford_cars()
        elif dataset_name == 'food101':
            return self._load_food101()
        elif dataset_name == 'mnist':
            return self._load_mnist()
        elif dataset_name == 'inaturalist':
            return self._load_inaturalist()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def _load_cifar10(self) -> Tuple[DataLoader, DataLoader]:
        """Load CIFAR-10 dataset."""

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])

        train_set = torchvision.datasets.CIFAR10(
            root=str(self.data_root), train=True, download=True, transform=transform_train
        )
        test_set = torchvision.datasets.CIFAR10(
            root=str(self.data_root), train=False, download=True, transform=transform_test
        )

        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True,
            num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )

        return train_loader, test_loader

    def _load_cifar100(self) -> Tuple[DataLoader, DataLoader]:
        """Load CIFAR-100 dataset."""

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761))
        ])

        train_set = torchvision.datasets.CIFAR100(
            root=str(self.data_root), train=True, download=True, transform=transform_train
        )
        test_set = torchvision.datasets.CIFAR100(
            root=str(self.data_root), train=False, download=True, transform=transform_test
        )

        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True,
            num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )

        return train_loader, test_loader

    def _load_imagenet(self) -> Tuple[DataLoader, DataLoader]:
        """Load ImageNet dataset."""

        from torchvision.datasets import ImageFolder

        train_dir = self.data_root / 'imagenet' / 'train'
        val_dir = self.data_root / 'imagenet' / 'val'

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

        train_set = ImageFolder(str(train_dir), transform=transform_train)
        test_set = ImageFolder(str(val_dir), transform=transform_test)

        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True,
            num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )

        return train_loader, test_loader

    def _load_stanford_cars(self) -> Tuple[DataLoader, DataLoader]:
        """Load Stanford Cars dataset."""

        from torchvision.datasets import ImageFolder

        # Assume dataset is in stanford_cars/train and stanford_cars/test
        train_dir = self.data_root / 'stanford_cars' / 'train'
        test_dir = self.data_root / 'stanford_cars' / 'test'

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

        train_set = ImageFolder(str(train_dir), transform=transform_train)
        test_set = ImageFolder(str(test_dir), transform=transform_test)

        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True,
            num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )

        return train_loader, test_loader

    def _load_food101(self) -> Tuple[DataLoader, DataLoader]:
        """Load Food-101 dataset."""

        from torchvision.datasets import ImageFolder

        train_dir = self.data_root / 'food101' / 'train'
        test_dir = self.data_root / 'food101' / 'test'

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

        train_set = ImageFolder(str(train_dir), transform=transform_train)
        test_set = ImageFolder(str(test_dir), transform=transform_test)

        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True,
            num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )

        return train_loader, test_loader

    def _load_mnist(self) -> Tuple[DataLoader, DataLoader]:
        """Load MNIST dataset."""

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_set = torchvision.datasets.MNIST(
            root=str(self.data_root), train=True, download=True, transform=transform_train
        )
        test_set = torchvision.datasets.MNIST(
            root=str(self.data_root), train=False, download=True, transform=transform_test
        )

        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True,
            num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )

        return train_loader, test_loader

    def _load_inaturalist(self) -> Tuple[DataLoader, DataLoader]:
        """Load iNaturalist dataset."""

        from torchvision.datasets import ImageFolder

        # iNaturalist dataset structure: inaturalist/train and inaturalist/val
        train_dir = self.data_root / 'inaturalist' / 'train'
        val_dir = self.data_root / 'inaturalist' / 'val'

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

        train_set = ImageFolder(str(train_dir), transform=transform_train)
        test_set = ImageFolder(str(val_dir), transform=transform_test)

        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True,
            num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )

        return train_loader, test_loader

    def evaluate(self, model: nn.Module, test_loader: DataLoader) -> float:
        """Evaluate classification accuracy."""

        model.eval()
        model.to(self.device)

        correct = 0
        total = 0
        batch_idx = 0

        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs, targets = batch['image'], batch['label']

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = model(inputs)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                elif isinstance(outputs, dict):
                    outputs = outputs['logits'] if 'logits' in outputs else outputs['pred']

                _, predicted = outputs.max(1)
                batch_correct = predicted.eq(targets).sum().item()
                batch_total = targets.size(0)
                total += batch_total
                correct += batch_correct

                # Log batch-level accuracy to TensorBoard
                if hasattr(self, 'writer') and self.writer is not None:
                    batch_accuracy = 100.0 * batch_correct / batch_total
                    self.writer.add_scalar('evaluation/batch_accuracy', batch_accuracy, batch_idx)
                    batch_idx += 1

        accuracy = 100.0 * correct / total
        return accuracy
