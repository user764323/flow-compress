"""Models module."""

import os

from kaggle.api.kaggle_api_extended import KaggleApi
import timm
import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder


def get_transform_for_model_and_dataset(model_name, dataset_name):
    """Return the appropriate image transformation pipeline for the given model and dataset."""

    # Standard ImageNet mean and std for normalization
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Define the transform for the dataset
    if dataset_name in [
        "cifar10",
        "cifar100",
        "stanford_cars",
        "flowers102",
        "food101",
        "oxford_iiit_pet",
        "fashion_mnist",
        "fer2013",
        "inaturalist",
    ]:
        # Map of model names to their required input sizes
        input_size_map = {
            "efficientnet_b4": 380,
            "efficientnetv2_s": 384,
            "vit_base_patch16_224": 224,
            "mobilenetv3_large_100": 224,
            "densenet121": 224,
            "convnext_small": 224,
            "inception_v3": 299,
            "vgg19_bn": 224,
            "shufflenet_v2_x2_0": 224,
            "resnet50": 224,
        }
        # Get the required input size for the model, default to 224 if not found
        size = input_size_map.get(model_name.lower(), 224)

        # Special handling for grayscale datasets
        if dataset_name in ["fashion_mnist", "fer2013"]:
            # For grayscale datasets, convert to 3 channels to match model input
            transform = transforms.Compose(
                [
                    transforms.Resize((size, size)),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
                ]
            )
        else:
            # For color images, simple resize and normalization
            transform = transforms.Compose(
                [
                    transforms.Resize((size, size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
                ]
            )

    elif dataset_name == "imagenet":
        # Classic ImageNet transforms
        if model_name.lower() == "inception_v3":
            # InceptionV3 requires 299x299 input
            transform = transforms.Compose(
                [
                    transforms.Resize((299, 299)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
                ]
            )
        else:
            # Other models use standard 224x224 for ImageNet
            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
                ]
            )

    else:
        # Raise error for unsupported datasets
        raise ValueError(f"Unsupported dataset for transforms: {dataset_name}")

    return transform


def download_fer2013_kaggle(destination="./data/fer2013"):
    """Download and unzip the FER2013 dataset from Kaggle to the specified destination."""

    # Create the destination directory if it does not exist
    os.makedirs(destination, exist_ok=True)

    # Initialize Kaggle API and authenticate using local credentials
    api = KaggleApi()
    api.authenticate()

    # Download and unzip the FER2013 dataset files into the destination directory
    api.dataset_download_files(
        "msambare/fer2013", path=destination, unzip=True)


def load_model_and_dataset(model_path, model_name="resnet50", dataset_name="cifar10"):
    """Load a pretrained model checkpoint and the associated dataset with proper transforms.

    Args:
        model_path: str: Path to the model checkpoint or serialized model.
        model_name: str: Name of the model architecture.
        dataset_name: str: Name of the dataset to load.

    Returns:
        tuple: (model, num_classes, train_dataset, test_dataset)
            - model: Loaded PyTorch model ready for evaluation or training
            - num_classes: Number of classes in the dataset
            - train_dataset: Training dataset with appropriate transforms
            - test_dataset: Testing dataset with appropriate transforms

    Raises:
        ValueError: If the dataset name is invalid
        ValueError: If the model name is invalid
    """

    # Flag to determine if we're loading a checkpoint or a serialized model
    weights_only = True

    try:
        # Try to load the model as a checkpoint
        checkpoint = torch.load(model_path, map_location="cpu")
    except Exception as e:
        # If loading as a checkpoint fails, try loading it as a model
        weights_only = False
        model = torch.load(
            model_path, weights_only=weights_only, map_location="cpu")

    # Get image transformation pipeline for the model and dataset
    transform = get_transform_for_model_and_dataset(model_name, dataset_name)

    # Load the appropriate dataset based on dataset_name
    if dataset_name == "cifar10":
        num_classes = 10
        train_dataset = datasets.CIFAR10(
            root="./data/train", train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root="./data/test", train=False, download=True, transform=transform
        )
    elif dataset_name == "cifar100":
        num_classes = 100
        train_dataset = datasets.CIFAR100(
            root="./data/train", train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR100(
            root="./data/test", train=False, download=True, transform=transform
        )
    elif dataset_name == "flowers102":
        num_classes = 102
        train_dataset = datasets.Flowers102(
            root="./data/train", split="train", download=True, transform=transform
        )
        test_dataset = datasets.Flowers102(
            root="./data/test", split="test", download=True, transform=transform
        )
    elif dataset_name == "stanford_cars":
        num_classes = 196
        train_dataset = datasets.StanfordCars(
            root="./data/train", split="train", download=True, transform=transform
        )
        test_dataset = datasets.StanfordCars(
            root="./data/test", split="test", download=True, transform=transform
        )
    elif dataset_name == "imagenet":
        num_classes = 1000
        train_dataset = datasets.ImageNet(
            root="./data/train", split="train", download=True, transform=transform
        )
        test_dataset = datasets.ImageNet(
            root="./data/test", split="val", download=True, transform=transform
        )
    elif dataset_name == "inaturalist":
        num_classes = 8142
        train_dataset = datasets.INaturalist(
            "./data/train", version="2021_train", download=True, transform=transform
        )
        test_dataset = datasets.INaturalist(
            "./data/test", version="2021_valid", download=True, transform=transform
        )
    elif dataset_name == "food101":
        num_classes = 101
        train_dataset = datasets.Food101(
            "./data/train", split="train", download=True, transform=transform
        )
        test_dataset = datasets.Food101(
            "./data/test", split="test", download=True, transform=transform
        )
    elif dataset_name == "oxford_pets":
        num_classes = 37
        train_dataset = datasets.OxfordIIITPet(
            "./data/train", split="trainval", download=True, transform=transform
        )
        test_dataset = datasets.OxfordIIITPet(
            "./data/test", split="test", download=True, transform=transform
        )
    elif dataset_name == "fashion_mnist":
        num_classes = 10
        train_dataset = datasets.FashionMNIST(
            "./data/train", train=True, download=True, transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            "./data/test", train=False, download=True, transform=transform
        )
    elif dataset_name == "fer2013":
        num_classes = 7
        fer_path = "./data/fer2013"

        # Download FER2013 if not already present
        if not os.path.exists(os.path.join(fer_path, "train")):
            download_fer2013_kaggle(fer_path)

        train_dataset = ImageFolder(
            root=os.path.join(fer_path, "train"), transform=transform
        )
        test_dataset = ImageFolder(
            root=os.path.join(fer_path, "test"), transform=transform
        )
    else:
        raise ValueError("Invalid dataset_name.")

    # If we loaded a checkpoint, create and configure the model
    if weights_only:
        model_key = model_name.lower()
        if model_key == "resnet50":
            model = timm.create_model(
                "resnet50", pretrained=False, num_classes=num_classes
            )
            state_dict = checkpoint["model"]
            # Remove 'module.' prefix if present (from DataParallel)
            state_dict = {
                k.replace("module.", ""): v for k, v in state_dict.items()
            }
        elif model_key == "densenet121":
            model = timm.create_model(
                "densenet121", pretrained=False, num_classes=num_classes
            )
            state_dict = checkpoint["model"]
            state_dict = {
                k.replace("module.", ""): v for k, v in state_dict.items()
            }
        elif model_key == "efficientnet_b4":
            model = timm.create_model(
                "efficientnet_b4", pretrained=False, num_classes=num_classes
            )
            state_dict = checkpoint["model"]
            state_dict = {
                k.replace("module.", ""): v for k, v in state_dict.items()
            }
        elif model_key == "vit_base_patch16_224":
            model = timm.create_model(
                "vit_base_patch16_224", pretrained=False, num_classes=num_classes
            )
            state_dict = checkpoint["model"]
            state_dict = {
                k.replace("module.", ""): v for k, v in state_dict.items()
            }
        elif model_key == "mobilenetv3_large_100":
            model = timm.create_model(
                "mobilenetv3_large_100", pretrained=False, num_classes=num_classes
            )
            state_dict = checkpoint["model"]
            state_dict = {
                k.replace("module.", ""): v for k, v in state_dict.items()
            }
        elif model_key == "convnext_small":
            model = timm.create_model(
                "convnext_small", pretrained=False, num_classes=num_classes
            )
            state_dict = checkpoint["model"]
            state_dict = {
                k.replace("module.", ""): v for k, v in state_dict.items()
            }
        elif model_key == "inception_v3":
            model = timm.create_model(
                "inception_v3", pretrained=False, num_classes=num_classes
            )
            state_dict = checkpoint["model"]
            state_dict = {
                k.replace("module.", ""): v for k, v in state_dict.items()
            }
        elif model_key == "efficientnetv2_s":
            model = timm.create_model(
                "efficientnetv2_s", pretrained=False, num_classes=num_classes
            )
            state_dict = checkpoint["model"]
            state_dict = {
                k.replace("module.", ""): v for k, v in state_dict.items()
            }
        elif model_key == "vgg19_bn":
            model = timm.create_model(
                "vgg19_bn", pretrained=False, num_classes=num_classes
            )
            state_dict = checkpoint["model"]
            state_dict = {
                k.replace("module.", ""): v for k, v in state_dict.items()
            }
        elif model_key == "shufflenet_v2_x2_0":
            model = timm.create_model(
                "shufflenet_v2_x2_0", pretrained=False, num_classes=num_classes
            )
            state_dict = checkpoint["model"]
            state_dict = {
                k.replace("module.", ""): v for k, v in state_dict.items()
            }
        else:
            raise ValueError(f"Invalid model_name: {model_name}")

        # Load the state dict into the model
        model.load_state_dict(state_dict)

    return model, num_classes, train_dataset, test_dataset
