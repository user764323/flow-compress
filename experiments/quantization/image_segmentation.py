import logging
from typing import Any, Tuple

import numpy as np

from experiments.base_experiment import BaseExperiment, ExperimentResult
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

try:
    import timm
except ImportError:
    timm = None

logger = logging.getLogger(__name__)


class ImageSegmentationExperiment(BaseExperiment):
    """Experiment for image segmentation tasks."""

    def __init__(
        self,
        dataset: str,
        model_name: str,
        device: str = "cuda",
        data_root: str = "./data",
        output_dir: str = "./experiments/results",
        batch_size: int = 8,
    ):
        super().__init__(
            task="image_segmentation",
            dataset=dataset,
            model_name=model_name,
            device=device,
            data_root=data_root,
            output_dir=output_dir,
        )
        self.batch_size = batch_size

    def load_model(self) -> nn.Module:
        """Load segmentation model."""

        # Use torchvision segmentation models
        import torchvision.models.segmentation as seg_models

        model_name_lower = self.model_name.lower()
        num_classes = self._get_num_classes()

        # Helper function to get encoder name from model name
        def get_encoder_name():
            """Extract encoder name from model name."""
            if "efficientnet" in model_name_lower:
                if "b7" in model_name_lower:
                    return "efficientnet-b7"
                elif "b6" in model_name_lower:
                    return "efficientnet-b6"
                elif "b5" in model_name_lower:
                    return "efficientnet-b5"
                elif "b4" in model_name_lower:
                    return "efficientnet-b4"
                elif "b3" in model_name_lower:
                    return "efficientnet-b3"
                elif "b2" in model_name_lower:
                    return "efficientnet-b2"
                elif "b1" in model_name_lower:
                    return "efficientnet-b1"
                elif "b0" in model_name_lower:
                    return "efficientnet-b0"
                else:
                    return "efficientnet-b0"
            elif "resnet101" in model_name_lower:
                return "resnet101"
            elif "resnet50" in model_name_lower:
                return "resnet50"
            elif "resnet34" in model_name_lower:
                return "resnet34"
            elif "resnet18" in model_name_lower:
                return "resnet18"
            elif "densenet" in model_name_lower:
                if "169" in model_name_lower:
                    return "densenet169"
                elif "161" in model_name_lower:
                    return "densenet161"
                elif "121" in model_name_lower:
                    return "densenet121"
                else:
                    return "densenet121"
            elif "mobilenet" in model_name_lower:
                if "v3" in model_name_lower:
                    return "timm-mobilenetv3_large_100"
                else:
                    return "timm-mobilenetv3_large_100"
            elif "vgg" in model_name_lower:
                if "19" in model_name_lower:
                    return "vgg19"
                elif "16" in model_name_lower:
                    return "vgg16"
                elif "13" in model_name_lower:
                    return "vgg13"
                elif "11" in model_name_lower:
                    return "vgg11"
                else:
                    return "vgg16"
            elif "senet" in model_name_lower or "se" in model_name_lower:
                if "154" in model_name_lower:
                    return "senet154"
                else:
                    return "se_resnext50_32x4d"
            elif "xception" in model_name_lower:
                return "xception"
            elif "inception" in model_name_lower:
                return "inceptionresnetv2"
            else:
                return "resnet50"  # Default encoder

        # Try to load models from segmentation_models_pytorch first
        try:
            import segmentation_models_pytorch as smp
            encoder_name = get_encoder_name()

            if "unet" in model_name_lower or "u-net" in model_name_lower:
                model = smp.Unet(
                    encoder_name=encoder_name,
                    encoder_weights="imagenet",
                    classes=num_classes,
                    activation=None,
                )
            elif "linknet" in model_name_lower:
                model = smp.Linknet(
                    encoder_name=encoder_name,
                    encoder_weights="imagenet",
                    classes=num_classes,
                    activation=None,
                )
            elif "fpn" in model_name_lower and "deeplab" not in model_name_lower:
                model = smp.FPN(
                    encoder_name=encoder_name,
                    encoder_weights="imagenet",
                    classes=num_classes,
                    activation=None,
                )
            elif "pan" in model_name_lower and "deeplab" not in model_name_lower:
                model = smp.PAN(
                    encoder_name=encoder_name,
                    encoder_weights="imagenet",
                    classes=num_classes,
                    activation=None,
                )
            elif "pspnet" in model_name_lower or "psp" in model_name_lower:
                model = smp.PSPNet(
                    encoder_name=encoder_name,
                    encoder_weights="imagenet",
                    classes=num_classes,
                    activation=None,
                )
            elif "deeplabv3plus" in model_name_lower or "deeplabv3+" in model_name_lower:
                model = smp.DeepLabV3Plus(
                    encoder_name=encoder_name,
                    encoder_weights="imagenet",
                    classes=num_classes,
                    activation=None,
                )
            elif "deeplabv3" in model_name_lower:
                # Try smp first, fallback to torchvision
                try:
                    model = smp.DeepLabV3(
                        encoder_name=encoder_name,
                        encoder_weights="imagenet",
                        classes=num_classes,
                        activation=None,
                    )
                except:
                    # Fallback to torchvision
                    if "resnet50" in model_name_lower:
                        model = seg_models.deeplabv3_resnet50(pretrained=True)
                    elif "resnet101" in model_name_lower:
                        model = seg_models.deeplabv3_resnet101(pretrained=True)
                    else:
                        model = seg_models.deeplabv3_resnet50(pretrained=True)
            elif "manet" in model_name_lower:
                model = smp.MAnet(
                    encoder_name=encoder_name,
                    encoder_weights="imagenet",
                    classes=num_classes,
                    activation=None,
                )
            else:
                # If no smp model matched, continue to torchvision models
                model = None
        except ImportError:
            smp = None
            model = None

        # If smp model was not created, try torchvision models
        if model is None:
            if "deeplabv3" in model_name_lower and smp is None:
                if "resnet50" in model_name_lower:
                    model = seg_models.deeplabv3_resnet50(pretrained=True)
                elif "resnet101" in model_name_lower:
                    model = seg_models.deeplabv3_resnet101(pretrained=True)
                else:
                    model = seg_models.deeplabv3_resnet50(pretrained=True)
            elif "fcn" in model_name_lower:
                if "resnet50" in model_name_lower:
                    model = seg_models.fcn_resnet50(pretrained=True)
                elif "resnet101" in model_name_lower:
                    model = seg_models.fcn_resnet101(pretrained=True)
                else:
                    model = seg_models.fcn_resnet50(pretrained=True)
            elif "lraspp" in model_name_lower:
                model = seg_models.lraspp_mobilenet_v3_large(pretrained=True)
            elif "yolact" in model_name_lower:
                # YOLACT support - would need custom implementation or external library
                logger.info(
                    "Warning: YOLACT requires custom implementation. Using DeepLabV3 as fallback."
                )
                model = seg_models.deeplabv3_resnet50(pretrained=True)
            elif "segformer" in model_name_lower:
                # SegFormer support - try timm or transformers
                try:
                    model = timm.create_model(
                        "segformer_b0", pretrained=True, num_classes=num_classes
                    )
                except:
                    try:
                        from transformers import SegformerForSemanticSegmentation

                        model = SegformerForSemanticSegmentation.from_pretrained(
                            "nvidia/segformer-b0-finetuned-ade-640-640"
                        )
                    except:
                        logger.info(
                            "Warning: SegFormer requires timm or transformers. Using DeepLabV3 as fallback."
                        )
                        model = seg_models.deeplabv3_resnet50(pretrained=True)
            elif "swin" in model_name_lower and "transformer" in model_name_lower:
                # Swin Transformer for segmentation - try timm
                try:
                    model = timm.create_model(
                        "swin_base_patch4_window7_224", pretrained=True, num_classes=num_classes
                    )
                except:
                    logger.info(
                        "Warning: Swin Transformer segmentation requires timm. Using DeepLabV3 as fallback."
                    )
                    model = seg_models.deeplabv3_resnet50(pretrained=True)
            else:
                # Default to DeepLabV3 with ResNet50
                model = seg_models.deeplabv3_resnet50(pretrained=True)

        # Adjust number of classes for torchvision models
        if hasattr(model, "classifier"):
            if isinstance(model.classifier, nn.Sequential):
                # For DeepLabV3, FCN
                model.classifier[-1] = nn.Conv2d(
                    model.classifier[-1].in_channels, num_classes, kernel_size=1
                )
            else:
                model.classifier = nn.Conv2d(
                    model.classifier.in_channels, num_classes, kernel_size=1
                )
        elif hasattr(model, "aux_classifier") and model.aux_classifier is not None:
            # For models with auxiliary classifier
            if isinstance(model.aux_classifier, nn.Sequential):
                model.aux_classifier[-1] = nn.Conv2d(
                    model.aux_classifier[-1].in_channels, num_classes, kernel_size=1
                )
            else:
                model.aux_classifier = nn.Conv2d(
                    model.aux_classifier.in_channels, num_classes, kernel_size=1
                )

        return model

    def _get_num_classes(self) -> int:
        """Get number of classes for dataset."""

        dataset_map = {
            "cityscapes": 19,
            "pascal_voc": 21,
            "coco": 21,
            "ade20k": 150,
        }
        return dataset_map.get(self.dataset.lower(), 21)

    def load_dataset(self) -> Tuple[DataLoader, DataLoader]:
        """Load segmentation dataset."""

        dataset_name = self.dataset.lower()

        if dataset_name == "cityscapes":
            return self._load_cityscapes()
        elif dataset_name == "pascal_voc":
            return self._load_pascal_voc_seg()
        elif dataset_name == "coco":
            return self._load_coco_seg()
        elif dataset_name == "ade20k":
            return self._load_ade20k()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def _load_cityscapes(self) -> Tuple[DataLoader, DataLoader]:
        """Load Cityscapes dataset."""

        try:
            from torchvision.datasets import Cityscapes

            # Define transforms for images and targets
            image_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            target_transform = transforms.Compose([
                transforms.ToTensor()
            ])

            # Load Cityscapes dataset
            train_set = Cityscapes(
                root=str(self.data_root / "cityscapes"),
                split="train",
                mode="fine",
                target_type="semantic",
                transform=image_transform,
                target_transform=target_transform,
                download=False  # Set to True if dataset needs to be downloaded
            )

            val_set = Cityscapes(
                root=str(self.data_root / "cityscapes"),
                split="val",
                mode="fine",
                target_type="semantic",
                transform=image_transform,
                target_transform=target_transform,
                download=False
            )

            train_loader = DataLoader(
                train_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                drop_last=True
            )

            test_loader = DataLoader(
                val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )

            return train_loader, test_loader

        except FileNotFoundError as e:
            error_msg = (
                f"Cityscapes dataset not found at {self.data_root / 'cityscapes'}. "
                f"Please download the dataset from https://www.cityscapes-dataset.com/ "
                f"and extract it to {self.data_root / 'cityscapes'}. "
                f"Error: {e}"
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg) from e
        except Exception as e:
            error_msg = (
                f"Failed to load Cityscapes dataset: {e}. "
                f"Please ensure the dataset is properly downloaded and available at "
                f"{self.data_root / 'cityscapes'}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _load_pascal_voc_seg(self) -> Tuple[DataLoader, DataLoader]:
        """Load Pascal VOC segmentation dataset."""

        try:
            from torchvision.datasets import VOCSegmentation

            # Define transforms for images and targets
            image_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            target_transform = transforms.Compose([
                transforms.ToTensor()
            ])

            # Load Pascal VOC segmentation dataset
            train_set = VOCSegmentation(
                root=str(self.data_root / "VOC"),
                year="2012",
                image_set="train",
                download=False,  # Set to True if dataset needs to be downloaded
                transform=image_transform,
                target_transform=target_transform
            )

            val_set = VOCSegmentation(
                root=str(self.data_root / "VOC"),
                year="2012",
                image_set="val",
                download=False,
                transform=image_transform,
                target_transform=target_transform
            )

            train_loader = DataLoader(
                train_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                drop_last=True
            )

            test_loader = DataLoader(
                val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )

            return train_loader, test_loader

        except FileNotFoundError as e:
            error_msg = (
                f"Pascal VOC dataset not found at {self.data_root / 'VOC'}. "
                f"Please download Pascal VOC 2012 from "
                f"http://host.robots.ox.ac.uk/pascal/VOC/voc2012/ "
                f"and extract it to {self.data_root / 'VOC'}. "
                f"Error: {e}"
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg) from e
        except Exception as e:
            error_msg = (
                f"Failed to load Pascal VOC dataset: {e}. "
                f"Please ensure the dataset is properly downloaded and available at "
                f"{self.data_root / 'VOC'}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _load_coco_seg(self) -> Tuple[DataLoader, DataLoader]:
        """Load COCO segmentation dataset."""

        try:
            from torchvision.datasets import CocoDetection
            from pycocotools.coco import COCO
            from PIL import Image
            import os

            class CocoSegmentationDataset(torch.utils.data.Dataset):
                """Custom dataset wrapper for COCO segmentation."""

                def __init__(self, root, annFile, transform=None, target_transform=None):
                    self.root = root
                    self.coco = COCO(annFile)
                    self.ids = list(self.coco.imgs.keys())
                    self.transform = transform
                    self.target_transform = target_transform

                def __getitem__(self, index):
                    coco = self.coco
                    img_id = self.ids[index]
                    ann_ids = coco.getAnnIds(imgIds=img_id)
                    target = coco.loadAnns(ann_ids)

                    path = coco.loadImgs(img_id)[0]['file_name']
                    img = Image.open(os.path.join(self.root, path)).convert('RGB')

                    # Create semantic segmentation mask
                    mask = np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
                    for ann in target:
                        if 'segmentation' in ann:
                            if isinstance(ann['segmentation'], list):
                                for seg in ann['segmentation']:
                                    if isinstance(seg, list) and len(seg) > 0:
                                        # Convert polygon to mask
                                        from pycocotools import mask as mask_util
                                        rle = mask_util.frPyObjects([seg], img.size[1], img.size[0])
                                        m = mask_util.decode(rle)
                                        if m.shape[2] > 0:
                                            mask = np.maximum(mask, m[:, :, 0] * ann['category_id'])
                            elif isinstance(ann['segmentation'], dict):
                                # RLE format
                                from pycocotools import mask as mask_util
                                m = mask_util.decode(ann['segmentation'])
                                if len(m.shape) == 3:
                                    m = m[:, :, 0]
                                mask = np.maximum(mask, m * ann['category_id'])

                    mask = Image.fromarray(mask, mode='L')

                    if self.transform is not None:
                        img = self.transform(img)
                    if self.target_transform is not None:
                        mask = self.target_transform(mask)
                    else:
                        mask = torch.from_numpy(np.array(mask)).long()

                    return img, mask

                def __len__(self):
                    return len(self.ids)

            # Define transforms
            image_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            coco_root = self.data_root / "coco"
            train_ann_file = coco_root / "annotations" / "instances_train2017.json"
            val_ann_file = coco_root / "annotations" / "instances_val2017.json"
            train_img_dir = coco_root / "train2017"
            val_img_dir = coco_root / "val2017"

            train_set = CocoSegmentationDataset(
                root=str(train_img_dir),
                annFile=str(train_ann_file),
                transform=image_transform,
                target_transform=None
            )

            val_set = CocoSegmentationDataset(
                root=str(val_img_dir),
                annFile=str(val_ann_file),
                transform=image_transform,
                target_transform=None
            )

            train_loader = DataLoader(
                train_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                drop_last=True
            )

            test_loader = DataLoader(
                val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )

            return train_loader, test_loader

        except FileNotFoundError as e:
            error_msg = (
                f"COCO dataset not found at {self.data_root / 'coco'}. "
                f"Please download COCO dataset from https://cocodataset.org/ "
                f"and extract it to {self.data_root / 'coco'}. "
                f"Expected structure: coco/train2017/, coco/val2017/, "
                f"coco/annotations/instances_train2017.json, coco/annotations/instances_val2017.json. "
                f"Error: {e}"
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg) from e
        except ImportError as e:
            error_msg = (
                f"Required library not found for COCO dataset: {e}. "
                f"Please install pycocotools: pip install pycocotools"
            )
            logger.error(error_msg)
            raise ImportError(error_msg) from e
        except Exception as e:
            error_msg = (
                f"Failed to load COCO dataset: {e}. "
                f"Please ensure the dataset is properly downloaded and available at "
                f"{self.data_root / 'coco'}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _load_ade20k(self) -> Tuple[DataLoader, DataLoader]:
        """Load ADE20K dataset."""

        try:
            from PIL import Image
            import os
            from pathlib import Path

            class ADE20KDataset(torch.utils.data.Dataset):
                """Custom dataset for ADE20K semantic segmentation."""

                def __init__(self, root, split="training", transform=None, target_transform=None):
                    self.root = Path(root)
                    self.split = split
                    self.transform = transform
                    self.target_transform = target_transform

                    # ADE20K structure: images/training/ and annotations/training/
                    img_dir = self.root / "images" / split
                    ann_dir = self.root / "annotations" / split

                    if not img_dir.exists() or not ann_dir.exists():
                        raise FileNotFoundError(
                            f"ADE20K dataset not found at {self.root}. "
                            f"Expected directories: {img_dir} and {ann_dir}"
                        )

                    # Get all image files
                    self.images = sorted([f for f in img_dir.glob("*.jpg")])
                    self.annotations = sorted([f for f in ann_dir.glob("*.png")])

                    # Match images with annotations
                    self.samples = []
                    for img_path in self.images:
                        ann_path = ann_dir / (img_path.stem + ".png")
                        if ann_path.exists():
                            self.samples.append((img_path, ann_path))

                    if len(self.samples) == 0:
                        raise ValueError(f"No matching image-annotation pairs found in {self.root}")

                def __getitem__(self, index):
                    img_path, ann_path = self.samples[index]

                    # Load image
                    img = Image.open(img_path).convert('RGB')

                    # Load annotation (semantic segmentation mask)
                    mask = Image.open(ann_path).convert('L')

                    # Apply transforms
                    if self.transform is not None:
                        img = self.transform(img)
                    if self.target_transform is not None:
                        mask = self.target_transform(mask)
                    else:
                        # Convert PIL to tensor and ensure it's long type for class indices
                        mask = torch.from_numpy(np.array(mask)).long()
                        # ADE20K uses 0-150 for classes, but some pixels might be 255 (void)
                        mask = torch.clamp(mask, 0, 150)

                    return img, mask

                def __len__(self):
                    return len(self.samples)

            # Define transforms
            image_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            ade20k_root = self.data_root / "ADE20K"

            train_set = ADE20KDataset(
                root=str(ade20k_root),
                split="training",
                transform=image_transform,
                target_transform=None
            )

            val_set = ADE20KDataset(
                root=str(ade20k_root),
                split="validation",
                transform=image_transform,
                target_transform=None
            )

            train_loader = DataLoader(
                train_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                drop_last=True
            )

            test_loader = DataLoader(
                val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )

            return train_loader, test_loader

        except FileNotFoundError as e:
            error_msg = (
                f"ADE20K dataset not found at {self.data_root / 'ADE20K'}. "
                f"Please download ADE20K dataset from "
                f"https://groups.csail.mit.edu/vision/datasets/ADE20K/ "
                f"and extract it to {self.data_root / 'ADE20K'}. "
                f"Expected structure: ADE20K/images/training/, ADE20K/images/validation/, "
                f"ADE20K/annotations/training/, ADE20K/annotations/validation/. "
                f"Error: {e}"
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg) from e
        except Exception as e:
            error_msg = (
                f"Failed to load ADE20K dataset: {e}. "
                f"Please ensure the dataset is properly downloaded and available at "
                f"{self.data_root / 'ADE20K'}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def evaluate(self, model: nn.Module, test_loader: DataLoader) -> float:
        """Evaluate segmentation mIoU."""

        model.eval()
        model.to(self.device)

        num_classes = self._get_num_classes()
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        batch_idx = 0

        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (tuple, list)):
                    images, targets = batch[0], batch[1]
                else:
                    images, targets = batch["image"], batch["mask"]

                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = model(images)
                if isinstance(outputs, dict):
                    outputs = outputs["out"]
                elif isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]

                # Resize outputs to match targets
                if outputs.shape[-2:] != targets.shape[-2:]:
                    outputs = F.interpolate(
                        outputs,
                        size=targets.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                preds = outputs.argmax(dim=1).cpu().numpy()
                targets_np = targets.cpu().numpy()

                # Update confusion matrix
                for pred, target in zip(preds, targets_np):
                    confusion_matrix += self._fast_hist(
                        pred.flatten(), target.flatten(), num_classes
                    )

                # Log batch-level pixel accuracy to TensorBoard
                if hasattr(self, 'writer') and self.writer is not None:
                    batch_correct = (preds == targets_np).sum()
                    batch_total = preds.size
                    batch_pixel_acc = 100.0 * batch_correct / batch_total if batch_total > 0 else 0.0
                    self.writer.add_scalar('evaluation/batch_pixel_accuracy', batch_pixel_acc, batch_idx)
                    batch_idx += 1

        # Calculate mIoU
        iou_per_class = np.diag(confusion_matrix) / (
            confusion_matrix.sum(axis=1)
            + confusion_matrix.sum(axis=0)
            - np.diag(confusion_matrix)
        )
        iou_per_class = np.nan_to_num(iou_per_class)
        miou = np.mean(iou_per_class) * 100.0

        return miou

    @staticmethod
    def _fast_hist(
        pred: np.ndarray, target: np.ndarray, num_classes: int
    ) -> np.ndarray:
        """Compute confusion matrix."""

        mask = (target >= 0) & (target < num_classes)
        hist = np.bincount(
            num_classes * target[mask].astype(int) + pred[mask].astype(int),
            minlength=num_classes**2,
        ).reshape(num_classes, num_classes)
        return hist
