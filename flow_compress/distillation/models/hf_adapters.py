"""
HuggingFace Transformers adapters for trajectory alignment mechanism.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

from flow_compress.distillation.flows.divergence import compute_layerwise_flow_divergence
from flow_compress.distillation.flows.hooks import ActivationHook
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import (
        ViTModel,
        ViTForImageClassification,
        ResNetModel,
        ResNetForImageClassification,
        CLIPModel,
        CLIPVisionModel,
        AutoModel,
        AutoProcessor,
    )
    from transformers.modeling_outputs import (
        BaseModelOutput,
        ImageClassifierOutput,
    )

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logging.info(
        "Warning: transformers library not available. Install with: pip install transformers>=4.35.0"
    )


class TrajectoryAlignmentAdapter(nn.Module):
    """
    Base adapter for trajectory alignment mechanism across heterogeneous architectures.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: List[str],
        adapter_type: str = "base",
    ):
        super().__init__()
        if not HF_AVAILABLE:
            raise ImportError(
                "transformers library is required for HuggingFace adapters"
            )

        self.model = model
        self.layer_names = layer_names
        self.adapter_type = adapter_type
        self.activation_hook = ActivationHook(self.model, layer_names)

    def forward_with_flows(
        self,
        x: torch.Tensor,
        return_dict: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with flow divergence computation.
        """

        self.activation_hook.clear()
        outputs = self._forward_model(x)
        logits = self._extract_logits(outputs)
        divergences = compute_layerwise_flow_divergence(
            self.activation_hook.activations,
            self.layer_names,
        )
        return logits, divergences

    def _forward_model(self, x: torch.Tensor):
        """Forward pass through the model."""
        return self.model(x)

    def _extract_logits(self, outputs) -> torch.Tensor:
        """Extract logits from model outputs."""

        if hasattr(outputs, "logits"):
            return outputs.logits
        elif isinstance(outputs, tuple):
            return outputs[0]
        else:
            return outputs

    def forward(self, x: torch.Tensor):
        """Standard forward pass."""
        outputs = self._forward_model(x)
        return self._extract_logits(outputs)


class ViTAdapter(TrajectoryAlignmentAdapter):
    """
    Adapter for Vision Transformers (ViT-B/16 to ViT-L/16).
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        num_labels: Optional[int] = None,
        pretrained: bool = True,
    ):
        if not HF_AVAILABLE:
            raise ImportError("transformers library is required")

        if num_labels is not None:
            model = ViTForImageClassification.from_pretrained(
                model_name if pretrained else None,
                num_labels=num_labels,
            )
        else:
            model = ViTModel.from_pretrained(
                model_name if pretrained else None,
            )

        # Extract layer names for ViT
        layer_names = []
        for i in range(len(model.vit.encoder.layer)):
            layer_names.append(f"vit.encoder.layer.{i}")

        super().__init__(model, layer_names, adapter_type="vit")
        self.model_name = model_name

    def _forward_model(self, x: torch.Tensor):
        """Forward pass for ViT."""
        return self.model(pixel_values=x)


class ResNetAdapter(TrajectoryAlignmentAdapter):
    """
    Adapter for ResNet models (ResNet-50 to ResNet-152).
    """

    def __init__(
        self,
        model_name: str = "microsoft/resnet-50",
        num_labels: Optional[int] = None,
        pretrained: bool = True,
    ):
        if not HF_AVAILABLE:
            raise ImportError("transformers library is required")

        if num_labels is not None:
            model = ResNetForImageClassification.from_pretrained(
                model_name if pretrained else None,
                num_labels=num_labels,
            )
        else:
            model = ResNetModel.from_pretrained(
                model_name if pretrained else None,
            )

        # Extract layer names for ResNet
        layer_names = []
        if hasattr(model, "resnet"):
            backbone = model.resnet
        else:
            backbone = model

        # ResNet stages
        if hasattr(backbone, "stage1"):
            for i in range(1, 5):
                stage = getattr(backbone, f"stage{i}", None)
                if stage is not None:
                    for j, block in enumerate(stage):
                        layer_names.append(f"resnet.stage{i}.{j}")
        else:
            # Fallback: use encoder layers if available
            for i in range(len(backbone.encoder.stages)):
                layer_names.append(f"encoder.stages.{i}")

        super().__init__(model, layer_names, adapter_type="resnet")
        self.model_name = model_name

    def _forward_model(self, x: torch.Tensor):
        """Forward pass for ResNet."""
        return self.model(pixel_values=x)


class CLIPViTAdapter(TrajectoryAlignmentAdapter):
    """
    Adapter for CLIP-ViT models.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        pretrained: bool = True,
    ):
        if not HF_AVAILABLE:
            raise ImportError("transformers library is required")

        model = CLIPModel.from_pretrained(
            model_name if pretrained else None,
        )

        # Extract layer names from CLIP vision encoder
        layer_names = []
        vision_model = model.vision_model
        for i in range(len(vision_model.encoder.layers)):
            layer_names.append(f"vision_model.encoder.layers.{i}")

        super().__init__(model, layer_names, adapter_type="clip")
        self.model_name = model_name
        self.vision_model = vision_model

    def _forward_model(self, x: torch.Tensor):
        """Forward pass for CLIP."""
        outputs = self.model.get_image_features(pixel_values=x)

        # Return a simple wrapper to maintain compatibility
        class CLIPOutput:
            def __init__(self, features):
                self.logits = features

        return CLIPOutput(outputs)

    def _extract_logits(self, outputs) -> torch.Tensor:
        """Extract features/logits from CLIP outputs."""
        if hasattr(outputs, "logits"):
            return outputs.logits
        return outputs


class LLaVAAdapter(TrajectoryAlignmentAdapter):
    """
    Adapter for LLaVA (Large Language and Vision Assistant) models.
    """

    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        pretrained: bool = True,
    ):
        if not HF_AVAILABLE:
            raise ImportError("transformers library is required")

        try:
            model = AutoModel.from_pretrained(
                model_name if pretrained else None,
                trust_remote_code=True,
            )
        except Exception as e:
            raise ValueError(f"Failed to load LLaVA model: {e}")

        # Extract layer names from LLaVA vision tower
        layer_names = []
        if hasattr(model, "vision_tower"):
            vision_tower = model.vision_tower
            if hasattr(vision_tower, "vision_model"):
                encoder = vision_tower.vision_model.encoder
                for i in range(len(encoder.layers)):
                    layer_names.append(f"vision_tower.vision_model.encoder.layers.{i}")

        super().__init__(model, layer_names, adapter_type="llava")
        self.model_name = model_name

    def _forward_model(self, x: torch.Tensor):
        """Forward pass for LLaVA."""

        if hasattr(self.model, "get_image_features"):
            outputs = self.model.get_image_features(pixel_values=x)
        else:
            # Fallback: try to get vision tower outputs
            vision_tower = self.model.vision_tower
            outputs = vision_tower(pixel_values=x)

        class LLaVAOutput:
            def __init__(self, features):
                self.logits = features

        return LLaVAOutput(outputs)

    def _extract_logits(self, outputs) -> torch.Tensor:
        """Extract features from LLaVA outputs."""
        if hasattr(outputs, "logits"):
            return outputs.logits
        return outputs


class ImageBindAdapter(TrajectoryAlignmentAdapter):
    """
    Adapter for ImageBind models.
    """

    def __init__(
        self,
        model_name: str = "facebook/imagebind-base",
        pretrained: bool = True,
    ):
        if not HF_AVAILABLE:
            raise ImportError("transformers library is required")

        try:
            model = AutoModel.from_pretrained(
                model_name if pretrained else None,
                trust_remote_code=True,
            )
        except Exception as e:
            raise ValueError(f"Failed to load ImageBind model: {e}")

        # Extract layer names from ImageBind vision encoder
        layer_names = []
        if hasattr(model, "vision_model"):
            vision_model = model.vision_model
            if hasattr(vision_model, "encoder"):
                encoder = vision_model.encoder
                for i in range(len(encoder.layers)):
                    layer_names.append(f"vision_model.encoder.layers.{i}")

        super().__init__(model, layer_names, adapter_type="imagebind")
        self.model_name = model_name

    def _forward_model(self, x: torch.Tensor):
        """Forward pass for ImageBind."""

        if hasattr(self.model, "get_image_features"):
            outputs = self.model.get_image_features(pixel_values=x)
        else:
            outputs = self.model(pixel_values=x)

        class ImageBindOutput:
            def __init__(self, features):
                self.logits = features

        return ImageBindOutput(outputs)

    def _extract_logits(self, outputs) -> torch.Tensor:
        """Extract features from ImageBind outputs."""
        if hasattr(outputs, "logits"):
            return outputs.logits
        return outputs


def create_hf_adapter(
    architecture: str,
    model_name: Optional[str] = None,
    num_labels: Optional[int] = None,
    pretrained: bool = True,
) -> TrajectoryAlignmentAdapter:
    """
    Factory function to create HuggingFace adapters.
    """

    architecture = architecture.lower()

    if architecture == "vit":
        if model_name is None:
            model_name = "google/vit-base-patch16-224"
        return ViTAdapter(model_name, num_labels, pretrained)

    elif architecture == "resnet":
        if model_name is None:
            model_name = "microsoft/resnet-50"
        return ResNetAdapter(model_name, num_labels, pretrained)

    elif architecture == "clip":
        if model_name is None:
            model_name = "openai/clip-vit-base-patch32"
        return CLIPViTAdapter(model_name, pretrained)

    elif architecture == "llava":
        if model_name is None:
            model_name = "llava-hf/llava-1.5-7b-hf"
        return LLaVAAdapter(model_name, pretrained)

    elif architecture == "imagebind":
        if model_name is None:
            model_name = "facebook/imagebind-base"
        return ImageBindAdapter(model_name, pretrained)

    else:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Supported: 'vit', 'resnet', 'clip', 'llava', 'imagebind'"
        )
