"""
FAD trainer with Contrastive Distillation integration.
"""

import logging
from typing import Any, Callable, Dict, Optional

from flow_compress.distillation.losses.contrastive_distillation import ContrastiveDistillationLoss
from flow_compress.distillation.models.student_wrapper import StudentWrapper
from flow_compress.distillation.models.teacher_wrapper import TeacherWrapper
from flow_compress.distillation.trainer.fad_trainer import FADTrainer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ContrastiveDistillationFADTrainer(FADTrainer):
    """
    FAD trainer integrated with Contrastive Distillation.
    """

    def __init__(
        self,
        teacher: TeacherWrapper,
        student: StudentWrapper,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        temperature: float = 0.07,
        lambda_fakd: float = 0.1,
        embedding_extractor: Optional[Callable] = None,
        use_weights: bool = True,
        precompute_teacher_divs: bool = True,
        use_selective_alignment: bool = False,
        selective_error_threshold: float = 0.1,
        selective_update_freq: int = 10,
    ):
        # Initialize base FAD trainer
        super().__init__(
            teacher=teacher,
            student=student,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            temperature=temperature,
            lambda_weight=lambda_fakd,
            use_weights=use_weights,
            precompute_teacher_divs=precompute_teacher_divs,
            use_selective_alignment=use_selective_alignment,
            selective_error_threshold=selective_error_threshold,
            selective_update_freq=selective_update_freq,
        )

        self.lambda_fakd = lambda_fakd
        self.embedding_extractor = embedding_extractor

        # Initialize Contrastive Distillation loss
        self.contrastive_loss = ContrastiveDistillationLoss(
            temperature=temperature,
            lambda_fakd=lambda_fakd,
            use_flow_alignment=True,
        )

    def _extract_embeddings(
        self, features: Dict[str, torch.Tensor], layer_name: Optional[str] = None
    ) -> torch.Tensor:
        """
        Extract embeddings from features.
        """

        if self.embedding_extractor is not None:
            # Use custom extractor
            if layer_name:
                return self.embedding_extractor(features[layer_name])
            else:
                # Use last layer
                last_layer = sorted(features.keys())[-1]
                return self.embedding_extractor(features[last_layer])

        # Default: use last layer with appropriate pooling
        if not layer_name:
            layer_name = sorted(features.keys())[-1]

        feat = features[layer_name]

        # Handle different feature shapes
        if len(feat.shape) == 4:
            # Convolutional features: (B, C, H, W) -> (B, C) via global average pooling
            embeddings = (
                torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
                .squeeze(-1)
                .squeeze(-1)
            )
        elif len(feat.shape) == 3:
            # Sequence features: (B, L, D) -> (B, D) via mean pooling
            embeddings = feat.mean(dim=1)
        elif len(feat.shape) == 2:
            # Already 2D: (B, D)
            embeddings = feat
        else:
            raise ValueError(f"Unsupported feature shape: {feat.shape}")

        return embeddings

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        One epoch of training with Contrastive Distillation + FAD.
        """

        self.student.train()
        if self.use_selective_alignment:
            self.batch_count = 0

        metrics_agg = {
            "loss": 0.0,
            "L_contrastive": 0.0,
            "L_fad_alignment": 0.0,
        }
        num_batches = 0

        for batch in self.train_loader:
            # Get batch
            if isinstance(batch, (list, tuple)):
                x, y = batch
            else:
                x = batch
                y = None
            x = x.to(self.device)

            # Forward teacher
            with torch.no_grad():
                t_logits, t_div = self.teacher.forward_with_flows(x)
                t_features = self.teacher.activation_hook.activations.copy()

            # Forward student
            s_logits, s_div = self.student.forward_with_flows(x)
            s_features = self.student.activation_hook.activations.copy()

            # Extract embeddings
            t_embeddings = self._extract_embeddings(t_features)
            s_embeddings = self._extract_embeddings(s_features)

            # Compute layer weights if enabled
            layer_weights = None
            if self.use_weights:
                from flow_compress.distillation.flows.alignment import compute_layer_weights

                layer_weights = compute_layer_weights(
                    t_div, self.lambda_weight)
                # Convert to float dict
                layer_weights = {
                    k: v.item() if isinstance(v, torch.Tensor) else v
                    for k, v in layer_weights.items()
                }

            # Compute contrastive distillation loss with FAD alignment
            l_total, info = self.contrastive_loss(
                teacher_embeddings=t_embeddings,
                student_embeddings=s_embeddings,
                teacher_div=t_div,
                student_div=s_div,
                layer_weights=layer_weights,
            )

            # Selective alignment update
            if self.use_selective_alignment:
                if self.batch_count % self.selective_update_freq == 0:
                    with torch.no_grad():
                        from flow_compress.distillation.flows.selective import selective_flow_alignment

                        self.selective_mask, _, stats = selective_flow_alignment(
                            teacher_div=t_div,
                            student_div=s_div,
                            error_threshold=self.selective_error_threshold,
                            use_critical_points=True,
                        )
            self.batch_count += 1

            # Backward pass
            self.optimizer.zero_grad()
            l_total.backward()
            self.optimizer.step()

            # Aggregate metrics
            metrics_agg["loss"] += l_total.item()
            metrics_agg["L_contrastive"] += info["L_contrastive"].item()
            metrics_agg["L_fad_alignment"] += info["L_fad_alignment"].item()
            num_batches += 1

        # Average metrics
        for k in metrics_agg:
            metrics_agg[k] /= max(1, num_batches)

        return metrics_agg
