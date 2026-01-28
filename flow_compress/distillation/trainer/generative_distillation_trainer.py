"""
FAD trainer with Generative Distillation integration.
"""

import logging
from typing import Any, Callable, Dict, Optional

from flow_compress.distillation.losses.generative_distillation import (
    DiffusionDistillationLoss,
    GANDistillationLoss,
    GenerativeDistillationLoss,
    VAEDistillationLoss,
)
from flow_compress.distillation.models.student_wrapper import StudentWrapper
from flow_compress.distillation.models.teacher_wrapper import TeacherWrapper
from flow_compress.distillation.trainer.fad_trainer import FADTrainer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class GenerativeDistillationFADTrainer(FADTrainer):
    """
    FAD trainer integrated with Generative Distillation.
    """

    def __init__(
        self,
        teacher: TeacherWrapper,
        student: StudentWrapper,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        model_type: str = "vae",  # "vae", "gan", "diffusion"
        lambda_encoder: float = 0.1,
        lambda_decoder: float = 0.1,
        base_generative_loss_fn: Optional[Callable] = None,
        encoder_layer_prefix: str = "encoder",
        decoder_layer_prefix: str = "decoder",
        use_weights: bool = True,
        precompute_teacher_divs: bool = True,
    ):
        # Initialize base FAD trainer (without some parameters not needed for generative)
        super().__init__(
            teacher=teacher,
            student=student,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            temperature=1.0,  # Not used for generative models
            lambda_weight=0.1,
            use_weights=use_weights,
            precompute_teacher_divs=precompute_teacher_divs,
            use_selective_alignment=False,  # Not typically used for generative
        )

        self.model_type = model_type.lower()
        self.lambda_encoder = lambda_encoder
        self.lambda_decoder = lambda_decoder
        self.base_generative_loss_fn = base_generative_loss_fn
        self.encoder_layer_prefix = encoder_layer_prefix
        self.decoder_layer_prefix = decoder_layer_prefix

        # Initialize appropriate generative loss
        if self.model_type == "vae":
            self.generative_loss = VAEDistillationLoss(
                lambda_encoder=lambda_encoder,
                lambda_decoder=lambda_decoder,
            )
        elif self.model_type == "gan":
            self.generative_loss = GANDistillationLoss(
                lambda_encoder=lambda_encoder,
                lambda_decoder=lambda_decoder,
            )
        elif self.model_type == "diffusion":
            self.generative_loss = DiffusionDistillationLoss(
                lambda_encoder=lambda_encoder,
                lambda_decoder=lambda_decoder,
            )
        else:
            self.generative_loss = GenerativeDistillationLoss(
                lambda_encoder=lambda_encoder,
                lambda_decoder=lambda_decoder,
                model_type=model_type,
            )

    def _split_divergences_by_path(
        self,
        divergences: Dict[str, torch.Tensor],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Split divergences into encoder and decoder paths.
        """

        encoder_div = {}
        decoder_div = {}

        for layer_name, div in divergences.items():
            if self.encoder_layer_prefix in layer_name.lower():
                encoder_div[layer_name] = div
            elif self.decoder_layer_prefix in layer_name.lower():
                decoder_div[layer_name] = div
            else:
                # Default: assign to encoder if ambiguous
                encoder_div[layer_name] = div

        return {
            "encoder": encoder_div,
            "decoder": decoder_div,
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        One epoch of training with Generative Distillation + FAD.
        """

        self.student.train()

        metrics_agg = {
            "loss": 0.0,
            "L_generative": 0.0,
            "L_encoder_alignment": 0.0,
            "L_decoder_alignment": 0.0,
        }
        num_batches = 0

        for batch in self.train_loader:
            # Get batch (for generative models, format may vary)
            if isinstance(batch, (list, tuple)):
                x = batch[0]  # Input data
            else:
                x = batch
            x = x.to(self.device)

            # Forward teacher
            with torch.no_grad():
                t_output, t_div = self.teacher.forward_with_flows(x)

            # Forward student
            s_output, s_div = self.student.forward_with_flows(x)

            # Split divergences by path
            t_paths = self._split_divergences_by_path(t_div)
            s_paths = self._split_divergences_by_path(s_div)

            # Compute base generative loss if function provided
            base_gen_loss = None
            if self.base_generative_loss_fn is not None:
                base_gen_loss = self.base_generative_loss_fn(
                    s_output, x, t_output)
            elif self.model_type == "vae" and isinstance(
                self.generative_loss, VAEDistillationLoss
            ):
                # For VAE, we need mu and logvar - this is model-specific
                # User should provide base_generative_loss_fn
                base_gen_loss = torch.tensor(
                    0.0, device=self.device, requires_grad=True
                )

            # Compute generative distillation loss with FAD alignment
            l_total, info = self.generative_loss(
                teacher_encoder_div=t_paths["encoder"],
                student_encoder_div=s_paths["encoder"],
                teacher_decoder_div=t_paths["decoder"],
                student_decoder_div=s_paths["decoder"],
                base_generative_loss=base_gen_loss,
            )

            # Backward pass
            self.optimizer.zero_grad()
            l_total.backward()
            self.optimizer.step()

            # Aggregate metrics
            metrics_agg["loss"] += l_total.item()
            metrics_agg["L_generative"] += info["L_generative"].item()
            metrics_agg["L_encoder_alignment"] += info["L_encoder_alignment"].item()
            metrics_agg["L_decoder_alignment"] += info["L_decoder_alignment"].item()
            num_batches += 1

        # Average metrics
        for k in metrics_agg:
            metrics_agg[k] /= max(1, num_batches)

        return metrics_agg
