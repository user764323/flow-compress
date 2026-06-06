"""
Generative Distillation loss for VAEs, GANs, and diffusion models.
"""

from typing import Dict, Optional, Tuple

from flow_compress.distillation.flows.alignment import compute_dalign
import torch
import torch.nn as nn
import torch.nn.functional as F


class GenerativeDistillationLoss(nn.Module):
    """
    Generative Distillation loss with FAD integration for generative models.
    """

    def __init__(
        self,
        lambda_encoder: float = 0.1,
        lambda_decoder: float = 0.1,
        model_type: str = "vae",  # "vae", "gan", "diffusion"
    ):
        super().__init__()
        self.lambda_encoder = lambda_encoder
        self.lambda_decoder = lambda_decoder
        self.model_type = model_type.lower()

    def forward(
        self,
        teacher_encoder_div: Optional[Dict[str, torch.Tensor]] = None,
        student_encoder_div: Optional[Dict[str, torch.Tensor]] = None,
        teacher_decoder_div: Optional[Dict[str, torch.Tensor]] = None,
        student_decoder_div: Optional[Dict[str, torch.Tensor]] = None,
        base_generative_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute generative distillation loss with FAD alignment.
        """

        # Initialize base generative loss if not provided
        if base_generative_loss is None:
            base_generative_loss = torch.tensor(
                0.0,
                device=(
                    next(iter(teacher_encoder_div.values())).device
                    if teacher_encoder_div
                    else torch.device("cpu")
                ),
            )

        l_generative = base_generative_loss

        # Encoder path alignment
        l_encoder_alignment = torch.tensor(0.0, device=l_generative.device)
        if teacher_encoder_div is not None and student_encoder_div is not None:
            l_encoder_alignment = compute_dalign(
                teacher_encoder_div, student_encoder_div
            )

        # Decoder path alignment
        l_decoder_alignment = torch.tensor(0.0, device=l_generative.device)
        if teacher_decoder_div is not None and student_decoder_div is not None:
            l_decoder_alignment = compute_dalign(
                teacher_decoder_div, student_decoder_div
            )

        # Total loss: L_generative + λ_enc · L_encoder_alignment + λ_dec · L_decoder_alignment
        l_total = (
            l_generative
            + self.lambda_encoder * l_encoder_alignment
            + self.lambda_decoder * l_decoder_alignment
        )

        info = {
            "L_generative": l_generative.detach(),
            "L_encoder_alignment": l_encoder_alignment.detach(),
            "L_decoder_alignment": l_decoder_alignment.detach(),
            "L_total": l_total.detach(),
        }

        return l_total, info


class VAEDistillationLoss(GenerativeDistillationLoss):
    """
    Specialized loss for VAE distillation with FAD.
    """

    def __init__(
        self,
        lambda_encoder: float = 0.1,
        lambda_decoder: float = 0.1,
        beta_vae: float = 1.0,
    ):
        super().__init__(lambda_encoder, lambda_decoder, model_type="vae")
        self.beta_vae = beta_vae

    def vae_loss(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute VAE loss (reconstruction + KL divergence).
        """

        # Reconstruction loss (MSE or BCE)
        recon_loss = F.mse_loss(recon_x, x, reduction="sum") / x.size(0)

        # KL divergence: -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.sum(1 + logvar -
                                   mu.pow(2) - logvar.exp()) / x.size(0)

        return recon_loss + self.beta_vae * kl_loss


class GANDistillationLoss(GenerativeDistillationLoss):
    """
    Specialized loss for GAN distillation with FAD.
    """

    def __init__(
        self,
        lambda_encoder: float = 0.1,
        lambda_decoder: float = 0.1,
        use_wgan: bool = False,
    ):
        super().__init__(lambda_encoder, lambda_decoder, model_type="gan")
        self.use_wgan = use_wgan

    def gan_loss(
        self,
        real_score: torch.Tensor,
        fake_score: torch.Tensor,
        is_generator: bool = False,
    ) -> torch.Tensor:
        """
        Compute GAN adversarial loss.
        """

        if self.use_wgan:
            # Wasserstein GAN loss
            if is_generator:
                return -fake_score.mean()
            else:
                return fake_score.mean() - real_score.mean()
        else:
            # Standard GAN loss (BCE)
            if is_generator:
                return F.binary_cross_entropy_with_logits(
                    fake_score, torch.ones_like(fake_score)
                )
            else:
                real_loss = F.binary_cross_entropy_with_logits(
                    real_score, torch.ones_like(real_score)
                )
                fake_loss = F.binary_cross_entropy_with_logits(
                    fake_score, torch.zeros_like(fake_score)
                )
                return (real_loss + fake_loss) / 2


class DiffusionDistillationLoss(GenerativeDistillationLoss):
    """
    Specialized loss for diffusion model distillation with FAD.
    """

    def __init__(
        self,
        lambda_encoder: float = 0.1,
        lambda_decoder: float = 0.1,
        noise_schedule: Optional[torch.Tensor] = None,
    ):
        super().__init__(lambda_encoder, lambda_decoder, model_type="diffusion")
        self.noise_schedule = noise_schedule

    def diffusion_loss(
        self,
        predicted_noise: torch.Tensor,
        target_noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute diffusion model loss (noise prediction).
        """

        return F.mse_loss(predicted_noise, target_noise)
