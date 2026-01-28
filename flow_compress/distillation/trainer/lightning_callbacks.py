"""
PyTorch Lightning callbacks for FAD training.
"""

import logging
import os
from typing import Any, Dict, Optional

import torch

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback
    from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

    PL_AVAILABLE = True

except ImportError:
    PL_AVAILABLE = False
    logging.info(
        "Warning: pytorch-lightning not available. Install with: pip install pytorch-lightning"
    )


if PL_AVAILABLE:

    class CurriculumLearningScheduler(Callback):
        """
        Curriculum Learning Scheduler callback for PyTorch Lightning.
        """

        def __init__(
            self,
            num_stages: int = 4,
            gamma_0: float = 1.0,
            lambda_weight: float = 0.1,
            fine_tune_epochs: int = 5,
            checkpoint_dir: str = "./checkpoints",
            resume_from_checkpoint: Optional[str] = None,
            monitor_metric: str = "val_acc",
            mode: str = "max",
        ):
            super().__init__()
            self.num_stages = num_stages
            self.gamma_0 = gamma_0
            self.lambda_weight = lambda_weight
            self.fine_tune_epochs = fine_tune_epochs
            self.checkpoint_dir = checkpoint_dir
            self.resume_from_checkpoint = resume_from_checkpoint
            self.monitor_metric = monitor_metric
            self.mode = mode

            # State tracking
            self.current_stage = 1
            self.stage_metrics = []
            self.fine_tuning = False

            # Create checkpoint directory
            os.makedirs(checkpoint_dir, exist_ok=True)

            logging.info(
                f"Initialized CurriculumLearningScheduler with {num_stages} stages, "
                f"checkpoint_dir={checkpoint_dir}"
            )

        def on_train_start(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
        ) -> None:
            """Called when training starts."""

            # Resume from checkpoint if provided
            if self.resume_from_checkpoint and os.path.exists(
                self.resume_from_checkpoint
            ):
                logging.info(
                    f"Resuming from checkpoint: {self.resume_from_checkpoint}")
                checkpoint = torch.load(self.resume_from_checkpoint)

                # Restore state
                if "curriculum_state" in checkpoint:
                    state = checkpoint["curriculum_state"]
                    self.current_stage = state.get("current_stage", 1)
                    self.stage_metrics = state.get("stage_metrics", [])
                    self.fine_tuning = state.get("fine_tuning", False)
                    logging.info(
                        f"Resumed: stage={self.current_stage}, "
                        f"fine_tuning={self.fine_tuning}"
                    )

        def on_train_epoch_start(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
        ) -> None:
            """Called at the start of each training epoch."""

            # Update curriculum stage if needed
            if hasattr(pl_module, "set_curriculum_stage"):
                pl_module.set_curriculum_stage(
                    self.current_stage, self.num_stages)

            # Compute stage alignment weight
            gamma_k = self.gamma_0 * (self.current_stage / self.num_stages)
            if hasattr(pl_module, "set_alignment_weight"):
                pl_module.set_alignment_weight(gamma_k)

            logging.info(
                f"Epoch {trainer.current_epoch}: "
                f"Stage {self.current_stage}/{self.num_stages}, "
                f"γ_k={gamma_k:.4f}"
            )

        def on_train_epoch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
        ) -> None:
            """Called at the end of each training epoch."""

            # Check if we should advance to next stage
            # Typically, we advance after a certain number of epochs per stage
            epochs_per_stage = (
                trainer.max_epochs // self.num_stages if trainer.max_epochs else 10
            )

            if (trainer.current_epoch + 1) % epochs_per_stage == 0:
                if self.current_stage < self.num_stages:
                    self.current_stage += 1
                    logging.info(
                        f"Advanced to curriculum stage {self.current_stage}/{self.num_stages}"
                    )
                elif not self.fine_tuning:
                    # Start fine-tuning phase
                    self.fine_tuning = True
                    logging.info(
                        f"Starting fine-tuning phase for {self.fine_tune_epochs} epochs"
                    )
                    if hasattr(pl_module, "set_fine_tuning"):
                        pl_module.set_fine_tuning(True)

        def on_validation_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
        ) -> None:
            """Called at the end of validation."""

            # Save checkpoint with curriculum state
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"curriculum_stage_{self.current_stage}_epoch_{trainer.current_epoch}.ckpt",
            )

            # Save curriculum state
            curriculum_state = {
                "current_stage": self.current_stage,
                "stage_metrics": self.stage_metrics,
                "fine_tuning": self.fine_tuning,
                "num_stages": self.num_stages,
                "gamma_0": self.gamma_0,
                "lambda_weight": self.lambda_weight,
            }

            # Use PyTorch Lightning's checkpoint saving
            trainer.save_checkpoint(checkpoint_path)

            # Also save curriculum state separately
            state_path = checkpoint_path.replace(
                ".ckpt", "_curriculum_state.pt")
            torch.save(curriculum_state, state_path)

        def on_train_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
        ) -> None:
            """Called when training ends."""

            logging.info(
                f"Training completed. Final stage: {self.current_stage}/{self.num_stages}, "
                f"Fine-tuning: {self.fine_tuning}"
            )

            # Save final checkpoint
            final_checkpoint = os.path.join(
                self.checkpoint_dir, "final_curriculum_checkpoint.ckpt"
            )
            trainer.save_checkpoint(final_checkpoint)

            curriculum_state = {
                "current_stage": self.current_stage,
                "stage_metrics": self.stage_metrics,
                "fine_tuning": self.fine_tuning,
                "num_stages": self.num_stages,
                "gamma_0": self.gamma_0,
                "lambda_weight": self.lambda_weight,
            }
            state_path = final_checkpoint.replace(
                ".ckpt", "_curriculum_state.pt")
            torch.save(curriculum_state, state_path)

        def get_current_stage(self) -> int:
            """Get current curriculum stage."""

            return self.current_stage

        def is_fine_tuning(self) -> bool:
            """Check if in fine-tuning phase."""

            return self.fine_tuning

    class FADMetricsCallback(Callback):
        """
        Callback for logging FAD-specific metrics.
        """

        def __init__(self, log_every_n_steps: int = 100):
            super().__init__()
            self.log_every_n_steps = log_every_n_steps
            self.step_count = 0

        def on_train_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: Dict[str, Any],
            batch: Any,
            batch_idx: int,
        ) -> None:
            """Called at the end of each training batch."""
            self.step_count += 1

            if self.step_count % self.log_every_n_steps == 0:
                # Log FAD metrics if available
                if hasattr(pl_module, "get_fad_metrics"):
                    metrics = pl_module.get_fad_metrics()
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            trainer.logger.log_metrics(
                                {f"fad/{key}": value},
                                step=self.step_count,
                            )
                        elif isinstance(value, dict):
                            # Log nested metrics
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, (int, float)):
                                    trainer.logger.log_metrics(
                                        {f"fad/{key}/{sub_key}": sub_value},
                                        step=self.step_count,
                                    )

else:
    # Dummy classes when PyTorch Lightning is not available
    class CurriculumLearningScheduler:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "pytorch-lightning is required for CurriculumLearningScheduler"
            )

    class FADMetricsCallback:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "pytorch-lightning is required for FADMetricsCallback")
