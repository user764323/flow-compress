"""
Base configuration for FAD (Flow-Aligned Distillation) training.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelConfig:
    """Configuration for teacher and student models."""

    # Teacher model
    teacher_name: str = ""
    teacher_weights: Optional[str] = "IMAGENET1K_V2"
    teacher_pretrained: bool = True

    # Student model
    student_name: str = ""
    student_pretrained: bool = False
    student_num_classes: int = 10

    # Layer selection
    teacher_layer_filter: str = "layer"
    exclude_bn: bool = True
    student_layer_filter: str = "layer"


@dataclass
class DatasetConfig:
    """Configuration for dataset and data loading."""

    name: str = ""
    data_dir: str = "./data"
    train_split: float = 0.9
    val_split: float = 0.1

    # Image transforms
    image_size: int = 224
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    # DataLoader
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""

    type: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # SGD specific
    momentum: float = 0.9
    nesterov: bool = False

    # Adam/AdamW specific
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduler."""

    type: Optional[str] = None
    # CosineAnnealingLR
    T_max: int = 100
    eta_min: float = 1e-6

    # StepLR
    step_size: int = 30
    gamma: float = 0.1

    # ReduceLROnPlateau
    mode: str = "min"
    factor: float = 0.1
    patience: int = 10
    threshold: float = 1e-4

    # Warmup
    warmup_steps: int = 0
    warmup_lr: float = 1e-6


@dataclass
class FADTrainerConfig:
    """Configuration for FAD trainer."""

    # Basic training
    num_epochs: int = 100
    device: str = "cuda"  # cuda, cpu, or auto

    # FAD specific parameters
    temperature: float = 4.0  # Temperature for logit-KD loss
    lambda_weight: float = 0.1  # Weighting parameter λ
    alpha_flow: float = 1.0

    # Algorithm options
    use_weights: bool = True
    precompute_teacher_divs: bool = True

    use_selective_alignment: bool = False
    selective_error_threshold: float = 0.1
    selective_update_freq: int = 10  # Update frequency in batches

    # FAD Loss options
    use_integrated_fad: bool = True
    num_integration_points: int = 100  # For numerical integration


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""

    log_dir: str = "./logs"
    use_tensorboard: bool = True
    tensorboard_log_dir: str = "./runs"

    use_wandb: bool = False
    wandb_project: str = "fad-training"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None

    log_freq: int = 10  # Log every N batches
    print_freq: int = 50  # Print every N batches


@dataclass
class CheckpointConfig:
    """Configuration for model checkpointing."""

    save_dir: str = "./checkpoints"
    save_freq: int = 10  # Save every N epochs
    save_best: bool = True  # Save best model based on validation metric
    save_last: bool = True  # Save last model

    resume_from: Optional[str] = None  # Path to checkpoint to resume from
    resume_optimizer: bool = True  # Resume optimizer state
    resume_scheduler: bool = True  # Resume scheduler state
    resume_epoch: bool = True  # Resume epoch counter

    metric_to_monitor: str = "val_acc"  # Metric for best model selection
    mode: str = "max"  # max or min


@dataclass
class ExperimentConfig:
    """Main configuration class combining all sub-configs."""

    # Experiment metadata
    experiment_name: str = "fad_experiment"
    seed: int = 42
    comment: Optional[str] = None

    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    trainer: FADTrainerConfig = field(default_factory=FADTrainerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "comment": self.comment,
            "model": {
                "teacher_name": self.model.teacher_name,
                "teacher_weights": self.model.teacher_weights,
                "teacher_pretrained": self.model.teacher_pretrained,
                "student_name": self.model.student_name,
                "student_pretrained": self.model.student_pretrained,
                "student_num_classes": self.model.student_num_classes,
                "teacher_layer_filter": self.model.teacher_layer_filter,
                "exclude_bn": self.model.exclude_bn,
                "student_layer_filter": self.model.student_layer_filter,
            },
            "dataset": {
                "name": self.dataset.name,
                "data_dir": self.dataset.data_dir,
                "train_split": self.dataset.train_split,
                "val_split": self.dataset.val_split,
                "image_size": self.dataset.image_size,
                "normalize_mean": self.dataset.normalize_mean,
                "normalize_std": self.dataset.normalize_std,
                "batch_size": self.dataset.batch_size,
                "num_workers": self.dataset.num_workers,
                "pin_memory": self.dataset.pin_memory,
                "drop_last": self.dataset.drop_last,
            },
            "optimizer": {
                "type": self.optimizer.type,
                "lr": self.optimizer.lr,
                "weight_decay": self.optimizer.weight_decay,
                "momentum": self.optimizer.momentum,
                "nesterov": self.optimizer.nesterov,
                "beta1": self.optimizer.beta1,
                "beta2": self.optimizer.beta2,
                "eps": self.optimizer.eps,
            },
            "scheduler": {
                "type": self.scheduler.type,
                "T_max": self.scheduler.T_max,
                "eta_min": self.scheduler.eta_min,
                "step_size": self.scheduler.step_size,
                "gamma": self.scheduler.gamma,
                "mode": self.scheduler.mode,
                "factor": self.scheduler.factor,
                "patience": self.scheduler.patience,
                "threshold": self.scheduler.threshold,
                "warmup_steps": self.scheduler.warmup_steps,
                "warmup_lr": self.scheduler.warmup_lr,
            },
            "trainer": {
                "num_epochs": self.trainer.num_epochs,
                "device": self.trainer.device,
                "temperature": self.trainer.temperature,
                "lambda_weight": self.trainer.lambda_weight,
                "alpha_flow": self.trainer.alpha_flow,
                "use_weights": self.trainer.use_weights,
                "precompute_teacher_divs": self.trainer.precompute_teacher_divs,
                "use_selective_alignment": self.trainer.use_selective_alignment,
                "selective_error_threshold": self.trainer.selective_error_threshold,
                "selective_update_freq": self.trainer.selective_update_freq,
                "use_integrated_fad": self.trainer.use_integrated_fad,
                "num_integration_points": self.trainer.num_integration_points,
            },
            "logging": {
                "log_dir": self.logging.log_dir,
                "use_tensorboard": self.logging.use_tensorboard,
                "tensorboard_log_dir": self.logging.tensorboard_log_dir,
                "use_wandb": self.logging.use_wandb,
                "wandb_project": self.logging.wandb_project,
                "wandb_entity": self.logging.wandb_entity,
                "wandb_run_name": self.logging.wandb_run_name,
                "log_freq": self.logging.log_freq,
                "print_freq": self.logging.print_freq,
            },
            "checkpoint": {
                "save_dir": self.checkpoint.save_dir,
                "save_freq": self.checkpoint.save_freq,
                "save_best": self.checkpoint.save_best,
                "save_last": self.checkpoint.save_last,
                "resume_from": self.checkpoint.resume_from,
                "resume_optimizer": self.checkpoint.resume_optimizer,
                "resume_scheduler": self.checkpoint.resume_scheduler,
                "resume_epoch": self.checkpoint.resume_epoch,
                "metric_to_monitor": self.checkpoint.metric_to_monitor,
                "mode": self.checkpoint.mode,
            },
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExperimentConfig":
        """Create configuration from dictionary."""

        # Extract sub-configs
        model_dict = config_dict.get("model", {})
        dataset_dict = config_dict.get("dataset", {})
        optimizer_dict = config_dict.get("optimizer", {})
        scheduler_dict = config_dict.get("scheduler", {})
        trainer_dict = config_dict.get("trainer", {})
        logging_dict = config_dict.get("logging", {})
        checkpoint_dict = config_dict.get("checkpoint", {})

        return cls(
            experiment_name=config_dict.get("experiment_name", "fad_experiment"),
            seed=config_dict.get("seed", 42),
            comment=config_dict.get("comment", None),
            model=ModelConfig(**model_dict),
            dataset=DatasetConfig(**dataset_dict),
            optimizer=OptimizerConfig(**optimizer_dict),
            scheduler=SchedulerConfig(**scheduler_dict),
            trainer=FADTrainerConfig(**trainer_dict),
            logging=LoggingConfig(**logging_dict),
            checkpoint=CheckpointConfig(**checkpoint_dict),
        )


def get_default_config() -> ExperimentConfig:
    """Get default FAD configuration with reasonable defaults."""

    return ExperimentConfig()


def get_cifar10_config() -> ExperimentConfig:
    """Get configuration optimized for CIFAR-10 dataset."""

    config = get_default_config()
    config.dataset.name = "cifar10"
    config.dataset.image_size = 32
    config.dataset.normalize_mean = [0.4914, 0.4822, 0.4465]
    config.dataset.normalize_std = [0.2023, 0.1994, 0.2010]
    config.dataset.batch_size = 128
    config.model.student_num_classes = 10
    config.optimizer.lr = 0.1
    config.optimizer.type = "sgd"
    config.optimizer.weight_decay = 5e-4
    config.scheduler.type = "cosine"
    config.scheduler.T_max = 200
    config.trainer.num_epochs = 200
    config.trainer.temperature = 4.0

    return config


def get_imagenet_config() -> ExperimentConfig:
    """Get configuration optimized for ImageNet dataset."""

    config = get_default_config()
    config.dataset.name = "imagenet"
    config.dataset.image_size = 224
    config.dataset.batch_size = 64
    config.model.student_num_classes = 1000
    config.optimizer.lr = 0.001
    config.optimizer.type = "adamw"
    config.optimizer.weight_decay = 1e-4
    config.scheduler.type = "cosine"
    config.scheduler.T_max = 100
    config.trainer.num_epochs = 100
    config.trainer.temperature = 4.0

    return config
