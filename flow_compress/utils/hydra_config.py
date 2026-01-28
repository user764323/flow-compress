"""
Hydra configuration management for FAD experiments.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from hydra import compose, initialize_config_dir, initialize
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import DictConfig, OmegaConf

    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    logging.warning("Warning: hydra-core not available. Install with: pip install hydra-core")


class HydraConfigManager:
    """
    Configuration manager using Hydra for hierarchical YAML configuration.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config_name: str = "config",
        version_base: Optional[str] = None,
    ):
        if not HYDRA_AVAILABLE:
            raise ImportError(
                "hydra-core is required. Install with: pip install hydra-core"
            )

        self.config_path = config_path or os.path.join(os.getcwd(), "configs")
        self.config_name = config_name
        self.version_base = version_base

        # Ensure config directory exists
        os.makedirs(self.config_path, exist_ok=True)

        # Initialize Hydra if not already initialized
        if not GlobalHydra().is_initialized():
            if os.path.exists(self.config_path):
                initialize_config_dir(
                    config_dir=self.config_path,
                    config_name=config_name,
                    version_base=version_base,
                )
            else:
                # Initialize with default config
                initialize(
                    config_path=None,
                    config_name=config_name,
                    version_base=version_base,
                )

        logging.info(
            f"Initialized HydraConfigManager: config_path={self.config_path}")

    def get_config(self, overrides: Optional[list] = None) -> DictConfig:
        """
        Get configuration with optional overrides.
        """
        if overrides is None:
            overrides = []

        cfg = compose(config_name=self.config_name, overrides=overrides)
        return cfg

    def save_config(self, config: DictConfig, path: str) -> None:
        """
        Save configuration to file.
        """
        OmegaConf.save(config, path)
        logging.info(f"Saved configuration to {path}")

    def print_config(self, config: DictConfig) -> None:
        """
        Print configuration in a readable format.
        """
        logging.info(OmegaConf.to_yaml(config))

    def to_dict(self, config: DictConfig) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        """
        return OmegaConf.to_container(config, resolve=True)


def create_default_configs(config_dir: str = "./configs") -> None:
    """
    Create default configuration files for FAD experiments.
    """
    os.makedirs(config_dir, exist_ok=True)

    # Main config
    main_config = ""

    # Model configs
    model_dir = os.path.join(config_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    resnet_config = ""

    # Dataset configs
    dataset_dir = os.path.join(config_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    cifar10_config = ""

    # Trainer configs
    trainer_dir = os.path.join(config_dir, "trainer")
    os.makedirs(trainer_dir, exist_ok=True)

    fad_trainer_config = ""

    curriculum_trainer_config = ""

    # Optimizer configs
    optimizer_dir = os.path.join(config_dir, "optimizer")
    os.makedirs(optimizer_dir, exist_ok=True)

    adamw_config = ""

    # Scheduler configs
    scheduler_dir = os.path.join(config_dir, "scheduler")
    os.makedirs(scheduler_dir, exist_ok=True)

    cosine_config = ""

    # Write config files
    configs = {
        "config.yaml": main_config,
        "model/resnet50.yaml": resnet_config,
        "model/vit_base.yaml": vit_config,
        "dataset/cifar10.yaml": cifar10_config,
        "trainer/fad_trainer.yaml": fad_trainer_config,
        "trainer/curriculum_trainer.yaml": curriculum_trainer_config,
        "optimizer/adamw.yaml": adamw_config,
        "scheduler/cosine_annealing.yaml": cosine_config,
    }

    for rel_path, content in configs.items():
        file_path = os.path.join(config_dir, rel_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(content.strip())

    logging.info(f"Created default configuration files in {config_dir}")


if __name__ == "__main__":
    # Create default configs when run as script
    create_default_configs()
