import hydra
from omegaconf import DictConfig


def read_config(*_, config_path: str, config_name: str) -> DictConfig:
    try:
        # Initialize Hydra
        hydra.initialize(config_path=config_path)
    except ValueError:
        pass

    # Compose the configuration
    cfg = hydra.compose(config_name=config_name)

    return cfg
