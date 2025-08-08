import yaml
import os
import logging
from .logger import default_logger as logger

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        logger.error("❌ Config file not found: %s", config_path)
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        try:
            cfg = yaml.safe_load(f)
            logger.info("✅ Loaded config from %s", config_path)
            return cfg
        except yaml.YAMLError as e:
            logger.error("❌ Failed to parse config file: %s", e)
            raise