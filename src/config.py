"""
Configuration loader for the ABSA system.
Reads config.yaml and provides typed access to all settings.
"""

import yaml
import os

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
_config = None


def load_config(path=None):
    """Load configuration from YAML file.
    
    Args:
        path: Optional path to config file. Defaults to project root config.yaml.
    
    Returns:
        dict: Configuration dictionary.
    
    Raises:
        FileNotFoundError: If config file doesn't exist.
    """
    global _config
    if _config is not None and path is None:
        return _config

    config_path = path or _CONFIG_PATH
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        _config = yaml.safe_load(f)

    return _config


def get_model_paths():
    """Get all model file paths from config."""
    config = load_config()
    return config.get('models', {})


def get_inference_thresholds():
    """Get inference correction thresholds."""
    config = load_config()
    return config.get('inference', {})


def get_training_params():
    """Get training hyperparameters."""
    config = load_config()
    return config.get('training', {})


def get_logging_config():
    """Get logging configuration."""
    config = load_config()
    return config.get('logging', {})
