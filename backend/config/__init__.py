"""
Configuration management for the application.
Loads environment variables and provides configuration access.
"""

import os
import importlib
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Determine environment
ENV = os.getenv("APP_ENV", "development").lower()
valid_envs = ["development", "production", "testing"]

if ENV not in valid_envs:
    print(f"Warning: Unknown environment '{ENV}', defaulting to 'development'")
    ENV = "development"

# Import the appropriate configuration module
try:
    config_module = importlib.import_module(f"backend.config.{ENV}")
except ImportError:
    print(f"Error: Could not load configuration for environment '{ENV}'")
    raise

# Configuration dictionary to hold all settings
config = {}

# Load all uppercase variables from the config module into the config dictionary
for key in dir(config_module):
    if key.isupper():
        config[key] = getattr(config_module, key)

# Override config values with environment variables
for key in config:
    env_value = os.getenv(key)
    if env_value is not None:
        # Convert environment variable value to the appropriate type
        original_type = type(config[key]) if config[key] is not None else str
        if original_type == bool:
            config[key] = env_value.lower() in ('true', 'yes', '1', 'y')
        elif original_type == int:
            config[key] = int(env_value)
        elif original_type == float:
            config[key] = float(env_value)
        else:
            config[key] = env_value

def get(key, default=None):
    """
    Get a configuration value by key.
    
    Args:
        key (str): Configuration key
        default: Default value if key is not found
        
    Returns:
        The configuration value or default if not found
    """
    return config.get(key, default)

def set(key, value):
    """
    Set a configuration value.
    
    Args:
        key (str): Configuration key
        value: Value to set
    """
    config[key] = value

def get_all():
    """
    Get all configuration values.
    
    Returns:
        dict: All configuration values
    """
    return config.copy() 