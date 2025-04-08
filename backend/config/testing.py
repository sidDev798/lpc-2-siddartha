"""
Testing environment configuration.
Contains test-specific settings.
"""

# OpenAI settings
OPENAI_API_KEY = "test-api-key"  # Mock key for testing
OPENAI_MODEL = "gpt-3.5-turbo"  # Use a cheaper model for testing
OPENAI_TEMPERATURE = 0.0  # Deterministic outputs for testing

# Application settings
DEBUG = True
LOG_LEVEL = "DEBUG"
TESTING = True 