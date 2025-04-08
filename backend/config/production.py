"""
Production environment configuration.
Contains production environment specific settings.
"""

# OpenAI settings
OPENAI_API_KEY = None  # Should be set from environment variables in production
OPENAI_MODEL = "gpt-4o"
OPENAI_TEMPERATURE = 0.5  # Lower temperature for more deterministic outputs in production

# Application settings
DEBUG = False
LOG_LEVEL = "INFO" 