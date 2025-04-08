# Configuration System

This module provides a centralized configuration system for the application.

## How It Works

1. Configuration values are stored in environment-specific Python modules:
   - `development.py` - Development environment settings
   - `production.py` - Production environment settings
   - `testing.py` - Testing environment settings

2. Environment variables can override any configuration value.

3. The `APP_ENV` environment variable determines which configuration module to load. If not specified, defaults to "development".

## Usage

### In your code

Import the configuration module and use the `get` function to retrieve configuration values:

```python
from backend.config import get

# Get OpenAI API key
api_key = get("OPENAI_API_KEY")

# Get a configuration value with a default
debug_mode = get("DEBUG", False)
```

You can also set configuration values programmatically:

```python
from backend.config import set

# Set a configuration value
set("CUSTOM_SETTING", "value")
```

To get all configuration values:

```python
from backend.config import get_all

# Get all configuration values
all_config = get_all()
```

### Environment Variables

You can override any configuration value by setting an environment variable with the same name.

For example, to override the `OPENAI_API_KEY` configuration value:

```bash
export OPENAI_API_KEY=your-api-key-here
```

Or set it in a `.env` file in the project root:

```
OPENAI_API_KEY=your-api-key-here
```

## Adding New Configuration Values

To add a new configuration value:

1. Add it to the relevant environment-specific configuration module(s).
2. Access it in your code using the `get` function.

Example:

```python
# In development.py
NEW_SETTING = "value"

# In your code
from backend.config import get
setting_value = get("NEW_SETTING")
``` 