# Backend Debugging Guide

This document provides information about the backend logging system and debugging tools.

## Logging System

The backend has comprehensive logging implemented across multiple components:

1. **Main Application Logs** - General application logs with all levels (DEBUG, INFO, WARNING, ERROR)
2. **API Request/Response Logs** - Detailed logs of all API requests and responses
3. **Tool-specific Logs** - Each tool has its own logging for detailed execution tracking

### Log Files

- `backend_debug.log` - Main application log file
- `api_requests.log` - API requests and responses
- `openai_code_generator.log` - Generated when running the code generator directly

## Log Viewer Tool

A log viewer utility script (`logviewer.py`) is provided to help filter and display logs:

```bash
# View all logs with default settings
python backend/logviewer.py

# View only ERROR and above messages
python backend/logviewer.py --level=ERROR

# View logs containing a specific string
python backend/logviewer.py --filter="OpenAI"

# Follow logs in real-time (like tail -f)
python backend/logviewer.py --follow

# Show only the last 20 lines
python backend/logviewer.py --last=20

# View API logs
python backend/logviewer.py --api

# Combine options
python backend/logviewer.py --level=WARNING --filter="error" --follow
```

## Common Issues and Debug Approaches

### OpenAI API Issues

If you're seeing issues with the OpenAI API:

1. Check the API key in your `.env` file
2. Look for API errors in the logs:
   ```bash
   python backend/logviewer.py --filter="OpenAI API error"
   ```
3. Verify model names and parameters in the code

### Python Code Execution Problems

For issues with code execution:

1. Check the execution environment logs:
   ```bash
   python backend/logviewer.py --filter="Executing Python code"
   ```
2. Look for errors in stderr:
   ```bash
   python backend/logviewer.py --filter="stderr"
   ```

### API Communication Issues

For problems with API requests or responses:

1. View the API logs:
   ```bash
   python backend/logviewer.py --api
   ```
2. Check for specific request IDs:
   ```bash
   python backend/logviewer.py --api --filter="1713480150.12345"
   ```

## Enabling Additional Debug Logging

To increase logging detail for specific components, edit the following files:

1. **main.py** - Change `level=logging.DEBUG` to a more detailed level if needed
2. **python_agent.py** - Add more debug statements for specific areas
3. **custom_tools/*.py** - Add logging to specific tools that need debugging

## Reporting Issues

When reporting issues, please include:

1. Relevant log sections (use the log viewer to filter for errors)
2. API request and response details if applicable
3. Steps to reproduce the issue
4. Information about your environment (OS, Python version) 