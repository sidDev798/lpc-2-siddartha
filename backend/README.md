# Python Code Assistant - Backend API

A FastAPI backend that provides code generation and debugging for Python code.

## Features

- Generate Python code based on text prompts
- Fix and improve existing Python code
- RESTful API endpoints for chat-based interaction

## Getting Started

### Prerequisites

- Python 3.8+
- FastAPI
- Uvicorn

### Installation

1. Install the required dependencies:

```bash
pip install fastapi uvicorn python-multipart
```

### Running the server

Start the FastAPI server:

```bash
cd backend
python main.py
```

The server will start on http://localhost:8000

### API Endpoints

- `GET /` - Health check endpoint
- `POST /chat` - Main endpoint for code generation and fixing

#### Chat Request Format

```json
{
  "message": "Create a function that calculates fibonacci numbers",
  "file_content": null  // Optional: Include Python code to fix
}
```

#### Chat Response Format

```json
{
  "text": "Here's some Python code for your request",
  "code": "def fibonacci(n):\n    ..."
}
```

## Integration with Frontend

This backend is designed to work with the Python Code Assistant frontend. By default, the frontend expects this backend to be running on `http://localhost:8000`.

## Customization

To enhance code generation capabilities, you can:

1. Replace the `generate_python_code` function with a more sophisticated implementation
2. Integrate with a language model API like OpenAI for better code generation
3. Implement more advanced code fixing in the `fix_python_code` function

## License

MIT 