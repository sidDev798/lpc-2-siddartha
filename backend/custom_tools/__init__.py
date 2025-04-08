from langchain_core.tools import Tool
from .code_generation import generate_code
from .code_debug import debug_code
from .python_responses import direct_response
from .code_execution import run_python_code

# Create the tools list
tools = [
    Tool(
        name="generate_code",
        description="Generate Python code based on the user's request",
        func=generate_code,
    ),
    Tool(
        name="debug_code",
        description="Debug and fix Python code",
        func=debug_code
    ),
    Tool(
        name="run_python_code",
        description="Run Python code in a virtual environment",
        func=run_python_code
    ),
    Tool(
        name="direct_response",
        description="Answer Python-related questions directly",
        func=direct_response
    )
]