
from langchain_core.tools import tool
from typing import Dict
from backend.config import get
import openai

def debug_code(code: str):
    """Debug and fix Python code."""
    
    # Get configuration values with provided parameters taking precedence
    api_key = get("OPENAI_API_KEY", "sk-...")
    model = get("OPENAI_MODEL", "gpt-4o")
    temperature = get("OPENAI_TEMPERATURE", 0.7)
    
    # Check for API key
    if api_key is None:
        return {
            "success": False,
            "error": "OpenAI API key not provided in parameters or configuration",
            "code": None
        }
    
    openai.api_key = api_key
    
    # Create system message to instruct the model to generate code
    system_message = "You are an expert debugger in Python. Generate only code with no explanations or comments unless specifically requested in the prompt. Make the code working.The code should be complete, correct, and ready to run."
    
    # Prepare the prompt with clear instructions
    user_prompt = f"Debug the following code. Return only the code, no explanations:\n\n{code}"
    
    # Make API call to OpenAI
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature
    )
    
    # Extract the generated code - fix potential None issue
    if not response or not hasattr(response, 'choices') or len(response.choices) == 0:
        return {
            "success": False, 
            "error": "No response content received from OpenAI API",
            "code": None
        }
        
    if not response.choices[0] or not hasattr(response.choices[0], 'message') or not response.choices[0].message:
        return {
            "success": False, 
            "error": "No message content in OpenAI API response",
            "code": None
        }
        
    if not hasattr(response.choices[0].message, 'content') or not response.choices[0].message.content:
        return {
            "success": False, 
            "error": "Empty content in OpenAI API response",
            "code": None
        }
        
    generated_code = response.choices[0].message.content.strip()
    
    # Remove markdown code blocks if present
    if generated_code.startswith("```") and generated_code.endswith("```"):
        # Extract language if specified
        first_line = generated_code.split("\n")[0].strip("`").strip()
        if first_line and not first_line.startswith("#") and not first_line.startswith("import"):
            generated_code = "\n".join(generated_code.split("\n")[1:])
        
        # Remove ending backticks
        if generated_code.endswith("```"):
            generated_code = generated_code[:-3].strip()

    return {
        "message": f"Fixed code:\n{generated_code}",
        "code": generated_code
    }