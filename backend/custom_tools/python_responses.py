import openai
from backend.config import get
from langchain_core.tools import tool
from langgraph.types import Command
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from typing_extensions import Annotated

@tool
def direct_response(query: str, tool_call_id: Annotated[str, InjectedToolCallId], config: RunnableConfig) -> dict:
    """Provide a direct response to Python-related questions."""

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
    system_message = "You are good at answering questions in a concise and helpful manner for a user who is a beginner in Python."
    
    # Prepare the prompt with clear instructions
    user_prompt = f"Write a response to the following query:\n\n{query}"
    
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
        
    generated_response = response.choices[0].message.content.strip()
    

    return Command(
        update={
            "messages": [ToolMessage(generated_response, tool_call_id=tool_call_id)]
        }
    )

