from typing import Annotated, Optional, List, Dict, Any, TypedDict, Literal, Union, cast
import os
import logging
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, FunctionMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import ToolMessage

from .config import get


from .custom_tools import tools, code_generation, code_debug
# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Define the state for our agent
class PythonAssistantState(TypedDict):
    """State for the Python Assistant agent."""
    messages: Annotated[List[BaseMessage], add_messages]  # Chat messages
    code: Optional[str]  # Optional code to process

# Initialize LLM
logger.info("Initializing ChatOpenAI with gpt-4o model")
llm = ChatOpenAI(
    model="gpt-4o", 
    temperature=0.1,
    api_key=get("OPENAI_API_KEY")
)

# Create the ToolNode
logger.info(f"Creating ToolNode with {len(tools)} tools")
tool_node = ToolNode(tools)
tools_by_name: dict[str, BaseTool] = {_tool.name: _tool for _tool in tools}

# Create the model with tools
logger.info("Binding tools to the model")
model_with_tools = llm.bind_tools(tools)

def should_continue(state: MessagesState) -> Union[Literal["generate_code"], Literal["debug_code"], Literal["END"]]:
    """Determine if we should route to generate_code, debug_code, or end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Safely check if the last message is from AI and has tool calls

    has_tool_calls = getattr(last_message, 'tool_calls', None) is not None and bool(getattr(last_message, 'tool_calls', []))
    if has_tool_calls:

        tool_calls = last_message.tool_calls
        tool_name = tool_calls[0]['name']
        
        if tool_name == 'generate_code':
            return cast(Literal["generate_code"], "generate_code")
        elif tool_name == 'debug_code':
            return cast(Literal["debug_code"], "debug_code")
    else:
        # For all other tools, just end the conversation
        return  cast(Literal["END"], END)
    
    logger.debug("No tool calls detected, ending conversation")
    return cast(Literal["END"], END)

def call_model(state: MessagesState) -> Dict[str, List[BaseMessage]]:
    """Call the model with the current messages."""
    messages = state["messages"]
    logger.debug(f"Calling model with {len(messages)} messages")
    
    # Add system message for context
    system_message = SystemMessage(content="""You are a Python assistant that can help with:
    1. Generating Python code
    2. Debugging and fixing code
    3. Answering Python-related questions
    4. Running Python code
    
    Use the available tools when appropriate:
    - generate_code: For creating new Python code
    - debug_code: For fixing and improving existing code
    - direct_response: For answering Python questions
    - run_python_code: For running Python code    
    
    Make sure to format code with ```python code blocks.
    """)
    
    # Add system message if not present
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        logger.debug("Adding system message to messages")
        messages = [system_message] + messages
    
    # Get model response
    logger.debug("Invoking model")
    try:
        # Log message types for debugging
        message_types = [f"{i}: {type(msg).__name__}" for i, msg in enumerate(messages)]
        logger.debug(f"Message types: {message_types}")
        
        # Log last message content for debugging (truncated)
        if messages and hasattr(messages[-1], 'content'):
            last_content = str(messages[-1].content)
            content_preview = f"{last_content[:100]}..." if len(last_content) > 100 else last_content
            logger.debug(f"Last message content preview: {content_preview}")
        
        response = model_with_tools.invoke(messages)

        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Error invoking model: {str(e)}", exc_info=True)
        raise

def generate_code_node(state: MessagesState) -> Dict[str, Any]:
    """Handle generate_code tool calls."""
    messages = state["messages"]
    last_message = messages[-1]
    
    result = []
    # Use getattr for safety
    tool_calls = getattr(last_message, 'tool_calls', [])
    if tool_calls:
        for tool_call in tool_calls:
            if tool_call['name'] == 'generate_code':
                
                # Extract arguments from the tool call
                args = tool_call['args']
                if isinstance(args, str):
                    import json
                    try:
                        args = json.loads(args)
                    except:
                        args = {}
                
                observation = code_generation.generate_code(query=args['__arg1'])
        

                result.append(ToolMessage(content=observation.get('message'),
                                        tool_call_id=tool_call['id']))
    
    return {"messages": result}

def debug_code_node(state: MessagesState) -> Dict[str, Any]:
    """Handle debug_code tool calls."""
    messages = state["messages"]
    last_message = messages[-1]
    
    result = []
    # Use getattr for safety
    tool_calls = getattr(last_message, 'tool_calls', [])
    if tool_calls:
        for tool_call in tool_calls:
            if tool_call['name'] == 'debug_code':
                
                # Extract arguments from the tool call
                args = tool_call['args']
                if isinstance(args, str):
                    import json
                    try:
                        args = json.loads(args)
                    except:
                        args = {}
                
                observation = code_debug.debug_code(code=args['__arg1'])
                
                message_content = observation.get('message')
                    
                result.append(ToolMessage(content=message_content,
                                        tool_call_id=tool_call['id']))
    
    return {"messages": result}

# Build the graph
def build_graph():
    """Build the LangGraph for the Python assistant."""
    logger.info("Building LangGraph for Python assistant")
    # Create the graph
    workflow = StateGraph(MessagesState)
    
    # Add nodes
    logger.debug("Adding nodes to graph")
    workflow.add_node("agent", call_model)
    workflow.add_node("generate_code", generate_code_node)
    workflow.add_node("debug_code", debug_code_node)
    
    # Add edges
    logger.debug("Adding edges to graph")
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "generate_code": "generate_code",
            "debug_code": "debug_code",
            END: END
        }
    )
    workflow.add_edge("generate_code", "agent")
    workflow.add_edge("debug_code", "agent")
    
    # Compile the graph
    logger.debug("Compiling graph")
    return workflow.compile()

# Create the agent
logger.info("Creating Python assistant agent")
python_assistant = build_graph()

# Function to run the agent
def run_python_assistant(query: str, code: Optional[str] = None) -> Dict[str, Any]:
    """
    Run the Python assistant with a query and optional code.
    
    Args:
        query: The user's question or request
        code: Optional Python code to analyze or fix
    
    Returns:
        Dict containing response text and optional code
    """
    logger.info(f"Running Python assistant with query: {query[:50]}...")
    # Initial state
    initial_message = query
    if code:
        logger.debug(f"Code provided ({len(code)} characters)")
        initial_message += f"\n```python\n{code}\n```"
        
    initial_state = {
        "messages": [HumanMessage(content=initial_message)]
    }
    
    # Run the graph
    try:
        logger.debug("Invoking the Python assistant graph")
        result = python_assistant.invoke(initial_state)
        logger.debug("Graph execution completed")
    except Exception as e:
        logger.error(f"Error running Python assistant: {str(e)}")
        raise
    
    # Extract final response
    final_messages = result["messages"]
    final_message = final_messages[-1].content if final_messages else ""
    logger.debug(f"Final message extracted (length: {len(final_message)})")
    
    # Extract code if present
    code_output = None
    if "```python" in final_message:
        logger.debug("Extracting code from message")
        code_blocks = final_message.split("```python")
        for block in code_blocks:
            if "```" in block:
                code_output = block.split("```")[0].strip()
                logger.debug(f"Code extracted (length: {len(code_output)})")
                break
    
    logger.info("Python assistant execution completed")
    return {
        "text": final_message,
        "code": code_output
    }
