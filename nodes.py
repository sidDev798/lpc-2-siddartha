"""
LangGraph nodes for DataStory using vanna.ai (Combined Version)
"""
from typing import Dict, Any, List, Optional, Union, TypedDict, Annotated
import time
import re
import io
import json
import pandas as pd
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt
import plotly.utils

# Configuration and utility imports
from backend.config.logging_config import get_logger
from backend.config.config import get_vanna_instance as get_default_vanna
from backend.vanna_custom.config import get_vanna_instance as get_custom_vanna
from backend.session_manager.utils import get_contextual_history
from backend.conversation_manager.manager_mongo import MongoConversationManager

# Import prompts
from backend.graph.prompts import (
    query_analysis_prompt, 
    human_interruption_query_prompt, 
    intent_classification_prompt
)

# Import utility functions
from backend.graph.utils import (
    extract_sql_from_response, 
    check_sql_syntax, 
    check_for_dangerous_operations, 
    is_complex_query, 
    is_high_risk_operation
)

# Create loggers for each node function
query_analyzer_logger = get_logger("datastory.graph.query_analyzer")
human_input_logger = get_logger("datastory.graph.human_input")
intent_classifier_logger = get_logger("datastory.graph.intent_classifier")
submit_prompt_logger = get_logger("datastory.graph.submit_prompt")
intermediate_check_logger = get_logger("datastory.graph.intermediate_check")
context_enhancer_logger = get_logger("datastory.graph.context_enhancer")
sql_extractor_logger = get_logger("datastory.graph.extract_sql")
validate_sql_logger = get_logger("datastory.graph.validate_sql")
human_review_logger = get_logger("datastory.graph.human_sql_review")
error_handler_logger = get_logger("datastory.graph.error_handler")
run_sql_logger = get_logger("datastory.graph.run_sql")
result_processor_logger = get_logger("datastory.graph.result_processor")
training_agent_logger = get_logger("datastory.graph.training_agent")
visualization_check_logger = get_logger("datastory.graph.visualization_check")
summary_logger = get_logger("datastory.graph.generate_summary")
plotly_code_logger = get_logger("datastory.graph.generate_plotly_code")
plotly_figure_logger = get_logger("datastory.graph.get_plotly_figure")
result_check_logger = get_logger("datastory.graph.result_check")
human_feedback_logger = get_logger("datastory.graph.human_feedback")
explainer_logger = get_logger("datastory.graph.explainer")
format_logger = get_logger("datastory.graph.format_response")

# ---- Utility Functions ----

def extract_sql_from_response(response: str) -> str:
    """
    Extract SQL code from LLM response
    
    Args:
        response: LLM response potentially containing SQL code
        
    Returns:
        Extracted SQL query or error message
    """
    sql_match = re.search(r'```sql\n(.*?)\n```', response, re.DOTALL)
    if sql_match:
        sql_code = sql_match.group(1).strip()
        # Remove any trailing semicolon or newline characters
        sql_code = sql_code.rstrip(';').rstrip('\n')
        # Further refinement to handle potential multiline SQL
        sql_code = sql_code.replace("\\n", "\n")
        # Remove quotes
        return sql_code.strip("'")
    else:
        return ""  # Empty string indicates extraction failed


# ---- LangGraph Node Functions ----

# Query Analysis
async def query_analyzer(state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Analyze the query to determine if clarification is needed
    
    Args:
        state: The current graph state
        config: Optional runnable configuration
        
    Returns:
        Updated state with clarification flags
    """
    # Create a unique identifier for this node execution
    node_id = id(state)
    query_analyzer_logger.info(f"[node_id={node_id}] Starting query_analyzer execution")
    query_analyzer_logger.debug(f"[node_id={node_id}] Input state keys: {list(state.keys())}")

    query = state.get("current_question", "")
    conversation_history = state.get("conversation_history", [])

    # Get ConversationManager from config if available
    configurable_config = config.get("configurable", {}) if config else {}
    conversation_manager = configurable_config.get("conversation_manager")
    
    # Update conversation history if needed
    if conversation_manager and not conversation_history:
        conversation_id = state.get("conversation_id")
        if conversation_id:
            try:
                query_analyzer_logger.debug(f"[node_id={node_id}] Fetching history for {conversation_id}")
                conversation_history = await get_contextual_history(
                    conversation_id=conversation_id,
                    conversation_manager=conversation_manager
                )
                query_analyzer_logger.info(f"[node_id={node_id}] Fetched {len(conversation_history)} messages from DB")
            except Exception as e:
                query_analyzer_logger.error(f"[node_id={node_id}] Error fetching history: {e}")

    prompt = ChatPromptTemplate.from_template(query_analysis_prompt)
    model = ChatOpenAI(model="gpt-4o", temperature=0, model_kwargs={"response_format": {"type": "json_object"}})
    chain = {"conversation_history": RunnablePassthrough(), "query": RunnablePassthrough()} | prompt | model

    query_analyzer_logger.debug(f"[node_id={node_id}] Analyzing query: {query}")
    response = chain.invoke({"conversation_history": conversation_history, "query": query})
    query_analyzer_logger.debug(f"[node_id={node_id}] LLM response: {response}")

    # Default values
    is_clarification_needed = False
    clarification_message = None
    
    # Try to extract the required information
    try:
        # Access the content directly
        content = response.content if hasattr(response, 'content') else ''
        
        if isinstance(content, str) and content.strip():
            try:
                parsed_content = json.loads(content)
                if isinstance(parsed_content, dict):
                    is_clarification_needed = parsed_content.get('is_clarification_needed', False)
                    clarification_message = parsed_content.get('clarification_message', None)
                    query_analyzer_logger.info(f"[node_id={node_id}] Clarification needed: {is_clarification_needed}")
            except json.JSONDecodeError as e:
                query_analyzer_logger.error(f"[node_id={node_id}] Error parsing JSON response: {e}")
    except Exception as e:
        query_analyzer_logger.error(f"[node_id={node_id}] Error processing response: {e}")
    
    query_analyzer_logger.info(f"[node_id={node_id}] Completed query_analyzer execution")
    return {
        "is_clarification_needed": is_clarification_needed,
        "clarification_message": clarification_message
    }

# Human Input for Clarification
async def human_input_node(state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Get human input for clarification
    
    Args:
        state: The current graph state
        config: Optional runnable configuration
        
    Returns:
        Updated state with clarified question
    """
    # Create a unique identifier for this node execution
    node_id = id(state)
    human_input_logger.info(f"[node_id={node_id}] Starting human_input_node execution")
    human_input_logger.debug(f"[node_id={node_id}] Input state keys: {list(state.keys())}")

    conversation_history = state.get("conversation_history", [])
    query = state.get("current_question", "")
    
    human_input_logger.info(f"[node_id={node_id}] Requesting clarification from user")
    clarification_response = interrupt({
        "task": "Clarification needed",
        "message": state.get("clarification_message", "Please provide more details for your query."),
        "current_question": state.get("current_question", "")
    })
    human_input_logger.debug(f"[node_id={node_id}] Received clarification: {clarification_response}")

    prompt = ChatPromptTemplate.from_template(human_interruption_query_prompt)
    model = ChatOpenAI(model="gpt-4o", temperature=0, model_kwargs={"response_format": {"type": "json_object"}})
    chain = {"conversation_history": RunnablePassthrough(), "query": RunnablePassthrough(), "human_input": RunnablePassthrough()} | prompt | model

    response = chain.invoke({"conversation_history": conversation_history, "query": query, "human_input": clarification_response})
    human_input_logger.debug(f"[node_id={node_id}] LLM response: {response}")

    # Default to current question
    updated_query = state.get("current_question", "")
    
    # Try to extract the required information
    try:
        # Access the content directly
        content = response.content if hasattr(response, 'content') else ''
        
        if isinstance(content, str) and content.strip():
            try:
                parsed_content = json.loads(content)
                if isinstance(parsed_content, dict):
                    updated_query = parsed_content.get('updated_query', updated_query)
                    human_input_logger.info(f"[node_id={node_id}] Updated query: {updated_query}")
            except json.JSONDecodeError as e:
                human_input_logger.error(f"[node_id={node_id}] Error parsing JSON response: {e}")
    except Exception as e:
        human_input_logger.error(f"[node_id={node_id}] Error processing response: {e}")
    
    human_input_logger.info(f"[node_id={node_id}] Completed human_input_node execution")
    return {
        "current_question": updated_query
    }

# Intent Classification
async def intent_classifier(state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Classify the intent of the query
    
    Args:
        state: The current graph state
        config: Optional runnable configuration
        
    Returns:
        Updated state with intent classification
    """
    # Create a unique identifier for this node execution
    node_id = id(state)
    intent_classifier_logger.info(f"[node_id={node_id}] Starting intent_classifier execution")
    intent_classifier_logger.debug(f"[node_id={node_id}] Input state keys: {list(state.keys())}")
    
    conversation_history = state.get("conversation_history", [])
    query = state.get("current_question", "")
    
    intent = "sql_generation"  # Default intent

    prompt = ChatPromptTemplate.from_template(intent_classification_prompt)
    model = ChatOpenAI(model="gpt-4o", temperature=0, model_kwargs={"response_format": {"type": "json_object"}})
    chain = {"conversation_history": RunnablePassthrough(), "query": RunnablePassthrough()} | prompt | model

    intent_classifier_logger.debug(f"[node_id={node_id}] Classifying intent for query: {query}")
    response = chain.invoke({"conversation_history": conversation_history, "query": query})
    intent_classifier_logger.debug(f"[node_id={node_id}] LLM response: {response}")

    # Try to extract the required information
    try:
        # Access the content directly
        content = response.content if hasattr(response, 'content') else ''
        
        if isinstance(content, str) and content.strip():
            try:
                parsed_content = json.loads(content)
                if isinstance(parsed_content, dict):
                    intent = parsed_content.get('intent', intent)
                    intent_classifier_logger.info(f"[node_id={node_id}] Classified intent: {intent}")
            except json.JSONDecodeError as e:
                intent_classifier_logger.error(f"[node_id={node_id}] Error parsing JSON response: {e}")
    except Exception as e:
        intent_classifier_logger.error(f"[node_id={node_id}] Error processing response: {e}")
    
    intent_classifier_logger.info(f"[node_id={node_id}] Completed intent_classifier execution")
    return {
        "intent": intent  # One of "sql_generation", "visualization_only", "explanation_only"
    }

async def submit_prompt(state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Fetches latest history using config, then submits the prompt 
    (question + context) to the Vanna instance to generate SQL.
    
    Args:
        state: The current graph state.
        config: Runnable configuration possibly containing 'conversation_manager'.
        
    Returns:
        Updated state with generated SQL query or error.
    """
    submit_prompt_logger.info(f"--- Entering submit_prompt node ---")
    
    # Create a unique identifier for this node execution
    node_id = id(state)
    submit_prompt_logger.info(f"[node_id={node_id}] Starting submit_prompt node execution")
    submit_prompt_logger.debug(f"[node_id={node_id}] Input state keys: {list(state.keys())}")
    
    # --- Get ConversationManager from config --- 
    configurable_config = config.get("configurable", {}) if config else {}
    conversation_manager = configurable_config.get("conversation_manager")
    if not conversation_manager:
        submit_prompt_logger.error(f"[node_id={node_id}] ConversationManager not found in config!")
        # Proceed without history
    
    # --- Extract necessary info from state --- 
    conversation_id = state.get("conversation_id")
    current_question = state.get("current_question")
    user_id = state.get("user_id")
    
    submit_prompt_logger.debug(f"[node_id={node_id}] Extracted from state: conv_id={conversation_id}, question='{current_question}'")

    # Handle missing question
    if not current_question or not current_question.strip():
        error_msg = "No question found in the current state."
        submit_prompt_logger.error(f"[node_id={node_id}] {error_msg}")
        return {
            "llm_response": error_msg, 
            "sql_query": "-- Error: No question", 
            "node_error": error_msg
        }

    # --- Fetch Latest History --- 
    latest_history_models = []
    if conversation_manager and conversation_id:
        submit_prompt_logger.info(f"[node_id={node_id}] Fetching history for conv_id: {conversation_id}")
        try:
            # First log what we received in the state
            received_history = state.get("conversation_history", [])
            submit_prompt_logger.info(f"[node_id={node_id}] Received {len(received_history)} messages in state.conversation_history")
            
            latest_history_models = await get_contextual_history(
                conversation_id=conversation_id,
                conversation_manager=conversation_manager,
                # context_window=5 # Default is 5, adjust if needed
            )
            submit_prompt_logger.info(f"[node_id={node_id}] Fetched {len(latest_history_models)} message models from DB for context enhancement.")
            
            # Log some details about the history messages
            if latest_history_models:
                history_preview = ", ".join([f"{msg.role[:1]}" for msg in latest_history_models[:5]])
                if len(latest_history_models) > 5:
                    history_preview += "..."
                submit_prompt_logger.debug(f"[node_id={node_id}] History roles sequence: {history_preview}")
        except Exception as e:
            submit_prompt_logger.exception(f"[node_id={node_id}] Error fetching history for {conversation_id}: {e}")
            # Continue without history on error
    else:
        submit_prompt_logger.warning(f"[node_id={node_id}] Cannot fetch history: Missing conversation_manager or conversation_id.")
        
    # --- Format History + Current Question for Vanna --- 
    # Vanna expects a list of messages (dicts or objects) representing the conversation
    # The current question should be the LAST item in this list.
    history_for_vanna = []
    for msg in latest_history_models:
         # Assuming get_contextual_history returns models with .role and .content
         if hasattr(msg, 'role') and hasattr(msg, 'content'):
             history_for_vanna.append(msg) # Pass the model object directly if Vanna handles it
         else:
             submit_prompt_logger.warning(f"[node_id={node_id}] Skipping history message with unexpected format: {type(msg)}")
             
    # Append the current question as the last user message
    try:
        from backend.conversation_manager.models.mongodb_models import MessageMongo # Adjust import if needed
        # Ensure conversation_id is included when creating the message object
        current_q_message = MessageMongo(
            role="user", 
            content=current_question, 
            conversation_id=conversation_id # Add missing conversation_id
        )
        history_for_vanna.append(current_q_message)
        submit_prompt_logger.info(f"[node_id={node_id}] Formatted {len(history_for_vanna)} total messages for Vanna (history + current question)")
    except ImportError as e:
        submit_prompt_logger.error(f"[node_id={node_id}] Could not import MessageMongo to format current question: {e}")

    # --- Get Vanna instance --- 
    # Use the CUSTOM ConversationalVannaAPI
    submit_prompt_logger.debug(f"[node_id={node_id}] Getting custom Vanna instance for conversation capabilities")
    vn_instance = get_custom_vanna()

    # --- Call Vanna instance --- 
    sql_query = None
    llm_response = None
    node_error = None
    
    try:
        # Check training status
        has_training = False
        try:
            if hasattr(vn_instance, 'check_training_status'):
                 training_status = vn_instance.check_training_status()
                 submit_prompt_logger.debug(f"[node_id={node_id}] Training status: {training_status}")
                 has_training = training_status.get("has_training", False)
            else:
                 # Fallback training check logic
                 training_data = vn_instance.get_training_data()
                 if isinstance(training_data, pd.DataFrame):
                     has_training = not training_data.empty
                 elif training_data:
                     has_training = True
                 submit_prompt_logger.debug(f"[node_id={node_id}] Fallback training data check: {has_training}, type: {type(training_data)}")
                 
            submit_prompt_logger.debug(f"[node_id={node_id}] Final training status: {has_training}")
        except Exception as e:
             submit_prompt_logger.warning(f"[node_id={node_id}] Error checking training status: {e}")
             # Fallback check
             try:
                 training_data = vn_instance.get_training_data()
                 if isinstance(training_data, pd.DataFrame): 
                     has_training = not training_data.empty
                 elif training_data: 
                     has_training = True
                 submit_prompt_logger.debug(f"[node_id={node_id}] Training data check: {has_training}, type: {type(training_data)}")
             except Exception as e2:
                 submit_prompt_logger.warning(f"[node_id={node_id}] Error in fallback training data check: {e2}")
                 
        if not has_training:
            error_msg = "Vanna has not been trained yet. Please train the model first."
            submit_prompt_logger.warning(f"[node_id={node_id}] No training data: {error_msg}")
            sql_query = "-- Error: No training data"
            llm_response = error_msg
            node_error = error_msg
            # Go directly to returning the error state
            return {
                "llm_response": llm_response, 
                "sql_query": sql_query, 
                "node_error": node_error
            }
        
        # Call generate_sql with the UPDATED history (fetched + current question)
        submit_prompt_logger.info(f"[node_id={node_id}] Calling Vanna.generate_sql with conversation context ({len(history_for_vanna)} messages)")
        
        if hasattr(vn_instance, 'generate_sql'):
            # Pass the combined history here
            generated_result = vn_instance.generate_sql(
                question=current_question, # Still pass question for clarity if needed by Vanna method
                conversation_history=history_for_vanna 
            )
            
            # Check if the result is a conversational response (dict with is_conversational=True)
            if isinstance(generated_result, dict) and generated_result.get("is_conversational"):
                submit_prompt_logger.info(f"[node_id={node_id}] Received conversational response from Vanna")
                # Extract the conversation_answer for the response
                conversation_answer = generated_result.get("conversation_answer", "I don't know how to answer that question.")
                
                # Return a state with the conversational flag and answer
                submit_prompt_logger.debug(f"[node_id={node_id}] Conversational answer: {conversation_answer[:50]}...")
                return {
                    "is_conversational": True,
                    "sql_query": None,
                    "llm_response": conversation_answer,
                    "conversation_answer": conversation_answer,
                    "query_result": [{"answer": conversation_answer, "type": "conversational"}]
                }
            else:
                # Handle normal SQL generation result
                submit_prompt_logger.info(f"[node_id={node_id}] Received SQL generation result from Vanna")
                sql_query = generated_result 
                
        # If we get here, we have a SQL query (not a conversational response)
        if sql_query:
            submit_prompt_logger.info(f"[node_id={node_id}] Generated SQL query (length: {len(sql_query)})")
            submit_prompt_logger.debug(f"[node_id={node_id}] SQL query: {sql_query}")
            return {"llm_response": sql_query, "sql_query": sql_query}
        else:
            llm_response = "Unable to generate SQL for this question."
            node_error = "generate_sql returned None or empty result"
            submit_prompt_logger.warning(f"[node_id={node_id}] {node_error}: {llm_response}")
            return {
                "llm_response": llm_response, 
                "sql_query": "-- Error: No SQL generated", 
                "node_error": node_error
            }
            
    except Exception as e:
        submit_prompt_logger.exception(f"[node_id={node_id}] Error in submit_prompt: {str(e)}")
        llm_response = f"Error generating SQL: {str(e)}"
        node_error = str(e)
        return {
            "llm_response": llm_response, 
            "sql_query": "-- Error: " + str(e), 
            "node_error": node_error
        }

async def intermediate_check(state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Check if context is sufficient for SQL generation
    
    Args:
        state: The current graph state
        config: Optional runnable configuration
        
    Returns:
        Updated state with context sufficiency flag
    """
    # Create a unique identifier for this node execution
    node_id = id(state)
    intermediate_check_logger.info(f"[node_id={node_id}] Starting intermediate_check execution")
    intermediate_check_logger.debug(f"[node_id={node_id}] Input state keys: {list(state.keys())}")
    
    sql_query = state.get("sql_query", "")
    is_context_sufficient = True  # Default to sufficient
    
    # Check for signs of insufficient context in the SQL query
    if "-- Need more context" in sql_query or "-- Need more information" in sql_query:
        is_context_sufficient = False
        intermediate_check_logger.info(f"[node_id={node_id}] Context insufficient based on SQL query comments")
    
    # Check for extraction error that might indicate insufficient context
    extraction_error = state.get("extraction_error")
    if extraction_error and ("context" in extraction_error.lower() or "information" in extraction_error.lower()):
        is_context_sufficient = False
        intermediate_check_logger.info(f"[node_id={node_id}] Context insufficient based on extraction error")
    
    intermediate_check_logger.info(f"[node_id={node_id}] Completed intermediate_check execution: is_context_sufficient={is_context_sufficient}")
    return {
        "is_context_sufficient": is_context_sufficient
    }

async def extract_sql(state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Extract SQL from the LLM response
    
    Args:
        state: The current graph state
        config: Runnable configuration (optional)
        
    Returns:
        Updated state with extracted SQL
    """
    # Create a unique identifier for this node execution
    node_id = id(state)
    sql_extractor_logger.info(f"[node_id={node_id}] Starting extract_sql execution")
    sql_extractor_logger.debug(f"[node_id={node_id}] Input state keys: {list(state.keys())}")
    
    # Check if we already have a conversational response
    if state.get("is_conversational", False):
        sql_extractor_logger.info(f"[node_id={node_id}] Skipping SQL extraction for conversational response")
        return {}  # Return empty dict to keep existing conversational state
    
    # Use 'llm_response' from the previous node
    llm_response = state.get("llm_response", "") 
    extraction_error = None
    sql_query = ""  # Initialize sql_query

    if not llm_response:
        sql_extractor_logger.warning(f"[node_id={node_id}] No LLM response found in state")
        extraction_error = "No LLM response provided for SQL extraction."
        sql_query = f"-- Error: No LLM response to extract SQL from."
    else:
        sql_extractor_logger.info(f"[node_id={node_id}] Attempting to extract SQL from LLM response")
        sql_query = extract_sql_from_response(llm_response)  # Use utility function
        
        if not sql_query:
            sql_extractor_logger.warning(f"[node_id={node_id}] Could not extract SQL from response using regex")
            extraction_error = "Could not extract SQL from response."
            # Try to use the actual question as a fallback message within the SQL comment
            question = state.get("current_question", "")
            sql_extractor_logger.debug(f"[node_id={node_id}] Using current_question for fallback message: {question}")
            sql_query = f"-- Unable to extract SQL from response. Question was: {question}"
        else:
            sql_extractor_logger.info(f"[node_id={node_id}] Successfully extracted SQL: {sql_query[:50]}...")

    result = {
        "sql_query": sql_query,
        "extraction_error": extraction_error
    }
    sql_extractor_logger.debug(f"[node_id={node_id}] Output state keys: {list(result.keys())}")
    sql_extractor_logger.info(f"[node_id={node_id}] Completed extract_sql execution")
    return result

async def validate_sql(state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Validate the generated SQL to ensure it's safe and correct
    
    Args:
        state: The current graph state
        config: Runnable configuration (optional)
        
    Returns:
        Updated state with validation info
    """
    # Create a unique identifier for this node execution
    node_id = id(state)
    validate_sql_logger.info(f"[node_id={node_id}] Starting validate_sql execution")
    validate_sql_logger.debug(f"[node_id={node_id}] Input state keys: {list(state.keys())}")
    
    sql_query = state.get("sql_query", "")
    validate_sql_logger.debug(f"[node_id={node_id}] SQL query to validate: {sql_query}")
    
    is_valid = True
    has_errors = False
    validation_message = ""
    needs_human_review = False
    
    # Basic validation checks
    if not sql_query or len(sql_query.strip()) == 0:
        is_valid = False
        has_errors = True
        validation_message = "Empty SQL query"
        validate_sql_logger.warning(f"[node_id={node_id}] Validation failed: {validation_message}")
    elif sql_query.strip().startswith('--'): 
        is_valid = False
        has_errors = True
        validation_message = "SQL query appears to be a comment or error message"
        validate_sql_logger.warning(f"[node_id={node_id}] Validation failed: {validation_message}")
    else:
        validate_sql_logger.info(f"[node_id={node_id}] Performing detailed SQL validation")
        # Syntax checking
        syntax_errors = check_sql_syntax(sql_query)
        if syntax_errors:
            is_valid = False
            has_errors = True
            validation_message = f"SQL syntax error: {syntax_errors}"
            validate_sql_logger.warning(f"[node_id={node_id}] Syntax validation failed: {validation_message}")
        else:
            # Check for dangerous operations
            dangerous_ops = check_for_dangerous_operations(sql_query)
            if dangerous_ops:
                is_valid = False
                has_errors = True
                validation_message = f"Dangerous SQL operations detected: {dangerous_ops}"
                validate_sql_logger.warning(f"[node_id={node_id}] Security validation failed: {validation_message}")
            else:
                # Check if query is complex and might need human review
                if is_complex_query(sql_query) or is_high_risk_operation(sql_query):
                    needs_human_review = True
                    validate_sql_logger.info(f"[node_id={node_id}] Complex query flagged for human review")
                else:
                    validate_sql_logger.info(f"[node_id={node_id}] SQL validated successfully")
    
    result = {
        "sql_valid": is_valid,
        "has_errors": has_errors,
        "validation_message": validation_message,
        "needs_human_review": needs_human_review
    }
    validate_sql_logger.debug(f"[node_id={node_id}] Validation result: valid={is_valid}, errors={has_errors}, needs_review={needs_human_review}")
    validate_sql_logger.info(f"[node_id={node_id}] Completed validate_sql execution")
    return result

async def context_enhancer(state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Enhance context for SQL generation
    
    Args:
        state: The current graph state
        config: Runnable configuration (optional)
        
    Returns:
        Updated state with enhanced context
    """
    # Create a unique identifier for this node execution
    node_id = id(state)
    context_enhancer_logger.info(f"[node_id={node_id}] Starting context_enhancer execution")
    context_enhancer_logger.debug(f"[node_id={node_id}] Input state keys: {list(state.keys())}")
    
    # Get current context
    context = state.get("context", {})
    current_question = state.get("current_question", "")
    
    # Get Vanna instance
    vn_instance = get_custom_vanna()
    
    # Enhance context with additional schema information
    try:
        # Get related schema information
        context_enhancer_logger.info(f"[node_id={node_id}] Fetching additional schema information")
        related_ddl = vn_instance.get_related_ddl(current_question)
        related_docs = vn_instance.get_related_documentation(current_question)
        
        # Update context
        enhanced_context = context.copy()
        if related_ddl:
            enhanced_context["related_schema"] = related_ddl
            context_enhancer_logger.info(f"[node_id={node_id}] Added {len(related_ddl)} related schema items")
        
        if related_docs:
            enhanced_context["related_documentation"] = related_docs
            context_enhancer_logger.info(f"[node_id={node_id}] Added {len(related_docs)} related documentation items")
        
        context_enhancer_logger.info(f"[node_id={node_id}] Completed context_enhancer execution")
        return {
            "context": enhanced_context
        }
    except Exception as e:
        context_enhancer_logger.error(f"[node_id={node_id}] Error enhancing context: {e}")
        return {
            "context": context,  # Return original context on error
            "node_error": f"Error enhancing context: {str(e)}"
        }

async def human_sql_review(state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Get human review for SQL query
    
    Args:
        state: The current graph state
        config: Runnable configuration (optional)
        
    Returns:
        Updated state with human review action
    """
    # Create a unique identifier for this node execution
    node_id = id(state)
    human_review_logger.info(f"[node_id={node_id}] Starting human_sql_review execution")
    human_review_logger.debug(f"[node_id={node_id}] Input state keys: {list(state.keys())}")
    
    sql_query = state.get("sql_query", "")
    current_question = state.get("current_question", "")
    
    # Prepare prompt for human review
    review_message = f"""The following SQL query was generated for your question:

Question: {current_question}

SQL Query:
```sql
{sql_query}
```

Please review this SQL query and decide on an action:
- Proceed: Execute the SQL query as-is
- Edit: Modify the SQL query (you'll be prompted to provide an edited version)
- Regenerate: Generate a new SQL query
- Cancel: Stop the execution"""
    
    human_review_logger.info(f"[node_id={node_id}] Requesting human review of SQL query")
    human_action = interrupt({
        "task": "Review SQL Query",
        "message": review_message,
        "options": ["Proceed", "Edit", "Regenerate", "Cancel"]
    })
    human_review_logger.debug(f"[node_id={node_id}] Received human action: {human_action}")
    
    human_review_action = None
    human_edited_sql = None
    
    # Map the human response to an action
    if human_action.lower() == "proceed":
        human_review_action = "proceed"
        human_review_logger.info(f"[node_id={node_id}] Human chose to proceed with the SQL query")
    elif human_action.lower() == "edit":
        human_review_action = "proceed"  # Still proceed, but with edited SQL
        
        # Request edited SQL
        human_review_logger.info(f"[node_id={node_id}] Requesting edited SQL from human")
        edited_sql = interrupt({
            "task": "Edit SQL Query",
            "message": f"Please provide your edited SQL query:\n\nOriginal:\n```sql\n{sql_query}\n```\n\nEdited:",
            "is_long_response": True
        })
        
        human_edited_sql = edited_sql.strip()
        human_review_logger.debug(f"[node_id={node_id}] Received edited SQL: {human_edited_sql}")
    elif human_action.lower() == "regenerate":
        human_review_action = "regenerate"
        human_review_logger.info(f"[node_id={node_id}] Human chose to regenerate the SQL query")
    else:  # Cancel or any unexpected response
        human_review_action = "cancel"
        human_review_logger.info(f"[node_id={node_id}] Human chose to cancel the SQL execution")
    
    result = {
        "human_review_action": human_review_action
    }
    
    # Include edited SQL if provided
    if human_edited_sql:
        result["human_edited_sql"] = human_edited_sql
        result["sql_query"] = human_edited_sql  # Update the query with edited version
    
    human_review_logger.debug(f"[node_id={node_id}] Human review result: action={human_review_action}")
    human_review_logger.info(f"[node_id={node_id}] Completed human_sql_review execution")
    return result

async def error_handler(state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Handle errors in SQL validation or execution
    
    Args:
        state: The current graph state
        config: Runnable configuration (optional)
        
    Returns:
        Updated state with error handling
    """
    # Create a unique identifier for this node execution
    node_id = id(state)
    error_handler_logger.info(f"[node_id={node_id}] Starting error_handler execution")
    error_handler_logger.debug(f"[node_id={node_id}] Input state keys: {list(state.keys())}")
    
    # Extract error information
    validation_message = state.get("validation_message", "")
    extraction_error = state.get("extraction_error", "")
    query_error = state.get("query_error", "")
    node_error = state.get("node_error", "")
    
    # Determine the most relevant error
    error_message = validation_message or extraction_error or query_error or node_error or "Unknown error"
    error_handler_logger.warning(f"[node_id={node_id}] Handling error: {error_message}")
    
    # Prepare a context with error information to help regeneration
    current_context = state.get("context", {})
    enhanced_context = current_context.copy()
    
    # Add error information to context
    enhanced_context["error_information"] = {
        "error_message": error_message,
        "failed_sql": state.get("sql_query", "")
    }
    
    error_handler_logger.info(f"[node_id={node_id}] Enhanced context with error information for regeneration")
    error_handler_logger.debug(f"[node_id={node_id}] Enhanced context: {enhanced_context}")
    
    # Reset SQL state for regeneration
    result = {
        "context": enhanced_context,
        "sql_query": None,
        "sql_valid": False,
        "has_errors": False,  # Reset error state
        "needs_human_review": False  # Reset human review flag
    }
    
    error_handler_logger.info(f"[node_id={node_id}] Completed error_handler execution")
    return result

async def run_sql(state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Execute the SQL query if it's valid
    
    Args:
        state: The current graph state
        config: Runnable configuration (optional)
        
    Returns:
        Updated state with query results
    """
    # Create a unique identifier for this node execution
    node_id = id(state)
    run_sql_logger.info(f"[node_id={node_id}] Starting run_sql execution")
    run_sql_logger.debug(f"[node_id={node_id}] Input state keys: {list(state.keys())}")
    
    sql_query = state.get("sql_query", "")
    run_sql_logger.debug(f"[node_id={node_id}] SQL query to execute: '{sql_query}'")
    
    # Check for human edited SQL
    human_edited_sql = state.get("human_edited_sql")
    if human_edited_sql:
        sql_query = human_edited_sql
        run_sql_logger.info(f"[node_id={node_id}] Using human-edited SQL: {sql_query}")
    
    # Skip execution if the SQL is invalid
    if not state.get("sql_valid", False):
        run_sql_logger.warning(f"[node_id={node_id}] Skipping execution because SQL is invalid")
        error_message = state.get("validation_message", "Invalid SQL query")
        
        return {
            "query_result": None,
            "query_error": error_message,
            "dataframe_json": None,
            "has_execution_error": True,
            "execution_successful": False,
            "execution_time": None
        }
    
    try:
        # Execute the SQL query using Vanna
        run_sql_logger.info(f"[node_id={node_id}] Executing SQL query...")
        vn_instance = get_default_vanna()
        start_time = time.time()
        result_df = vn_instance.run_sql(sql_query)
        execution_time = time.time() - start_time
        run_sql_logger.info(f"[node_id={node_id}] Query executed in {execution_time:.2f} seconds")
        
        # Convert DataFrame to JSON-serializable formats for API response
        query_result = None
        dataframe_json = None
        
        if isinstance(result_df, pd.DataFrame):
            run_sql_logger.info(f"[node_id={node_id}] Got DataFrame with shape: {result_df.shape}")
            
            # Convert to records for API response
            if not result_df.empty:
                query_result = result_df.to_dict(orient='records')
                run_sql_logger.debug(f"[node_id={node_id}] Converted to {len(query_result)} result records")
                
                # Create a JSON-serializable representation for the state
                try:
                    # Include column types and basic stats for API consumers
                    column_types = {col: str(dtype) for col, dtype in result_df.dtypes.items()}
                    dataframe_json = {
                        "columns": list(result_df.columns),
                        "column_types": column_types,
                        "row_count": len(result_df),
                        "records": query_result[:100] if len(query_result) > 100 else query_result,  # Limit records in state
                        "truncated": len(query_result) > 100  # Flag if truncated
                    }
                except Exception as e:
                    run_sql_logger.error(f"[node_id={node_id}] Error serializing DataFrame: {e}")
                    # Fallback to simpler serialization
                    dataframe_json = {
                        "columns": list(result_df.columns),
                        "row_count": len(result_df),
                        "records": query_result[:100] if len(query_result) > 100 else query_result
                    }
            else:
                run_sql_logger.warning(f"[node_id={node_id}] Query returned empty DataFrame")
                query_result = []
                dataframe_json = {"columns": list(result_df.columns), "row_count": 0, "records": []}    
        else:
            run_sql_logger.warning(f"[node_id={node_id}] Query did not return a DataFrame: {type(result_df)}")
            query_result = [{"result": str(result_df)}] if result_df is not None else []
            dataframe_json = {"special_result": str(result_df) if result_df is not None else "None"}
        
        result = {
            "query_result": query_result,
            "result_dataframe": result_df,  # Store the actual DataFrame for visualization
            "dataframe_json": dataframe_json,  # JSON serializable version
            "execution_time": execution_time,
            "has_execution_error": False,
            "execution_successful": True,
            "query_error": None
        }
        
        run_sql_logger.info(f"[node_id={node_id}] Completed run_sql execution successfully")
        return result
        
    except Exception as e:
        run_sql_logger.exception(f"[node_id={node_id}] Error executing SQL: {str(e)}")
        
        result = {
            "query_result": None,
            "result_dataframe": None,
            "dataframe_json": None,
            "execution_time": None,
            "has_execution_error": True,
            "execution_successful": False,
            "query_error": f"Error executing SQL: {str(e)}"
        }
        
        run_sql_logger.info(f"[node_id={node_id}] Completed run_sql execution with error")
        return result

async def result_processor(state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Process query results
    
    Args:
        state: The current graph state
        config: Runnable configuration (optional)
        
    Returns:
        Updated state with processed results
    """
    # Create a unique identifier for this node execution
    node_id = id(state)
    result_processor_logger.info(f"[node_id={node_id}] Starting result_processor execution")
    result_processor_logger.debug(f"[node_id={node_id}] Input state keys: {list(state.keys())}")
    
    query_result = state.get("query_result", [])
    current_question = state.get("current_question", "")
    
    # Default values
    needs_training = False
    needs_visualization = True  # Default to generating visualization
    
    # Check if we have results to process
    if not query_result or (isinstance(query_result, list) and len(query_result) == 0):
        result_processor_logger.warning(f"[node_id={node_id}] No query results to process")
        needs_visualization = False
        
    # Determine if we need training
    # For example, if the query was successful and not already in training examples
    if state.get("execution_successful", False) and state.get("sql_query"):
        try:
            # Get Vanna instance
            vn_instance = get_default_vanna()
            
            # Check if similar question exists in training data
            similar_questions = vn_instance.get_similar_question_sql(current_question)
            if not similar_questions:
                needs_training = True
                result_processor_logger.info(f"[node_id={node_id}] Marking for training: No similar questions found")
            else:
                result_processor_logger.debug(f"[node_id={node_id}] Similar questions found in training data: {len(similar_questions)}")
                
        except Exception as e:
            result_processor_logger.error(f"[node_id={node_id}] Error checking training need: {e}")
    
    result_processor_logger.info(f"[node_id={node_id}] Processing complete - needs_training={needs_training}, needs_visualization={needs_visualization}")
    return {
        "needs_training": needs_training,
        "needs_visualization": needs_visualization
    }

async def training_agent(state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Add examples to context for training
    
    Args:
        state: The current graph state
        config: Runnable configuration (optional)
        
    Returns:
        Updated state with training status
    """
    # Create a unique identifier for this node execution
    node_id = id(state)
    training_agent_logger.info(f"[node_id={node_id}] Starting training_agent execution")
    training_agent_logger.debug(f"[node_id={node_id}] Input state keys: {list(state.keys())}")
    
    if not state.get("needs_training", False) or not state.get("execution_successful", False):
        training_agent_logger.info(f"[node_id={node_id}] Skipping training - not needed or execution unsuccessful")
        return {}
    
    try:
        # Get training data
        question = state.get("current_question", "")
        sql_query = state.get("sql_query", "")
        
        if not question or not sql_query:
            training_agent_logger.warning(f"[node_id={node_id}] Missing question or SQL for training")
            return {}
        
        # Get Vanna instance
        vn_instance = get_default_vanna()
        
        # Add example to training data
        training_agent_logger.info(f"[node_id={node_id}] Adding example to training data: {question[:50]}...")
        vn_instance.train(question=question, sql=sql_query)
        
        training_agent_logger.info(f"[node_id={node_id}] Successfully added training example")
        return {
            "agent_messages": [{
                "role": "assistant",
                "content": "Added this query to training examples for future reference."
            }]
        }
    except Exception as e:
        training_agent_logger.error(f"[node_id={node_id}] Error adding training example: {e}")
        return {
            "node_error": f"Error adding training example: {str(e)}"
        }

async def visualization_check(state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Check if visualization is needed
    
    Args:
        state: The current graph state
        config: Runnable configuration (optional)
        
    Returns:
        Updated state with visualization need flag
    """
    # Create a unique identifier for this node execution
    node_id = id(state)
    visualization_check_logger.info(f"[node_id={node_id}] Starting visualization_check execution")
    visualization_check_logger.debug(f"[node_id={node_id}] Input state keys: {list(state.keys())}")
    
    # Default to needing visualization
    needs_visualization = True
    
    # Check if we have data to visualize
    result_df = state.get("result_dataframe")
    dataframe_json = state.get("dataframe_json")
    query_error = state.get("query_error")
    
    # Skip visualization if we have errors or no data
    if query_error or not (result_df is not None or dataframe_json is not None):
        needs_visualization = False
        visualization_check_logger.info(f"[node_id={node_id}] Skipping visualization due to errors or missing data")
    
    # Skip if dataframe is empty
    elif (isinstance(result_df, pd.DataFrame) and result_df.empty) or \
         (isinstance(dataframe_json, dict) and dataframe_json.get("row_count", 0) == 0):
        needs_visualization = False
        visualization_check_logger.info(f"[node_id={node_id}] Skipping visualization due to empty result set")
        
    # Skip visualization if the dataframe has only one row and one column
    elif isinstance(result_df, pd.DataFrame) and result_df.shape[0] == 1 and result_df.shape[1] == 1:
        needs_visualization = False
        visualization_check_logger.info(f"[node_id={node_id}] Skipping visualization for single-value result")
        
    else:
        visualization_check_logger.info(f"[node_id={node_id}] Visualization needed for result data")
    
    visualization_check_logger.info(f"[node_id={node_id}] Visualization check complete: needs_visualization={needs_visualization}")
    return {
        "needs_visualization": needs_visualization
    }

async def generate_summary(state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Generate natural language summary from query results
    
    Args:
        state: The current graph state
        config: Runnable configuration (optional)
        
    Returns:
        Updated state with summary
    """
    # Create a unique identifier for this node execution
    node_id = id(state)
    summary_logger.info(f"[node_id={node_id}] Starting generate_summary execution")
    summary_logger.debug(f"[node_id={node_id}] Input state keys: {list(state.keys())}")
    
    # Default to no summary
    summary = None
    summary_error = None
    
    # Check if we have data to summarize
    if state.get("is_conversational", False):
        # For conversational responses, use that as the summary
        summary = state.get("conversation_answer")
        summary_logger.info(f"[node_id={node_id}] Using conversational answer as summary")
        
    else:
        # For SQL results, generate a summary
        dataframe_json = state.get("dataframe_json")
        query_result = state.get("query_result")
        current_question = state.get("current_question", "")
        sql_query = state.get("sql_query", "")
    
        # Skip summary if no data
        if not query_result or not dataframe_json:
            summary_logger.warning(f"[node_id={node_id}] No query results to summarize")
            summary = "No data available to summarize."
            return {"summary": summary}
        
        # Prepare summary prompt
        try:
            # Get Vanna instance
            vn_instance = get_custom_vanna()
            
            # Generate a prompt with data preview and context
            result_info = {
                "question": current_question,
                "sql_query": sql_query,
                "column_info": dataframe_json.get("columns", []),
                "row_count": dataframe_json.get("row_count", 0),
                "data_preview": str(dataframe_json.get("records", [])[:5]),
                "data_truncated": dataframe_json.get("truncated", False)
            }
            
            # Generate summary using Vanna's summarization capability
            summary_logger.info(f"[node_id={node_id}] Generating summary for {result_info['row_count']} rows of data")
            
            summary_prompt = f"""
            You are analyzing the results of a database query. Please provide a concise, natural language summary of the following data.
            
            Original question: {result_info['question']}
            
            SQL Query: {result_info['sql_query']}
            
            Data columns: {', '.join(result_info['column_info'])}
            Number of rows: {result_info['row_count']}
            Sample data: {result_info['data_preview']}
            
            Please provide a helpful summary of these results in plain language. Focus on addressing the original question.
            Mention key insights, trends, or notable data points. Be specific about the numbers found in the data.
            """
            
            # Generate summary using LLM
            model = ChatOpenAI(model="gpt-4o", temperature=0) 
            messages = [SystemMessage(content="You are a helpful data analyst assistant."),
                         SystemMessage(content=summary_prompt)]
            summary_response = model.invoke(messages)
            
            summary = summary_response.content if hasattr(summary_response, 'content') else str(summary_response)
            summary_logger.info(f"[node_id={node_id}] Generated summary: {summary[:100]}...")
        except Exception as e:
            summary_logger.error(f"[node_id={node_id}] Error generating summary: {e}")
            summary_error = f"Error generating summary: {str(e)}"
            summary = "Could not generate a summary of the results."
    
    summary_logger.info(f"[node_id={node_id}] Completed generate_summary execution")
    return {
        "summary": summary,
        "summary_error": summary_error
    }

async def generate_plotly_code(state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Generate Plotly visualization code for query results
    
    Args:
        state: The current graph state
        config: Runnable configuration (optional)
        
    Returns:
        Updated state with visualization code
    """
    # Create a unique identifier for this node execution
    node_id = id(state)
    plotly_code_logger.info(f"[node_id={node_id}] Starting generate_plotly_code execution")
    plotly_code_logger.debug(f"[node_id={node_id}] Input state keys: {list(state.keys())}")
    
    # Skip if we've determined visualization is not needed
    if not state.get("needs_visualization", True):
        plotly_code_logger.info(f"[node_id={node_id}] Skipping visualization as needs_visualization=False")
        return {"plotly_code": None}
    
    # Skip for conversational responses
    if state.get("is_conversational", False):
        plotly_code_logger.info(f"[node_id={node_id}] Skipping visualization for conversational response")
        return {"plotly_code": None}
    
    try:
        # Get result dataframe
        result_df = state.get("result_dataframe")
        current_question = state.get("current_question", "")
        
        # Check if we have data to visualize
        if not isinstance(result_df, pd.DataFrame) or result_df.empty:
            plotly_code_logger.warning(f"[node_id={node_id}] No DataFrame to visualize")
            return {"plotly_code": None, "visualization_error": "No data available for visualization"}
        
        # Get Vanna instance
        vn_instance = get_custom_vanna()
        
        # Generate visualization code
        plotly_code_logger.info(f"[node_id={node_id}] Generating Plotly code for DataFrame with shape {result_df.shape}")
        plotly_code = vn_instance.generate_plotly_code(question=current_question, df=result_df)
        
        if not plotly_code:
            plotly_code_logger.warning(f"[node_id={node_id}] Failed to generate Plotly code")
            return {"plotly_code": None, "visualization_error": "Could not generate visualization code"}
        
        plotly_code_logger.info(f"[node_id={node_id}] Successfully generated Plotly code: {len(plotly_code)} characters")
        return {"plotly_code": plotly_code}
    
    except Exception as e:
        plotly_code_logger.error(f"[node_id={node_id}] Error generating Plotly code: {e}")
        return {"plotly_code": None, "visualization_error": f"Error generating visualization code: {str(e)}"}

async def get_plotly_figure(state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Convert Plotly code to an actual Plotly figure
    
    Args:
        state: The current graph state
        config: Runnable configuration (optional)
        
    Returns:
        Updated state with Plotly figure
    """
    # Create a unique identifier for this node execution
    node_id = id(state)
    plotly_figure_logger.info(f"[node_id={node_id}] Starting get_plotly_figure execution")
    plotly_figure_logger.debug(f"[node_id={node_id}] Input state keys: {list(state.keys())}")
    
    plotly_figure = None
    visualization_error = state.get("visualization_error") 
    
    try:
        # Only generate figure if we have code, dataframe, and no error
        plotly_code = state.get("plotly_code")
        result_df = state.get("result_dataframe")
        
        if not plotly_code:
            plotly_figure_logger.warning(f"[node_id={node_id}] No Plotly code available")
            return {"plotly_figure": None}
        
        plotly_figure_logger.debug(f"[node_id={node_id}] Plotly code length: {len(plotly_code)}")
        
        if not isinstance(result_df, pd.DataFrame):
            plotly_figure_logger.warning(f"[node_id={node_id}] No DataFrame available")
            return {"plotly_figure": None, "visualization_error": "Missing DataFrame for visualization"}
        
        plotly_figure_logger.debug(f"[node_id={node_id}] DataFrame shape: {result_df.shape}")
        
        # Only generate figure if dataframe has data
        if result_df.empty:
            plotly_figure_logger.warning(f"[node_id={node_id}] DataFrame is empty")
            return {"plotly_figure": None, "visualization_error": "Empty data for visualization"}
        
        # Get Vanna instance
        vn_instance = get_custom_vanna()
        
        # Use Vanna's get_plotly_figure function
        plotly_figure_logger.info(f"[node_id={node_id}] Creating Plotly figure from code")
        plotly_figure = vn_instance.get_plotly_figure(plotly_code=plotly_code, df=result_df)
        
        if not plotly_figure:
            plotly_figure_logger.warning(f"[node_id={node_id}] Failed to create Plotly figure")
            visualization_error = "Failed to create visualization from code"
        else:
            # Serialize the figure for state storage
            try:
                # Convert to JSON-serializable format
                plotly_figure_logger.debug(f"[node_id={node_id}] Converting Plotly figure to JSON")
                plotly_figure = json.loads(json.dumps(plotly_figure, cls=plotly.utils.PlotlyJSONEncoder))
                plotly_figure_logger.info(f"[node_id={node_id}] Successfully created and serialized figure")
            except Exception as e:
                plotly_figure_logger.error(f"[node_id={node_id}] Error serializing figure: {e}")
                visualization_error = f"Error serializing figure: {str(e)}"
                plotly_figure = None
            
    except Exception as e:
        plotly_figure_logger.error(f"[node_id={node_id}] Error creating Plotly figure: {e}")
        if not visualization_error:
            visualization_error = f"Error creating visualization figure: {str(e)}"
    
    plotly_figure_logger.info(f"[node_id={node_id}] Completed get_plotly_figure execution")
    return {
        "plotly_figure": plotly_figure,
        "visualization_error": visualization_error
    }

async def result_check(state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Check if results are sufficient
    
    Args:
        state: The current graph state
        config: Runnable configuration (optional)
        
    Returns:
        Updated state with result sufficiency flag
    """
    # Create a unique identifier for this node execution
    node_id = id(state)
    result_check_logger.info(f"[node_id={node_id}] Starting result_check execution")
    result_check_logger.debug(f"[node_id={node_id}] Input state keys: {list(state.keys())}")
    
    # Default to sufficient results
    results_sufficient = True
    
    # Check for errors or issues that might make results insufficient
    if state.get("has_execution_error", False):
        results_sufficient = False
        result_check_logger.warning(f"[node_id={node_id}] Results insufficient due to execution error")
    
    elif state.get("query_error"):
        results_sufficient = False
        result_check_logger.warning(f"[node_id={node_id}] Results insufficient due to query error")
    
    # Consider visualization errors
    elif state.get("visualization_error") and state.get("needs_visualization", True):
        # If visualization was needed but failed, might need human feedback
        result_check_logger.warning(f"[node_id={node_id}] Visualization error might require human feedback")
        # Don't automatically set to insufficient - let rest of check run
    
    # Check if we have empty results
    query_result = state.get("query_result", [])
    if not query_result or (isinstance(query_result, list) and len(query_result) == 0):
        # Empty results might be valid (e.g., "no records match your criteria")
        # But the agent should check if human feedback is needed
        result_check_logger.info(f"[node_id={node_id}] Empty results - might need human feedback")
        # Don't automatically set to insufficient - use context
    
    # Check if we have a summary
    if not state.get("summary") and not state.get("is_conversational", False):
        result_check_logger.warning(f"[node_id={node_id}] Missing summary for results")
        # This alone doesn't make results insufficient
    
    result_check_logger.info(f"[node_id={node_id}] Result check complete: results_sufficient={results_sufficient}")
    return {
        "results_sufficient": results_sufficient
    }

async def human_feedback_node(state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Get human feedback on results
    
    Args:
        state: The current graph state
        config: Runnable configuration (optional)
        
    Returns:
        Updated state with human feedback action
    """
    # Create a unique identifier for this node execution
    node_id = id(state)
    human_feedback_logger.info(f"[node_id={node_id}] Starting human_feedback_node execution")
    human_feedback_logger.debug(f"[node_id={node_id}] Input state keys: {list(state.keys())}")
    
    # Prepare feedback message
    feedback_message = "The results of your query may not fully address your question. What would you like to do next?"
    
    # Add context based on what might have gone wrong
    if state.get("has_execution_error", False) or state.get("query_error"):
        error_msg = state.get("query_error", "Unknown execution error")
        feedback_message = f"There was an error executing your query: {error_msg}\n\nWhat would you like to do?"
    elif state.get("visualization_error") and state.get("needs_visualization", True):
        viz_error = state.get("visualization_error", "Unknown visualization error")
        feedback_message = f"Your query executed successfully, but there was an error creating the visualization: {viz_error}\n\nWhat would you like to do?"
    
    human_feedback_logger.info(f"[node_id={node_id}] Requesting human feedback")
    feedback = interrupt({
        "task": "Feedback Needed",
        "message": feedback_message,
        "options": ["Refine SQL Query", "Improve Visualization", "Add More Context", "Proceed with Current Results"]
    })
    human_feedback_logger.debug(f"[node_id={node_id}] Received human feedback: {feedback}")
    
    # Map the human response to an action
    human_feedback = None
    
    if feedback.lower() == "refine sql query":
        human_feedback = "refine_sql"
    elif feedback.lower() == "improve visualization":
        human_feedback = "improve_viz"
    elif feedback.lower() == "add more context":
        human_feedback = "add_context"
    else:  # Proceed with current results
        human_feedback = None  # No change needed
    
    human_feedback_logger.info(f"[node_id={node_id}] Mapped feedback to action: {human_feedback}")
    return {
        "human_feedback": human_feedback
    }

async def explainer(state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Generate explanations for results
    
    Args:
        state: The current graph state
        config: Runnable configuration (optional)
        
    Returns:
        Updated state with explanations
    """
    # Create a unique identifier for this node execution
    node_id = id(state)
    explainer_logger.info(f"[node_id={node_id}] Starting explainer execution")
    explainer_logger.debug(f"[node_id={node_id}] Input state keys: {list(state.keys())}")
    
    # If we have a conversational response, no additional explanation needed
    if state.get("is_conversational", False):
        explainer_logger.info(f"[node_id={node_id}] Skipping explanation for conversational response")
        return {}
    
    # Get necessary state components
    current_question = state.get("current_question", "")
    sql_query = state.get("sql_query", "")
    summary = state.get("summary", "")
    query_result = state.get("query_result", [])
    plotly_figure = state.get("plotly_figure")
    
    # Generate an explanation message
    explanation = ""
    
    if state.get("has_execution_error", False) or state.get("query_error"):
        error_msg = state.get("query_error", "Unknown execution error")
        explanation = f"There was an error executing your query: {error_msg}"
    elif not query_result or (isinstance(query_result, list) and len(query_result) == 0):
        explanation = "Your query returned no results. This could mean that no data matches your criteria."
    else:
        explanation = summary if summary else "Here are the results of your query."
        
        # Add visualization mention if available
        if plotly_figure:
            explanation += "\n\nI've also created a visualization to help understand the data."
    
    explainer_logger.info(f"[node_id={node_id}] Generated explanation: {explanation[:100]}...")
    return {
        "agent_messages": [{
            "role": "assistant",
            "content": explanation
        }]
    }

async def format_response(state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Format the final response, collecting results and errors.
    
    Args:
        state: The final graph state
        config: Runnable configuration (optional)
        
    Returns:
        A dictionary representing the final structured response.
    """
    # Create a unique identifier for this node execution
    node_id = id(state)
    format_logger.info(f"[node_id={node_id}] Starting format_response execution")
    format_logger.debug(f"[node_id={node_id}] Input state keys: {list(state.keys())}")
    
    # Initialize response structure
    response = {
        "conversation_id": state.get("conversation_id"),
        "user_id": state.get("user_id"),
        "question": state.get("current_question", ""),
        "timestamp": time.time(),
        "successful": False,  # Default to unsuccessful until proven otherwise
        "error": None,
        "is_conversational": state.get("is_conversational", False),
        "sql": state.get("sql_query"),
        "data": state.get("dataframe_json"),
        "summary": state.get("summary"),
        "visualization": state.get("plotly_figure"),
        "messages": state.get("agent_messages", [])
    }
    
    # Check for errors
    all_errors = []
    if state.get("node_error"):
        all_errors.append(state.get("node_error"))
    if state.get("extraction_error"):
        all_errors.append(state.get("extraction_error"))
    if state.get("validation_message") and state.get("has_errors", False):
        all_errors.append(state.get("validation_message"))
    if state.get("query_error"):
        all_errors.append(state.get("query_error"))
    if state.get("summary_error"):
        all_errors.append(state.get("summary_error"))
    if state.get("visualization_error"):
        all_errors.append(state.get("visualization_error"))
    
    if all_errors:
        # Join all errors into a single message
        response["error"] = "; ".join(all_errors)
        format_logger.warning(f"[node_id={node_id}] Response contains errors: {response['error']}")
    
    # Mark as successful if we have results (could be SQL or conversational)
    if (state.get("is_conversational", False) and state.get("conversation_answer")) or \
       (state.get("execution_successful", False) and state.get("dataframe_json")):  
        response["successful"] = True
        format_logger.info(f"[node_id={node_id}] Response marked as successful")
    
    # Add conversation answer if conversational
    if state.get("is_conversational", False):
        response["conversation_answer"] = state.get("conversation_answer")
        format_logger.info(f"[node_id={node_id}] Added conversational answer to response")
    
    # Log total execution time
    execution_time = state.get("execution_time", 0)
    format_logger.info(f"[node_id={node_id}] Total execution time: {execution_time:.2f} seconds")
    
    format_logger.info(f"[node_id={node_id}] Completed format_response execution")
    return response
