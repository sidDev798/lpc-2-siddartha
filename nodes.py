"""
LangGraph nodes for DataStory using vanna.ai
"""
from typing import Dict, Any, List, Optional, Union, TypedDict
import time
import io
import json
import pandas as pd
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt
from backend.config.config import vn_instance

from backend.graph.prompts import query_analysis_prompt, human_interruption_query_prompt, intent_classification_prompt
from backend.graph.utils import extract_sql_from_response, check_sql_syntax, check_for_dangerous_operations, is_complex_query, is_high_risk_operation
from dotenv import load_dotenv
import os
load_dotenv()

model = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

# ---- LangGraph Node Functions ----

# Query Analysis
def query_analyzer(state: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the query to determine if clarification is needed"""
    # TODO: Implement actual query analysis logic
    print(f"[DEBUG] query_analyzer - Input state keys: {list(state.keys())}")

    query = state.get("current_question", "")
    conversation_history = state.get("conversation_history", [])

    prompt = query_analysis_prompt.format(query=query, conversation_history=conversation_history)

    response = model.bind(response_format={"type": "json_object"}).invoke(prompt)

    print(f"[DEBUG] query_analyzer - LLM response: {response}")

    if response.content:
        json_response = json.loads(response.content)
        is_clarification_needed = json_response['is_clarification_needed']
        clarification_message = json_response['clarification_message']
    else:
        is_clarification_needed = False
        clarification_message = None
    
    return {
        "is_clarification_needed": is_clarification_needed,
        "clarification_message": clarification_message
    }


# Human Input for Clarification
def human_input_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Get human input for clarification"""
    
    clarification_response = interrupt({
        "task": "Clarification needed",
        "message": state.get("clarification_message", "Please provide more details for your query."),
        "current_question": state.get("current_question", "")
    })

    prompt = human_interruption_query_prompt.format(query=state.get("current_question", ""),
                                                    conversation_history=state.get("conversation_history", ""),
                                                    human_input=clarification_response)

    response = model.bind(response_format={"type": "json_object"}).invoke(prompt)

    print(f"[DEBUG] human_input_node - LLM response: {response}")

    if response.content:
        json_response = json.loads(response.content)
        updated_query = json_response['updated_query']
    else:
        updated_query = state.get("current_question", "")
    
    return {
        "current_question": updated_query
    }

# Intent Classification
def intent_classifier(state: Dict[str, Any]) -> Dict[str, Any]:
    """Classify the intent of the query"""
    # TODO: Implement actual intent classification logic
    intent = "sql_generation"  # Default intent

    prompt = intent_classification_prompt.format(query=state.get("current_question", ""),
                                                conversation_history=state.get("conversation_history", ""))

    response = model.bind(response_format={"type": "json_object"}).invoke(prompt)

    print(f"[DEBUG] intent_classifier - LLM response: {response}")

    if response.content:
        json_response = json.loads(response.content)
        intent = json_response['intent']
    
    return {
        "intent": intent  # One of "sql_generation", "visualization_only", "explanation_only"
    }


def generate_sql(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Submit a prompt to Vanna.ai for SQL generation.
    
    This function retrieves the current question and conversation history,
    generates a prompt using training examples and schema information,
    and calls the enhanced submit_prompt method to generate SQL.
    
    Args:
        state: The current state of the conversation
        
    Returns:
        Dict[str, Any]: Updated state with SQL query and response
    """
    print(f"[DEBUG] generate_sql - Input state keys: {list(state.keys())}")

    # Get components from state
    current_question = state.get("current_question", "")
    conversation_history = state.get("conversation_history", [])
    
    # Generate base prompt with training examples and schema
    base_prompt = vn_instance.get_sql_prompt(
        initial_prompt=f"You are a data scientist. \n\n Conversation History: {conversation_history}",
        question=current_question,
        question_sql_list=vn_instance.get_similar_question_sql(current_question),
        ddl_list=vn_instance.get_related_ddl(current_question),
        doc_list=vn_instance.get_related_documentation(current_question)
    )
    
    # # Add conversation history to the prompt
    # enhanced_prompt = {
    #     "system_message": base_prompt.get("system_message", ""),
    #     "user_message": base_prompt.get("user_message", ""),
    #     "conversation_history": conversation_history
    # }
    
    # # Submit the enhanced prompt to Vanna
    # try:
    #     llm_response = vn_instance.submit_prompt(enhanced_prompt)
    #     sql_query = vn_instance.extract_sql_from_llm_response(llm_response)
    # except Exception as e:
    #     print(f"[ERROR] submit_prompt - Error during Vanna interaction: {e}")
    #     return {**state, "llm_response": None, "sql_query": None, "node_error": f"Vanna API error: {e}"}

    llm_response = vn_instance.submit_prompt(base_prompt)

    print(f"[DEBUG] generate_sql - LLM response: {llm_response}")
    
    # Update state with SQL and response
    return {
        **state,
        "sql_query": llm_response,
        "node_error": None # Clear any previous node error if successful
    }

# Intermediate Check
def intermediate_check(state: Dict[str, Any]) -> Dict[str, Any]:
    """Check if context is sufficient for SQL generation"""
    # TODO: Implement actual context check logic
    if 'intermediate_sql' in state.get("sql_query", ""):
        is_context_sufficient = False
    else:
        is_context_sufficient = True
    return {
        "is_context_sufficient": is_context_sufficient
    }


def extract_sql(state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Extract SQL from the LLM response
    
    Args:
        state: The current graph state
        config: Runnable configuration (optional)
        
    Returns:
        Updated state with extracted SQL
    """
    print(f"[DEBUG] extract_sql - Input state keys: {list(state.keys())}")
    
    sql_query = state.get("sql_query", "") 
    extraction_error = None

    if not sql_query:
        print("[DEBUG] extract_sql - No LLM response found in state.")
        extraction_error = "No LLM response provided for SQL extraction."
        sql_query = f"-- Error: No LLM response to extract SQL from."
    else:
        print("Attempting to extract SQL from LLM response...")
        sql_query, extraction_error = extract_sql_from_response(sql_query) # Use utility function
        
        if not sql_query:

            question = state.get("current_question", "") # Use current_question
            print(f"[DEBUG] extract_sql - Using current_question for fallback message: {question}")
            sql_query = f"-- Unable to extract SQL from response. Question was: {question}"
        else:
            print(f"[DEBUG] extract_sql - Extracted SQL: {sql_query}")

    result = {
        "sql_query": sql_query,
        "extraction_error": extraction_error
    }
    print(f"[DEBUG] extract_sql - Output state keys: {list(result.keys())}")
    return result


# Modify validate_sql to include human-in-the-loop
def validate_sql(state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Validate the generated SQL to ensure it's safe and correct
    
    Args:
        state: The current graph state
        config: Runnable configuration (optional)
        
    Returns:
        Updated state with validation info
    """
    print(f"[DEBUG] validate_sql - Input state keys: {list(state.keys())}")
    sql_query = state.get("sql_query", "")
    print(f"[DEBUG] validate_sql - SQL query to validate: {sql_query}")
    
    # Initialize validation state
    is_valid = True
    validation_message = ""
    has_errors = False
    needs_human_review = False
    
    # Skip validation if SQL is empty
    if not sql_query or sql_query.strip() == "":
        return {
            "sql_valid": False,
            "validation_message": "Empty SQL query",
            "has_errors": True,
            "needs_human_review": True
        }
    
    # 1. Basic syntax validation
    try:
        syntax_errors = check_sql_syntax(sql_query)
        if syntax_errors:
            is_valid = False
            validation_message = f"SQL syntax error: {syntax_errors}"
            has_errors = True
    except Exception as e:
        is_valid = False
        validation_message = f"Error during syntax validation: {str(e)}"
        has_errors = True
    
    # 2. Security validation
    if is_valid:
        dangerous_operations = check_for_dangerous_operations(sql_query)
        if dangerous_operations:
            # This doesn't invalidate the SQL but flags it for review
            needs_human_review = True
            validation_message += f"\nPotentially dangerous operations detected: {dangerous_operations}"
    
    # 3. Complexity analysis
    complex_query = is_complex_query(sql_query)
    if complex_query:
        needs_human_review = True
        validation_message += "\nComplex query detected, human review recommended."
    
    # 4. Risk assessment
    high_risk_operation = is_high_risk_operation(sql_query)
    if high_risk_operation:
        needs_human_review = True
        validation_message += "\nHigh-risk operation detected, human review required."
    
    return {
        "sql_valid": is_valid,
        "validation_message": validation_message if validation_message else None,
        "has_errors": has_errors,
        "needs_human_review": needs_human_review
    }

# Context Enhancer
def context_enhancer(state: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance context for SQL generation"""
    # TODO: Implement actual context enhancement logic
    enhanced_context = state.get("context", "")
    return {
        "context": enhanced_context
    }

# Human SQL Review
def human_sql_review(state: Dict[str, Any]) -> Dict[str, Any]:
    """Get human review for SQL"""
    
    review_response = interrupt({
        "task": "Review SQL query",
        "sql_query": state.get("sql_query", ""),
        "validation_message": state.get("validation_message", ""),
        "options": ["regenerate", "proceed", "cancel"]
    })
    
    if isinstance(review_response, dict):
        action = review_response.get("action")
        edited_sql = review_response.get("edited_sql")
        
        return {
            "human_review_action": action,
            "human_edited_sql": edited_sql if action == "proceed" and edited_sql else None
        }
    
    # Default fallback
    return {
        "human_review_action": "regenerate"
    }

# Error Handler
def error_handler(state: Dict[str, Any]) -> Dict[str, Any]:
    """Handle errors in SQL validation or execution"""
    # TODO: Implement actual error handling logic
    validation_errors = state.get("validation_errors", [])
    return {
        "validation_errors": validation_errors
    }

# Result Processor
def result_processor(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process query results"""
    # TODO: Implement actual result processing logic
    needs_training = False
    needs_visualization = False
    return {
        "needs_training": needs_training,
        "needs_visualization": needs_visualization
    }

# Training Agent
def training_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Add examples to context for training"""
    # TODO: Implement actual training agent logic
    updated_context = state.get("context", "")
    return {
        "context": updated_context
    }

# Visualization Check
def visualization_check(state: Dict[str, Any]) -> Dict[str, Any]:
    """Check if visualization is needed"""
    # TODO: Implement actual visualization check logic
    needs_visualization = False
    return {
        "needs_visualization": needs_visualization
    }

# Result Check
def result_check(state: Dict[str, Any]) -> Dict[str, Any]:
    """Check if results are sufficient"""
    # TODO: Implement actual result check logic
    results_sufficient = True
    return {
        "results_sufficient": results_sufficient
    }

# Human Feedback
def human_feedback_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Get human feedback on results"""
    
    feedback_response = interrupt({
        "task": "Provide feedback on results",
        "query_result": state.get("query_result", []),
        "summary": state.get("summary", ""),
        "visualization": state.get("plotly_figure", None),
        "options": ["refine_sql", "improve_viz", "add_context"]
    })
    
    return {
        "human_feedback": feedback_response
    }

# Explainer
def explainer(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate explanations for results"""
    # TODO: Implement actual explanation generation logic
    explanation = "This is a placeholder explanation."
    return {
        "agent_messages": [{"role": "assistant", "content": explanation}]
    }



def run_sql(state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Execute the SQL query if it's valid
    
    Args:
        state: The current graph state
        config: Runnable configuration (optional)
        
    Returns:
        Updated state with query results
    """
    print(f"[DEBUG] run_sql - Input state keys: {list(state.keys())}")
    sql_query = state.get("sql_query", "")
    print(f"[DEBUG] run_sql - SQL query to execute: '{sql_query}'")
    
    # Skip execution if the SQL is invalid
    if not state.get("sql_valid", False):
        print(f"[DEBUG] run_sql - Skipping execution because SQL is invalid")
        error_message = state.get("validation_message", "Invalid SQL query")
        print(f"[DEBUG] run_sql - Error message: {error_message}")
        
        return {
            "query_result": None,
            "query_error": error_message,
            "result_dataframe": None
        }
    
    try:
        # Execute the SQL query using the globally configured Vanna instance
        print(f"[DEBUG] run_sql - Executing SQL query...")
        start_time = time.time()
        result_df = vn_instance.run_sql(sql_query)
        execution_time = time.time() - start_time
        print(f"[DEBUG] run_sql - Query executed in {execution_time:.2f} seconds")
        
        # Convert DataFrame to JSON-serializable formats for API response
        # Convert DataFrame to dict records for API response
        query_result = None
        if isinstance(result_df, pd.DataFrame):
            print(f"[DEBUG] run_sql - Got DataFrame with shape: {result_df.shape}")
            # Store the full result as records for API consumption
            query_result = result_df.to_dict('records')
            print(f"[DEBUG] run_sql - Converted to {len(query_result)} records")
            
            # Create a serializable version of the DataFrame
            df_json = {}
            df_json['records'] = result_df.to_dict('records')
            df_json['columns'] = result_df.columns.tolist()
            df_json['index'] = result_df.index.tolist()
            df_json['dtypes'] = result_df.dtypes.astype(str).to_dict()
            df_json['shape'] = result_df.shape
            
            # Save the original DataFrame for internal processing
            print(f"[DEBUG] run_sql - Created JSON-serializable DataFrame format")
            
            if len(query_result) > 0:
                print(f"[DEBUG] run_sql - Sample result: {query_result[0] if query_result else 'N/A'}")
        else:
            print(f"[DEBUG] run_sql - Result is not a DataFrame, type: {type(result_df)}")

            query_result = []
            df_json = None
            result_df = pd.DataFrame() # Ensure result_dataframe is always a DataFrame

            
        result = {
            "query_result": query_result,
            "query_error": None,
            "result_dataframe": result_df,
            "dataframe_json": df_json,  # Add the JSON-serializable version
            "execution_time": execution_time
        }
        print(f"[DEBUG] run_sql - Returning success result with {len(query_result) if query_result else 0} records")
        return result
    except Exception as e:
        error_message = f"Error executing SQL: {str(e)}"
        print(f"[DEBUG] run_sql - ERROR: {error_message}")
        
        result = {
            "query_result": None,
            "query_error": error_message,
            "result_dataframe": None
        }
        print(f"[DEBUG] run_sql - Returning error result")
        return result


def generate_summary(state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Generate natural language summary from query results
    
    Args:
        state: The current graph state
        config: Runnable configuration (optional)
        
    Returns:
        Updated state with summary
    """
    print(f"[DEBUG] generate_summary - Input state keys: {list(state.keys())}")
    summary = ""
    summary_error = None
    
    try:
        # Only generate summary if we have results and no query error
        print(f"[DEBUG] generate_summary - Checking if we have results to summarize")
        result_df = state.get("result_dataframe")
        query_error = state.get("query_error")

        if isinstance(result_df, pd.DataFrame) and not query_error:
            print(f"[DEBUG] generate_summary - DataFrame shape: {result_df.shape}")
            
            if not result_df.empty:
                print(f"[DEBUG] generate_summary - DataFrame is not empty, generating summary")
                # Get the question for context
                question = state.get("current_question", "") # Use current_question
                print(f"[DEBUG] generate_summary - Question: {question}")
                
                # Generate summary using Vanna
                print(f"[DEBUG] generate_summary - Calling Vanna's generate_summary")
                summary = vn_instance.generate_summary(
                    question=question,
                    df=result_df
                )
                print(f"[DEBUG] generate_summary - Generated summary (first 100 chars): {summary[:100]}...")
            else:
                summary = "The query executed successfully but returned no results."
                print(f"[DEBUG] generate_summary - DataFrame is empty, providing default summary")
        else:
            print(f"[DEBUG] generate_summary - No valid results to summarize")
            if query_error:
                summary = f"Could not generate a summary because the query failed: {query_error}"
                print(f"[DEBUG] generate_summary - Query error prevents summary: {query_error}")
            else:
                summary = "No data available to generate a summary."
                print(f"[DEBUG] generate_summary - No DataFrame available for summary.")
                
    except Exception as e:
        summary_error = f"Error generating summary: {str(e)}"
        summary = f"An error occurred while generating the summary: {summary_error}"
        print(f"[DEBUG] generate_summary - ERROR: {summary_error}")
    
    return {
        "summary": summary,
        "summary_error": summary_error
    }

def generate_plotly_code(state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Generate Plotly visualization code for query results
    
    Args:
        state: The current graph state
        config: Runnable configuration (optional)
        
    Returns:
        Updated state with visualization code
    """
    print(f"[DEBUG] generate_plotly_code - Input state keys: {list(state.keys())}")
    plotly_code = None
    visualization_error = None
    
    try:
        # Only generate visualization if we have results and no query error
        print(f"[DEBUG] generate_plotly_code - Checking if we have results to visualize")
        result_df = state.get("result_dataframe")
        query_error = state.get("query_error")

        if isinstance(result_df, pd.DataFrame) and not query_error:
            print(f"[DEBUG] generate_plotly_code - DataFrame shape: {result_df.shape}")
            
            # Only generate visualization if dataframe has data
            if not result_df.empty:
                print(f"[DEBUG] generate_plotly_code - DataFrame has data, generating visualization")
                question = state.get("current_question", "") # Use current_question
                sql = state.get("sql_query", "")
                print(f"[DEBUG] generate_plotly_code - Question: {question}")
                print(f"[DEBUG] generate_plotly_code - SQL: {sql}")
                
                # Generate Plotly code
                print(f"[DEBUG] generate_plotly_code - Calling Vanna's generate_plotly_code")
                # Get DataFrame info as string
                buffer = io.StringIO()
                result_df.info(buf=buffer)
                df_metadata = buffer.getvalue()
                print(f"[DEBUG] generate_plotly_code - DataFrame metadata length: {len(df_metadata)}")
                
                plotly_code = vn_instance.generate_plotly_code(
                    question=question,
                    sql=sql,
                    df_metadata=df_metadata
                )
                print(f"[DEBUG] generate_plotly_code - Generated code length: {len(plotly_code) if plotly_code else 0}")
            else:
                print(f"[DEBUG] generate_plotly_code - DataFrame is empty, skipping visualization")
        else:
            print(f"[DEBUG] generate_plotly_code - No valid results to visualize")
            if query_error:
                print(f"[DEBUG] generate_plotly_code - Query error prevents visualization: {query_error}")
            else:
                 print(f"[DEBUG] generate_plotly_code - No DataFrame available for visualization.")

    except Exception as e:
        visualization_error = f"Error generating visualization code: {str(e)}"
        print(f"[DEBUG] generate_plotly_code - ERROR: {visualization_error}")
    
    return {
        "plotly_code": plotly_code,
        "visualization_error": visualization_error
    }


def get_plotly_figure(state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Convert Plotly code to an actual Plotly figure
    
    Args:
        state: The current graph state
        config: Runnable configuration (optional)
        
    Returns:
        Updated state with Plotly figure
    """
    print(f"[DEBUG] get_plotly_figure - Input state keys: {list(state.keys())}")
    plotly_figure = None
    # Carry over any previous visualization error
    visualization_error = state.get("visualization_error") 
    
    try:
        # Only generate figure if we have code, dataframe, and no query error
        plotly_code = state.get("plotly_code")
        result_df = state.get("result_dataframe")
        query_error = state.get("query_error")
        print(f"[DEBUG] get_plotly_figure - Plotly code length: {len(plotly_code) if plotly_code else 0}")
        
        if plotly_code and isinstance(result_df, pd.DataFrame) and not query_error:
            print(f"[DEBUG] get_plotly_figure - DataFrame shape: {result_df.shape}")
            
            # Only generate figure if dataframe has data and we have plotly code
            if not result_df.empty:
                print(f"[DEBUG] get_plotly_figure - Creating Plotly figure from code")
                
                # Use Vanna's get_plotly_figure function
                try:
                    plotly_figure = vn_instance.get_plotly_figure(plotly_code=plotly_code, df=result_df)
                    print(f"[DEBUG] get_plotly_figure - Successfully created figure object")
                except Exception as e:
                    # Don't overwrite previous error unless this is a new one
                    if not visualization_error: 
                        visualization_error = f"Error creating Plotly figure from code: {str(e)}"
                    print(f"[DEBUG] get_plotly_figure - ERROR creating figure: {visualization_error}")
            else:
                print(f"[DEBUG] get_plotly_figure - Skipping figure creation (empty data)")
        else:
            if not plotly_code:
                 print(f"[DEBUG] get_plotly_figure - Skipping figure creation (no code)")
            if not isinstance(result_df, pd.DataFrame):
                 print(f"[DEBUG] get_plotly_figure - Skipping figure creation (no DataFrame)")
            if query_error:
                 print(f"[DEBUG] get_plotly_figure - Skipping figure creation (query error)")
            
    except Exception as e:
        # Catch any unexpected errors during the process
        if not visualization_error: 
             visualization_error = f"Unexpected error handling Plotly figure creation: {str(e)}"
        print(f"[DEBUG] get_plotly_figure - ERROR: {visualization_error}")
    
    # Return the figure (can be None) and any accumulated error
    return {
        "plotly_figure": plotly_figure,
        "visualization_error": visualization_error
    }
