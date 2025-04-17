"""
Main LangGraph definition for the DataStory application (Combined Version)
"""
from typing import Dict, Any, TypedDict, List, Optional, Annotated, Literal
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from backend.config.config import get_vanna_instance
from backend.config.logging_config import get_logger
from backend.graph.combined_nodes import *  # Import all node functions from combined_nodes

# Create logger for the graph
graph_logger = get_logger("datastory.graph")

# Get Vanna instance
vn = get_vanna_instance()

# Define the enhanced state for type checking by extending MessagesState
class State(MessagesState):
    # MessagesState already includes 'messages: list'
    
    # Input and Query Analysis
    conversation_id: str
    user_id: str  # Added user_id tracking
    current_question: str  # The current user query
    conversation_history: Annotated[list, add_messages]  # List of messages in the conversation
    is_clarification_needed: bool = False  # Flag for whether clarification is needed
    clarification_message: Optional[str] = None  # Message explaining what clarification is needed
    
    # Intent Classification
    intent: Literal["sql_generation", "visualization_only", "explanation_only"] = "sql_generation"
    
    # Context Management
    context: Annotated[Dict[str, Any], "merge"] = {}  # Context information for SQL generation
    is_context_sufficient: bool = True  # Flag for whether context is sufficient
    
    # SQL Generation and Validation
    llm_response: Optional[str] = None  # Raw response from LLM
    sql_query: Optional[str] = None  # Final SQL query to execute
    extraction_error: Optional[str] = None  # Error during SQL extraction
    sql_valid: bool = False  # Flag for whether SQL is valid
    validation_message: Optional[str] = None  # Validation message
    has_errors: bool = False  # Flag for whether there are errors
    needs_human_review: bool = False  # Flag for whether human review is needed
    human_review_action: Optional[Literal["regenerate", "proceed", "cancel"]] = None  # Human review action
    human_edited_sql: Optional[str] = None  # SQL edited by human
    node_error: Optional[str] = None  # General error from a node
    
    # Conversational Handling
    is_conversational: bool = False  # Flag for non-SQL answers
    conversation_answer: Optional[str] = None  # Stores the direct answer for conversational replies
    
    # Query Execution
    query_result: Optional[List[Dict[str, Any]]] = None  # Query results
    query_error: Optional[str] = None  # Error during query execution
    dataframe_json: Optional[Dict[str, Any]] = None  # JSON serializable version for state/API
    execution_time: Optional[float] = None  # Execution time
    
    # Result Processing
    needs_training: bool = False  # Flag for whether training is needed
    needs_visualization: bool = False  # Flag for whether visualization is needed
    results_sufficient: bool = True  # Flag for whether results are sufficient
    
    # Summary and Visualization - Annotated for concurrent updates
    summary: Annotated[Optional[str], "override"] = None  # From generate_summary
    summary_error: Annotated[Optional[str], "last_value"] = None  # From generate_summary
    plotly_code: Annotated[Optional[str], "override"] = None  # From generate_plotly_code
    plotly_figure: Annotated[Optional[Dict], "override"] = None  # From get_plotly_figure
    visualization_error: Annotated[Optional[str], "last_value"] = None  # For visualization errors
    
    # Human Feedback
    human_feedback: Optional[Literal["refine_sql", "improve_viz", "add_context"]] = None  # Human feedback
    
    # Output
    agent_messages: Annotated[List[Dict[str, Any]], "append"] = []  # Messages from the agent


# Create the graph
def create_graph():
    """
    Create the enhanced DataStory graph
    
    Returns:
        The compiled graph
    """
    graph_logger.info("Creating DataStory graph")
    
    # Initialize a new graph
    graph = StateGraph(State)
    
    # Add all nodes from old_graph.py (about 20 nodes)
    graph_logger.debug("Adding nodes to graph")
    graph.add_node("query_analyzer", query_analyzer)
    graph.add_node("human_input", human_input_node)
    graph.add_node("intent_classifier", intent_classifier)
    graph.add_node("submit_prompt", submit_prompt)
    graph.add_node("intermediate_check", intermediate_check)
    graph.add_node("context_enhancer", context_enhancer)
    graph.add_node("extract_sql", extract_sql)
    graph.add_node("validate_sql", validate_sql) 
    graph.add_node("human_sql_review", human_sql_review)
    graph.add_node("error_handler", error_handler)
    graph.add_node("run_sql", run_sql)
    graph.add_node("result_processor", result_processor)
    graph.add_node("training_agent", training_agent)
    graph.add_node("visualization_check", visualization_check)
    graph.add_node("generate_summary", generate_summary)  
    graph.add_node("generate_plotly_code", generate_plotly_code) 
    graph.add_node("get_plotly_figure", get_plotly_figure) 
    graph.add_node("result_check", result_check)
    graph.add_node("human_feedback_node", human_feedback_node)
    graph.add_node("explainer", explainer)
    graph.add_node("format_response", format_response)
    
    # Define the flow - Start with query analysis
    graph_logger.debug("Setting up graph edges")
    graph.add_edge(START, "query_analyzer")
    
    # Conditional edge from query analyzer
    def route_after_query_analysis(state: Dict[str, Any]) -> str:
        graph_logger.debug(f"route_after_query_analysis - is_clarification_needed: {state.get('is_clarification_needed', False)}")
        if state.get("is_clarification_needed", False):
            return "human_input"
        return "intent_classifier"
    
    graph.add_conditional_edges("query_analyzer", route_after_query_analysis, {
        "human_input": "human_input",
        "intent_classifier": "intent_classifier"
    })
    
    # Human input goes to intent classifier
    graph.add_edge("human_input", "intent_classifier")

    # Conditional edge from intent classifier
    def route_after_intent_classification(state: Dict[str, Any]) -> str:
        intent = state.get("intent", "sql_generation")
        graph_logger.debug(f"route_after_intent_classification - intent: {intent}")
        if intent == "sql_generation":
            return "submit_prompt"
        elif intent == "visualization_only":
            return "visualization_check"
        elif intent == "explanation_only":
            return "explainer"
        return "submit_prompt"  # Default
    
    graph.add_conditional_edges("intent_classifier", route_after_intent_classification, {
        "submit_prompt": "submit_prompt",
        "visualization_only": "visualization_check",
        "explanation_only": "explainer"
    })
    
    # SQL generation path
    graph.add_edge("submit_prompt", "extract_sql")
    graph.add_edge("extract_sql", "intermediate_check")
    
    # Conditional edge from intermediate check
    def route_after_intermediate_check(state: Dict[str, Any]) -> str:
        graph_logger.debug(f"route_after_intermediate_check - is_context_sufficient: {state.get('is_context_sufficient', True)}")
        if state.get("is_context_sufficient", True):
            return "validate_sql"
        return "context_enhancer"
    
    graph.add_conditional_edges("intermediate_check", route_after_intermediate_check, {
        "validate_sql": "validate_sql",
        "context_enhancer": "context_enhancer"
    })
    
    # Context enhancer loops back to SQL generation
    graph.add_edge("context_enhancer", "submit_prompt")
    
    # Conversational response handling
    def route_after_extract_sql(state: Dict[str, Any]) -> str:
        graph_logger.debug(f"route_after_extract_sql - is_conversational: {state.get('is_conversational', False)}")
        if state.get("is_conversational", False):
            # Skip SQL processing for conversational responses
            return "format_response"
        return "validate_sql"
    
    graph.add_conditional_edges("extract_sql", route_after_extract_sql, {
        "format_response": "format_response",
        "validate_sql": "validate_sql"
    })
    
    # Conditional edge from SQL validation
    def route_after_validation(state: Dict[str, Any]) -> str:
        graph_logger.debug(f"route_after_validation - has_errors: {state.get('has_errors', False)}, needs_human_review: {state.get('needs_human_review', False)}")
        if state.get("has_errors", False):
            return "error_handler"
        elif state.get("needs_human_review", False):
            return "human_sql_review"
        return "run_sql"
    
    graph.add_conditional_edges("validate_sql", route_after_validation, {
        "error_handler": "error_handler",
        "human_sql_review": "human_sql_review",
        "run_sql": "run_sql"
    })
    
    # Error handler goes back to SQL generation
    graph.add_edge("error_handler", "submit_prompt")
    
    # Conditional edge from human SQL review
    def route_after_human_review(state: Dict[str, Any]) -> str:
        action = state.get("human_review_action", "regenerate")
        graph_logger.debug(f"route_after_human_review - action: {action}")
        if action == "regenerate":
            return "submit_prompt"
        elif action == "proceed":
            return "run_sql"
        return END  # Cancel
    
    graph.add_conditional_edges("human_sql_review", route_after_human_review, {
        "submit_prompt": "submit_prompt",
        "run_sql": "run_sql",
        END: END
    })
    
    # After running SQL, branch to multiple nodes in parallel (from graph.py)
    def branch_after_run_sql(state: Dict[str, Any]) -> List[str]:
        graph_logger.debug(f"branch_after_run_sql - checking results: dataframe_json exists: {state.get('dataframe_json') is not None}, query_error: {state.get('query_error')}")
        
        # Check if there was an execution error
        if state.get("has_execution_error", False):
            return ["error_handler"]
        
        # If we have valid results, proceed to result processor
        if state.get("dataframe_json") is not None and not state.get("query_error"):
            return ["result_processor"]
        
        # If we have an error, go to error handler
        return ["error_handler"]
    
    graph.add_conditional_edges("run_sql", branch_after_run_sql)
    
    # Conditional edge from result processor
    def route_after_result_processing(state: Dict[str, Any]) -> List[str]:
        graph_logger.debug(f"route_after_result_processing - needs_training: {state.get('needs_training', False)}")
        
        paths = []
        
        # Add training_agent if needed
        if state.get("needs_training", False):
            paths.append("training_agent")
            
        # Always process both summary and visualization in parallel
        paths.extend(["generate_summary", "visualization_check"])
        
        return paths
    
    graph.add_conditional_edges("result_processor", route_after_result_processing)
    
    # Training agent paths to visualization check
    graph.add_edge("training_agent", "result_check")
    
    # Conditional edge from visualization check
    def route_after_visualization_check(state: Dict[str, Any]) -> str:
        graph_logger.debug(f"route_after_visualization_check - needs_visualization: {state.get('needs_visualization', True)}")
        if state.get("needs_visualization", True):
            return "generate_plotly_code"
        return "result_check"
    
    graph.add_conditional_edges("visualization_check", route_after_visualization_check, {
        "generate_plotly_code": "generate_plotly_code",
        "result_check": "result_check"
    })

    # Conditional edge from plotly code generation
    def route_after_plotly_code(state: Dict[str, Any]) -> str:
        graph_logger.debug(f"route_after_plotly_code - Checking if we have Plotly code: {bool(state.get('plotly_code'))}")
        if state.get("plotly_code"):
            return "get_plotly_figure"
        return "result_check"
    
    # Visualization generation path
    graph.add_conditional_edges("generate_plotly_code", route_after_plotly_code, {
        "get_plotly_figure": "get_plotly_figure",
        "result_check": "result_check"
    })
    
    # Plotly figure goes to result check
    graph.add_edge("get_plotly_figure", "result_check")
    
    # Summary goes to result check directly
    graph.add_edge("generate_summary", "result_check")
    
    # Conditional edge from result check
    def route_after_result_check(state: Dict[str, Any]) -> str:
        graph_logger.debug(f"route_after_result_check - results_sufficient: {state.get('results_sufficient', True)}")
        if state.get("results_sufficient", True):
            return "explainer"
        return "human_feedback_node"
    
    graph.add_conditional_edges("result_check", route_after_result_check, {
        "explainer": "explainer",
        "human_feedback_node": "human_feedback_node"
    })
    
    # Conditional edge from human feedback
    def route_after_human_feedback(state: Dict[str, Any]) -> str:
        feedback = state.get("human_feedback", "refine_sql")
        graph_logger.debug(f"route_after_human_feedback - feedback: {feedback}")
        if feedback == "refine_sql":
            return "submit_prompt"
        elif feedback == "improve_viz":
            return "generate_plotly_code"
        elif feedback == "add_context":
            return "context_enhancer"
        return "explainer"  # Default
    
    graph.add_conditional_edges("human_feedback_node", route_after_human_feedback, {
        "submit_prompt": "submit_prompt",
        "generate_plotly_code": "generate_plotly_code",
        "context_enhancer": "context_enhancer",
        "explainer": "explainer"
    })
    
    # Explainer goes to format_response
    graph.add_edge("explainer", "format_response")
    
    # Format response node is the final node that standardizes output
    graph.add_edge("format_response", END)
    
    # Compile the graph with memory checkpointing from graph.py
    graph_logger.info("Compiling graph with memory checkpointing")
    memory = MemorySaver()
    graph_run = graph.compile(checkpointer=memory)
    return graph_run


# Create a singleton instance of the graph
datastory_graph = create_graph()  # Keeping the variable name for backward compatibility
