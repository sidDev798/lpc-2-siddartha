from fastapi import FastAPI, HTTPException, Body, Request, Response, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Callable, Dict, Any
import uvicorn
import logging
import sys
import traceback
import time
import json
import re
# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('backend_debug.log')
    ]
)
logger = logging.getLogger(__name__)

# Create API logger for request/response logging
api_logger = logging.getLogger("api")
api_logger.setLevel(logging.INFO)
api_file_handler = logging.FileHandler('api_requests.log')
api_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
api_logger.addHandler(api_file_handler)

# Import the run_python_assistant function from python_agent.py
from .python_agent import run_python_assistant

# Create FastAPI app
app = FastAPI(title="Python Code Assistant API")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class ChatRequest(BaseModel):
    message: str
    file_content: Optional[str] = None
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "Generate a Fibonacci sequence function",
                    "file_content": None
                }
            ]
        }
    }

class ChatResponse(BaseModel):
    text: str
    code: Optional[str] = None

# # Middleware to log requests and responses
# @app.middleware("http")
# async def log_requests(request: Request, call_next: Callable) -> Response:
#     request_id = str(time.time())
#     request_path = request.url.path
    
#     # Don't log health check endpoints
#     if request_path == "/":
#         return await call_next(request)
    
#     # Don't attempt to read the body for POST /chat requests as we'll handle that in the route
#     if request_path == "/chat" and request.method == "POST":
#         logger.debug("Skipping body logging for /chat POST request")
#         response = await call_next(request)
#         return response
    
#     # For other requests, log the body if possible
#     try:
#         # Only attempt to read and log the body for non-streaming requests
#         if request.headers.get("content-type") == "application/json":
#             # Create a copy of the request to read the body
#             body_bytes = await request.body()
            
#             # Create a new request with the same body
#             async def receive():
#                 return {"type": "http.request", "body": body_bytes}
            
#             request = Request(request.scope, receive)
            
#             # Log request body (parsed as JSON if possible)
#             try:
#                 body = json.loads(body_bytes)
#                 # Don't log full file content to keep logs manageable
#                 if 'file_content' in body and body['file_content']:
#                     file_content_length = len(body['file_content'])
#                     body['file_content'] = f"[{file_content_length} characters]"
#                 api_logger.info(f"REQUEST {request_id} | {request.method} {request_path} | {json.dumps(body)}")
#             except:
#                 # If can't parse as JSON, log as is (truncated)
#                 body_str = body_bytes.decode('utf-8', errors='replace')
#                 if len(body_str) > 200:
#                     body_str = body_str[:200] + "... [truncated]"
#                 api_logger.info(f"REQUEST {request_id} | {request.method} {request_path} | {body_str}")
#         else:
#             api_logger.info(f"REQUEST {request_id} | {request.method} {request_path} | Non-JSON body")
#     except Exception as e:
#         api_logger.error(f"Error logging request: {str(e)}")
    
#     # Process request
#     start_time = time.time()
#     response = await call_next(request)
#     process_time = time.time() - start_time
    
#     # Log response
#     response_body = b""
#     async for chunk in response.body_iterator:
#         response_body += chunk
    
#     # Log response details
#     try:
#         if response_body:
#             try:
#                 response_json = json.loads(response_body)
#                 # Truncate long responses for log readability
#                 if 'text' in response_json and len(response_json['text']) > 500:
#                     response_json['text'] = response_json['text'][:500] + "... [truncated]"
#                 if 'code' in response_json and response_json['code'] and len(response_json['code']) > 500:
#                     response_json['code'] = response_json['code'][:500] + "... [truncated]"
#                 api_logger.info(f"RESPONSE {request_id} | Status: {response.status_code} | Time: {process_time:.2f}s | {json.dumps(response_json)}")
#             except:
#                 # If can't parse as JSON, log raw (truncated)
#                 body_str = response_body.decode('utf-8', errors='replace')
#                 if len(body_str) > 200:
#                     body_str = body_str[:200] + "... [truncated]"
#                 api_logger.info(f"RESPONSE {request_id} | Status: {response.status_code} | Time: {process_time:.2f}s | {body_str}")
#         else:
#             api_logger.info(f"RESPONSE {request_id} | Status: {response.status_code} | Time: {process_time:.2f}s | Empty Body")
#     except Exception as e:
#         api_logger.error(f"Error logging response: {str(e)}")
    
#     # Reconstruct response
#     return Response(
#         content=response_body,
#         status_code=response.status_code,
#         headers=dict(response.headers),
#         media_type=response.media_type
#     )

@app.get("/")
async def read_root():
    logger.info("Root endpoint accessed")
    return {"message": "Python Code Assistant API is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: Request):
    """
    Process a chat request and generate a response by manually parsing the request body.
    
    Args:
        request: The raw HTTP request
    """
    try:
        # Manually parse the request body
        body_bytes = await request.body()
        
        if not body_bytes:
            raise HTTPException(status_code=400, detail="Empty request body")
        
        # Parse JSON
        try:
            body = json.loads(body_bytes)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
        
        # Extract message and file_content
        if 'message' not in body:
            raise HTTPException(status_code=400, detail="Missing 'message' field in request")
        
        message = body['message']
        file_content = body.get('file_content')
        
        logger.info(f"Chat endpoint accessed with message: {message[:50]}...")
        logger.debug(f"Full chat request: message={message}, has_code={file_content is not None}")
        
        # Process the request using run_python_assistant
        logger.info("Calling run_python_assistant...")
        result = run_python_assistant(
            query=message,
            code=file_content
        )
        
        logger.info("Successfully processed chat request")
        logger.debug(f"Response text length: {len(result['text'])}, has_code: {result['code'] is not None}")

        # Remove the code block from the text if it exists
        if result['code'] is not None:
            # Find code block in textusing regex and remove code block
            result['text'] = re.sub(r'```python\s*[\s\S]*?```', '', result['text'])
        
        return ChatResponse(
            text=result["text"],
            code=result["code"]
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        logger.error(f"Error processing chat request: {error_msg}")
        logger.error(f"Stack trace: {stack_trace}")
        raise HTTPException(status_code=500, detail=error_msg)

# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 