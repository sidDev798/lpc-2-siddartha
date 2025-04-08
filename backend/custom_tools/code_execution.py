import os
import subprocess
import tempfile
import shutil
import sys
import venv
import logging
# Import the Tool class from langchain_core.tools
from langchain_core.tools import tool

# Configure logging
logger = logging.getLogger(__name__)


@tool
def run_python_code(code_string, requirements=None):
    """
    Execute Python code in a virtual environment.
    
    Args:
        code_string (str): Python code to execute
        requirements (list, optional): List of pip packages to install
        
    Returns:
        dict: Dictionary containing stdout, stderr, and return code
    """
    logger.info("Starting Python code execution")
    logger.debug(f"Code length: {len(code_string)}")
    if requirements:
        logger.debug(f"Requirements: {requirements}")
    
    # Create a temporary directory for the virtual environment and code
    temp_dir = tempfile.mkdtemp()
    venv_dir = os.path.join(temp_dir, 'venv')
    temp_file_path = os.path.join(temp_dir, 'script.py')
    
    logger.debug(f"Created temp directory: {temp_dir}")
    logger.debug(f"Venv directory: {venv_dir}")
    logger.debug(f"Script path: {temp_file_path}")
    
    try:
        # Create a virtual environment
        logger.debug("Creating virtual environment")
        venv.create(venv_dir, with_pip=True)
        
        # Determine path to Python and pip executables in the virtual environment
        if sys.platform == 'win32':
            python_exe = os.path.join(venv_dir, 'Scripts', 'python.exe')
            pip_exe = os.path.join(venv_dir, 'Scripts', 'pip.exe')
        else:
            python_exe = os.path.join(venv_dir, 'bin', 'python')
            pip_exe = os.path.join(venv_dir, 'bin', 'pip')
        
        logger.debug(f"Python executable: {python_exe}")
        logger.debug(f"Pip executable: {pip_exe}")
        
        # Install requirements if provided
        if requirements:
            logger.info(f"Installing {len(requirements)} packages")
            try:
                pip_process = subprocess.run(
                    [pip_exe, 'install', '-q'] + requirements,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                logger.debug(f"Pip install completed with return code: {pip_process.returncode}")
                if pip_process.stderr:
                    logger.warning(f"Pip stderr: {pip_process.stderr}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install packages: {e}")
                logger.error(f"Pip stderr: {e.stderr}")
                return {
                    'stdout': '',
                    'stderr': f"Failed to install packages: {e.stderr}",
                    'return_code': e.returncode
                }
        
        # Write the code to a file
        logger.debug("Writing code to file")
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            f.write(code_string)
        
        # Execute the file in the virtual environment
        logger.info("Executing Python code")
        result = subprocess.run(
            [python_exe, temp_file_path],
            capture_output=True,
            text=True
        )
        
        logger.info(f"Code execution completed with return code: {result.returncode}")
        logger.debug(f"stdout length: {len(result.stdout)}")
        if result.stderr:
            logger.warning(f"stderr: {result.stderr}")
        
        return {
            'stdout': result.stdout,
            'stderr': result.stderr,
            'return_code': result.returncode
        }
        
    except Exception as e:
        logger.error(f"Error in code execution: {str(e)}", exc_info=True)
        return {
            'stdout': '',
            'stderr': f"Error setting up or running code: {str(e)}",
            'return_code': 1
        }
    finally:
        # Clean up the temporary directory and virtual environment
        logger.debug(f"Cleaning up temp directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)