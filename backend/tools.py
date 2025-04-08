import subprocess
import tempfile
import os
import sys
import shutil
import venv


# Test with a complex script
def test_complex_script():
    code = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json
import time

# Print environment info
print(f"Python version: {sys.version}")
print(f"pandas version: {pd.__version__}")
print(f"numpy version: {np.__version__}")
print(f"scikit-learn version: {sklearn.__version__}")

# Generate synthetic data
print("\\nGenerating synthetic dataset...")
np.random.seed(42)
n_samples = 1000
n_features = 5

X = np.random.randn(n_samples, n_features)
# Create a non-linear classification task
y = (np.sin(X[:, 0]) + np.cos(X[:, 1]) + X[:, 2]**2 + X[:, 3] - X[:, 4] > 0).astype(int)

# Create a DataFrame
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
df['target'] = y

print(f"Dataset shape: {df.shape}")
print("\\nData sample:")
print(df.head())

# Data statistics
print("\\nData statistics:")
print(df.describe())

# Create a visualization
print("\\nCreating visualization...")
plt.figure(figsize=(10, 6))
for i in range(n_features):
    plt.subplot(2, 3, i+1)
    for target in [0, 1]:
        plt.hist(df[df['target'] == target][f'feature_{i}'], 
                 alpha=0.5, 
                 bins=20, 
                 label=f'Class {target}')
    plt.title(f'Feature {i} distribution')
    plt.legend()
plt.tight_layout()
plt.savefig('feature_distribution.png')
print("Visualization saved to file (not displayed in console output)")

# Split data for modeling
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('target', axis=1), df['target'], test_size=0.3, random_state=42
)

# Train a model
print("\\nTraining Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
start_time = time.time()
model.fit(X_train, y_train)
training_time = time.time() - start_time

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")
print(f"Training time: {training_time:.2f} seconds")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': df.drop('target', axis=1).columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\\nFeature importance:")
print(feature_importance)

# Make an API request
print("\\nMaking API request to httpbin.org...")
try:
    response = requests.get('https://httpbin.org/get', params={'query': 'test'})
    print(f"API Response status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"API request failed: {str(e)}")

# Complex calculation
print("\\nPerforming complex matrix operations...")
matrix_a = np.random.randn(100, 100)
matrix_b = np.random.randn(100, 100)
start_time = time.time()
result = np.dot(matrix_a, matrix_b)
calc_time = time.time() - start_time
print(f"Matrix multiplication shape: {result.shape}")
print(f"Matrix multiplication time: {calc_time:.4f} seconds")
print(f"Matrix sum: {np.sum(result):.4f}")

print("\\nComplex script execution completed successfully!")
"""
    
    # List of required packages
    requirements = [
        "pandas",
        "numpy",
        "matplotlib", 
        "scikit-learn",
        "requests"
    ]
    
    # Run the code
    print("Running complex test script...")
    result = run_python_code(code, requirements=requirements)
    
    # Print results
    if result['return_code'] == 0:
        print("\n--- Script executed successfully ---")
        print(result['stdout'])
    else:
        print("\n--- Script execution failed ---")
        print(f"Error: {result['stderr']}")
    
    return result


def run_python_code(code_string, requirements=None):
    """
    Execute Python code in a virtual environment.
    
    Args:
        code_string (str): Python code to execute
        requirements (list, optional): List of pip packages to install
        
    Returns:
        dict: Dictionary containing stdout, stderr, and return code
    """
    # Create a temporary directory for the virtual environment and code
    temp_dir = tempfile.mkdtemp()
    venv_dir = os.path.join(temp_dir, 'venv')
    temp_file_path = os.path.join(temp_dir, 'script.py')
    
    try:
        # Create a virtual environment
        venv.create(venv_dir, with_pip=True)
        
        # Determine path to Python and pip executables in the virtual environment
        if sys.platform == 'win32':
            python_exe = os.path.join(venv_dir, 'Scripts', 'python.exe')
            pip_exe = os.path.join(venv_dir, 'Scripts', 'pip.exe')
        else:
            python_exe = os.path.join(venv_dir, 'bin', 'python')
            pip_exe = os.path.join(venv_dir, 'bin', 'pip')
        
        # Install requirements if provided
        if requirements:
            subprocess.run(
                [pip_exe, 'install', '-q'] + requirements,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        
        # Write the code to a file
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            f.write(code_string)
        
        # Execute the file in the virtual environment
        result = subprocess.run(
            [python_exe, temp_file_path],
            capture_output=True,
            text=True
        )
        
        return {
            'stdout': result.stdout,
            'stderr': result.stderr,
            'return_code': result.returncode
        }
    except Exception as e:
        return {
            'stdout': '',
            'stderr': f"Error setting up or running code: {str(e)}",
            'return_code': 1
        }
    finally:
        # Clean up the temporary directory and virtual environment
        shutil.rmtree(temp_dir, ignore_errors=True)








# if __name__ == "__main__":
#     # Run the complex test script
#     test_complex_script()