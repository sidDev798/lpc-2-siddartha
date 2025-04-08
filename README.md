# Python Code Assistant

A full-stack application that uses AI to assist with Python coding tasks. The application consists of a React frontend and a FastAPI backend.

## Project Structure

```
project-root/
│
├── frontend/              # React frontend application
│   ├── public/            # Static assets
│   ├── src/               # Source code
│   │   ├── api/           # API integration
│   │   ├── components/    # React components
│   │   ├── App.js         # Main application component
│   │   └── index.js       # Entry point
│   ├── package.json       # Dependencies and scripts
│   └── ...
│
├── backend/               # FastAPI backend application
│   ├── custom_tools/      # Custom tools for the AI assistant
│   ├── main.py            # FastAPI application entry point
│   ├── python_agent.py    # Python assistant agent implementation
│   ├── requirements.txt   # Python dependencies
│   └── ...
│
└── start.sh               # Script to start both frontend and backend
```

## Prerequisites

- Node.js and npm for the frontend
- Python 3.8+ for the backend
- API key for the OpenAI API (set in backend/.env)

## Installation

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   Then edit the `.env` file to add your OpenAI API key.

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

## Running the Application

### Option 1: Using the start script

The easiest way to run both frontend and backend is using the start script:

```bash
./start.sh
```

This script starts both the backend server and the frontend development server concurrently.

### Option 2: Manual startup

#### Start the Backend

```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### Start the Frontend

```bash
cd frontend
npm start
```

## Accessing the Application

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

## API Endpoints

- `GET /`: Health check endpoint
- `POST /chat`: Main endpoint for sending messages and receiving responses

## License

MIT 