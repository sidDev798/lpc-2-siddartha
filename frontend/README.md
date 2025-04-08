# Python Code Assistant - Frontend

A React-based frontend for a Python code assistant chatbot that helps users create and debug Python code.

## Features

- Interactive chat interface
- Python file upload and processing
- Collapsible code blocks with syntax highlighting
- Code download capability
- Responsive design

## Getting Started

### Prerequisites

- Node.js (v14+)
- npm or yarn

### Installation

1. Clone the repository or download the source code
2. Navigate to the project directory
3. Install dependencies:

```bash
npm install
```

### Development

Run the development server:

```bash
npm start
```

This will start the development server at [http://localhost:3000](http://localhost:3000).

### Production Build

Create a production build:

```bash
npm run build
```

## Backend Integration

The frontend is designed to work with a FastAPI backend. By default, it expects the backend to be running at `http://localhost:8000`.

To connect to your backend:

1. Update the `API_BASE_URL` in `src/api/chatApi.js` if your backend is running on a different URL
2. Uncomment the actual API call in the `sendMessage` function and remove the mock response

## File Structure

- `/src/components` - React components
- `/src/api` - API service for backend communication
- `/src/utils` - Utility functions

## License

MIT 