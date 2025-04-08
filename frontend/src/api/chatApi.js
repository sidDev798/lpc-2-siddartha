import axios from 'axios';

// API base URL - this should be configured based on your environment
const API_BASE_URL = 'http://localhost:8001';

// Create an axios instance with default configuration
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Send a message to the API and get a response
 * @param {string} message - The user's text message
 * @param {string|null} fileContent - Content of uploaded Python file (if any)
 * @returns {Promise<Object>} - The API response
 */
export const sendMessage = async (message, fileContent = null) => {
  try {
    const response = await apiClient.post('/chat', {
      message,
      file_content: fileContent,
    });
    return response.data;
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
};

export default { sendMessage }; 