import React, { useState, useEffect } from 'react';
import ChatInterface from './components/ChatInterface';
import CodeDisplay from './components/CodeDisplay';
import { sendMessage } from './api/chatApi';

function App() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  
  const handleSendMessage = async (message, file = null) => {
    // Add user message to the chat
    const userMessage = {
      id: Date.now(),
      content: message,
      sender: 'user',
      file: file ? { name: file.name, content: null } : null
    };
    
    setMessages(prevMessages => [...prevMessages, userMessage]);
    setLoading(true);
    
    try {
      // Handle file if present
      let fileContent = null;
      if (file) {
        const reader = new FileReader();
        fileContent = await new Promise((resolve) => {
          reader.onload = (e) => resolve(e.target.result);
          reader.readAsText(file);
        });
        
        // Update user message with file content
        userMessage.file.content = fileContent;
        setMessages(prevMessages => 
          prevMessages.map(msg => 
            msg.id === userMessage.id ? {...msg, file: {...msg.file, content: fileContent}} : msg
          )
        );
      }
      
      // Send to API
      const response = await sendMessage(message, fileContent);
      
      // Add assistant response
      const assistantMessage = {
        id: Date.now() + 1,
        content: response.text,
        sender: 'assistant',
        code: response.code || null
      };
      
      setMessages(prevMessages => [...prevMessages, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      // Add error message
      setMessages(prevMessages => [
        ...prevMessages, 
        {
          id: Date.now() + 1,
          content: "Sorry, I couldn't process your request. Please try again.",
          sender: 'assistant',
          isError: true
        }
      ]);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <header className="bg-blue-600 text-white p-4 shadow-md">
        <h1 className="text-xl font-bold">Python Code Assistant</h1>
      </header>
      
      <main className="flex-1 flex flex-col p-4 max-w-4xl mx-auto w-full">
        <ChatInterface 
          messages={messages} 
          onSendMessage={handleSendMessage}
          loading={loading}
        />
      </main>
      
      <footer className="bg-gray-100 border-t p-4 text-center text-gray-500 text-sm">
        Python Code Assistant &copy; {new Date().getFullYear()}
      </footer>
    </div>
  );
}

export default App; 