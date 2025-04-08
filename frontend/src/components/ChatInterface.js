import React, { useState, useRef, useEffect } from 'react';
import MessageList from './MessageList';
import MessageInput from './MessageInput';

const ChatInterface = ({ messages, onSendMessage, loading }) => {
  const messagesEndRef = useRef(null);
  
  // Auto-scroll to the bottom when messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-auto mb-4 bg-white rounded-lg shadow">
        <MessageList messages={messages} />
        <div ref={messagesEndRef} />
      </div>
      
      <div className="mt-auto">
        <MessageInput onSendMessage={onSendMessage} disabled={loading} />
      </div>
    </div>
  );
};

export default ChatInterface; 