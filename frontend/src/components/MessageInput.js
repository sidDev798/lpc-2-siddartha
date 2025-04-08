import React, { useState, useRef } from 'react';

const MessageInput = ({ onSendMessage, disabled }) => {
  const [message, setMessage] = useState('');
  const [file, setFile] = useState(null);
  const fileInputRef = useRef(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (message.trim() || file) {
      onSendMessage(message, file);
      setMessage('');
      setFile(null);
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.name.endsWith('.py')) {
      setFile(selectedFile);
    } else {
      alert('Please select a Python (.py) file');
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  const removeFile = () => {
    setFile(null);
    fileInputRef.current.value = '';
  };

  return (
    <form onSubmit={handleSubmit} className="bg-white rounded-lg shadow p-3">
      {file && (
        <div className="flex items-center bg-blue-50 p-2 rounded mb-2">
          <div className="flex-1 text-sm truncate">
            <span className="font-medium">File:</span> {file.name}
          </div>
          <button 
            type="button" 
            onClick={removeFile}
            className="ml-2 text-red-500 hover:text-red-700"
          >
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      )}
      
      <div className="flex items-end">
        <input
          type="file"
          accept=".py"
          ref={fileInputRef}
          onChange={handleFileChange}
          className="hidden"
        />
        
        <button
          type="button"
          onClick={triggerFileInput}
          disabled={disabled}
          className="p-2 text-gray-500 hover:text-blue-600 disabled:opacity-50"
          title="Upload Python file"
        >
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
            <path strokeLinecap="round" strokeLinejoin="round" d="M18.375 12.739l-7.693 7.693a4.5 4.5 0 01-6.364-6.364l10.94-10.94A3 3 0 1119.5 7.372L8.552 18.32m.009-.01l-.01.01m5.699-9.941l-7.81 7.81a1.5 1.5 0 002.112 2.13" />
          </svg>
        </button>
        
        <div className="flex-1 mx-2">
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Ask a Python question or upload a file..."
            disabled={disabled}
            className="w-full border-0 focus:ring-0 resize-none max-h-32 rounded-md py-2 px-3 bg-gray-50"
            rows={1}
            style={{ minHeight: '44px' }}
          />
        </div>
        
        <button
          type="submit"
          disabled={disabled || (!message.trim() && !file)}
          className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md disabled:opacity-50"
        >
          {disabled ? (
            <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
          ) : 'Send'}
        </button>
      </div>
    </form>
  );
};

export default MessageInput; 