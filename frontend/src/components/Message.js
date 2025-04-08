import React from 'react';
import CodeDisplay from './CodeDisplay';

const Message = ({ message }) => {
  const { content, sender, code, file, isError } = message;
  
  const isUser = sender === 'user';
  const bgColor = isUser ? 'bg-blue-100' : (isError ? 'bg-red-100' : 'bg-gray-100');
  const alignment = isUser ? 'justify-end' : 'justify-start';
  const textAlignment = isUser ? 'text-right' : 'text-left';
  
  return (
    <div className={`flex ${alignment} mb-4`}>
      <div className={`max-w-[80%] ${bgColor} rounded-lg p-4 shadow-sm`}>
        <div className={`text-sm text-gray-500 mb-1 ${textAlignment}`}>
          {isUser ? 'You' : 'Assistant'}
        </div>
        
        <div className="mb-2 whitespace-pre-wrap">
          {content}
        </div>
        
        {file && (
          <div className="mt-2">
            <CodeDisplay 
              fileName={file.name} 
              code={file.content} 
              language="python" 
            />
          </div>
        )}
        
        {code && (
          <div className="mt-2">
            <CodeDisplay 
              fileName="result.py" 
              code={code} 
              language="python" 
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default Message; 