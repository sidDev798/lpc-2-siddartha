import React, { useState, useEffect } from 'react';
import Prism from 'prismjs';
import 'prismjs/themes/prism.css';
import 'prismjs/components/prism-python';

const CodeDisplay = ({ fileName, code, language = 'python' }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const maxPreviewLines = 5;
  
  useEffect(() => {
    // Highlight code when component mounts or code changes
    if (code) {
      Prism.highlightAll();
    }
  }, [code, isExpanded]);
  
  if (!code) return null;
  
  const codeLines = code.split('\n');
  const previewCode = codeLines.slice(0, maxPreviewLines).join('\n');
  const isLongCode = codeLines.length > maxPreviewLines;
  
  const handleDownload = () => {
    const element = document.createElement('a');
    const file = new Blob([code], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = fileName || 'code.py';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };
  
  return (
    <div className="border rounded-md overflow-hidden bg-gray-50">
      <div className="flex items-center justify-between bg-gray-200 px-4 py-2">
        <div className="font-mono text-sm truncate flex-1" title={fileName}>
          {fileName || 'code.py'}
        </div>
        
        <div className="flex space-x-2">
          {isLongCode && (
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="text-xs bg-gray-100 hover:bg-gray-300 px-2 py-1 rounded"
              title={isExpanded ? 'Collapse code' : 'Expand code'}
            >
              {isExpanded ? 'Collapse' : 'Expand'}
            </button>
          )}
          
          <button
            onClick={handleDownload}
            className="text-xs bg-blue-100 hover:bg-blue-200 px-2 py-1 rounded flex items-center"
            title="Download code"
          >
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-3 h-3 mr-1">
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
            </svg>
            Download
          </button>
        </div>
      </div>
      
      <div className="p-4 overflow-x-auto">
        <pre className="text-sm">
          <code className={`language-${language}`}>
            {isExpanded || !isLongCode ? code : previewCode + (isLongCode ? '\n...' : '')}
          </code>
        </pre>
      </div>
    </div>
  );
};

export default CodeDisplay; 