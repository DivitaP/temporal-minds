import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isConnected, setIsConnected] = useState(true);
  const messagesEndRef = useRef(null);

  // Scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Initial greeting message when component mounts
  useEffect(() => {
    const initialMessage = { 
      sender: 'bot', 
      text: "Hello! I'm Alan Turing. I can tell you about my early life, education, and academic work from 1911 to 1938. What would you like to know?" 
    };
    setMessages([initialMessage]);
    
    // Check connectivity to backend
    axios.post('http://127.0.0.1:5000/chat', { message: "connection_test" })
      .then(() => setIsConnected(true))
      .catch(() => setIsConnected(false));
  }, []);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { sender: 'user', text: input };
    setMessages([...messages, userMessage]);
    setInput('');
    setIsTyping(true);

    try {
      const response = await axios.post('http://127.0.0.1:5000/chat', { message: input });
      const botMessage = { sender: 'bot', text: response.data.response };
      setMessages(prev => [...prev, botMessage]);
      setIsConnected(true);
    } catch (error) {
      setMessages(prev => [...prev, { 
        sender: 'bot', 
        text: 'I apologize, but I seem to be having trouble connecting to my knowledge base. Please ensure the backend service is running.' 
      }]);
      setIsConnected(false);
      console.error('Error:', error);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') sendMessage();
  };

  return (
    <div className="chat-container">
      <h1 className="chat-title">Alan Turing Knowledge Graph</h1>
      <div className="chat-subtitle">
        Historical knowledge covering 1911-1938
      </div>
      
      {!isConnected && (
        <div className="connection-warning">
          ⚠️ Not connected to backend service. Please ensure the Flask server is running.
        </div>
      )}
      
      <div className="chat-box">
        {messages.map((msg, index) => (
          <div key={index} className={`chat-message ${msg.sender}`}>
            <div className="message-avatar">
              {msg.sender === 'bot' ? '፨' : '👤'}
            </div>
            <div className="message-content">
              {msg.text}
            </div>
          </div>
        ))}
        {isTyping && (
          <div className="chat-message bot">
            <div className="message-avatar"></div>
            <div className="message-content typing">
              <span className="dot"></span>
              <span className="dot"></span>
              <span className="dot"></span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <div className="input-box">
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyPress}
          placeholder="Ask Alan Turing something..."
          disabled={isTyping}
        />
        <button onClick={sendMessage} disabled={isTyping || !input.trim()}>
          {isTyping ? 'Thinking...' : 'Send'}
        </button>
      </div>
    </div>
  );
}

export default App;