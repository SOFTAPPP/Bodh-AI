import React, { useState, useRef, useEffect } from 'react';
import { Upload, Send, FileText, Bot, User, Loader2, CheckCircle2, Mic, MicOff } from 'lucide-react';
import './App.css';

interface Message {
  id: string;
  role: 'user' | 'bot';
  content: string;
  sources?: { page: number; source: string }[];
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [currentFile, setCurrentFile] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (SpeechRecognition) {
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = false;
      recognitionRef.current.lang = 'en-US';

      recognitionRef.current.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        setInput(transcript);
        setIsListening(false);
      };

      recognitionRef.current.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
      };

      recognitionRef.current.onend = () => {
        setIsListening(false);
      };
    }
  }, []);

  const toggleListening = () => {
    if (isListening) {
      recognitionRef.current?.stop();
    } else {
      setInput('');
      recognitionRef.current?.start();
      setIsListening(true);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isThinking]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setCurrentFile(data.filename);
        setMessages(prev => [...prev, {
          id: Date.now().toString(),
          role: 'bot',
          content: `Successfully uploaded **${data.filename}**. You can now ask questions about it!`
        }]);
      }
    } catch (error) {
      console.error('Upload failed:', error);
    } finally {
      setIsUploading(false);
    }
  };

  const handleSendMessage = async () => {
    if (!input.trim() || isThinking) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
    };

    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsThinking(true);

    // Create a unique ID for the bot message we're about to stream
    const botMsgId = (Date.now() + 1).toString();
    const botPlaceholder: Message = {
      id: botMsgId,
      role: 'bot',
      content: '',
      sources: []
    };

    setMessages(prev => [...prev, botPlaceholder]);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: input }),
      });

      if (!response.ok) throw new Error('Network response was not ok');

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let streamedContent = '';

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          if (chunk.trim()) setIsThinking(false); // Only stop thinking when real text arrives

          streamedContent += chunk;

          // Update the message in the state as it streams
          setMessages(prev => prev.map(msg =>
            msg.id === botMsgId ? { ...msg, content: streamedContent } : msg
          ));
        }
      }
    } catch (error) {
      console.error('Chat failed:', error);
      setMessages(prev => prev.map(msg =>
        msg.id === botMsgId ? { ...msg, content: '❌ Sorry, I encountered an error. Please try again.' } : msg
      ));
    } finally {
      setIsThinking(false);
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="logo">
          <div className="logo-icon">
            <Bot size={24} color="white" />
          </div>
          <h2>BodhAI</h2>
        </div>

        <div className="upload-section">
          <label className={`upload-card ${isUploading ? 'loading' : ''}`}>
            <input type="file" accept=".pdf" onChange={handleFileUpload} hidden />
            <Upload size={32} className="upload-icon" />
            <p>{isUploading ? 'Processing...' : 'Upload Legal PDF'}</p>
            <span>Supports .pdf files</span>
          </label>
        </div>

        {currentFile && (
          <div className="file-info fade-in">
            <div className="file-pill">
              <FileText size={16} />
              <span>{currentFile}</span>
              <CheckCircle2 size={14} color="#10b981" />
            </div>
          </div>
        )}

        <div className="sidebar-footer">
          <p>Demo Version 1.0</p>
        </div>
      </aside>

      {/* Main Chat */}
      <main className="chat-area">
        <header className="chat-header">
          <h3>Chat Interface</h3>
          <div className="status-indicator">
            <div className={`dot ${currentFile ? 'active' : ''}`}></div>
            {currentFile ? 'AI Ready' : 'Upload a file to start'}
          </div>
        </header>

        <div className="messages-container">
          {messages.length === 0 && (
            <div className="empty-state">
              <div className="empty-icon">
                <Bot size={54} strokeWidth={1.5} />
              </div>
              <h2>PDF Intelligence</h2>
              <p>Upload your documents and let's unlock their secrets together. Your personal AI assistant is ready.</p>
            </div>
          )}

          {messages.map((msg, index) => (
            <div key={msg.id} className={`message-wrapper ${msg.role}`}>
              <div className="message-icon">
                {msg.role === 'bot' ? <Bot size={20} /> : <User size={20} />}
              </div>
              <div className="message-content">
                <div className={`message-bubble ${!msg.content && msg.role === 'bot' ? 'pulse' : ''} ${!msg.content && msg.role === 'bot' && isThinking ? 'thinking' : ''}`}>
                  {msg.content ? (
                    msg.content.split('\n').map((line, i) => {
                      const trimmedLine = line.trim();
                      const isBullet = /^[-*•]\s+/.test(trimmedLine);
                      const content = trimmedLine.replace(/^[-*•]+\s+/, '');

                      // Detect category headers: starts with ** and ends with ** (optionally with a colon)
                      const isCategoryHeader = /^(\*\*.*?\*\*:?|.*?:\s*)$/.test(trimmedLine) && !isBullet && trimmedLine.length < 50;

                      const parts = content.split(/(\*\*.*?\*\*)/g);
                      const renderedLine = parts.map((part, index) => {
                        if (part.startsWith('**') && part.endsWith('**')) {
                          return <strong key={index}>{part.slice(2, -2)}</strong>;
                        }
                        return part;
                      });

                      if (!trimmedLine) return <div key={i} style={{ height: '8px' }}></div>;

                      return (
                        <div key={i} className={`${isBullet ? 'bullet-line' : 'text-line'} ${isCategoryHeader ? 'category-header' : ''}`}>
                          {isBullet && <span className="bullet-dot">•</span>}
                          <span>{renderedLine}</span>
                        </div>
                      );
                    })
                  ) : (
                    <div className="typing-indicator-wrapper">
                       {isThinking && index === messages.length - 1 ? (
                        <div className="thinking-content">
                          <Loader2 size={16} className="spin" />
                          <span>AI is analyzing...</span>
                        </div>
                       ) : (
                        <div className="typing-indicator">
                          <span></span><span></span><span></span>
                        </div>
                       )}
                    </div>
                  )}
                </div>

                {msg.sources && msg.sources.length > 0 && (
                  <div className="sources-container">
                    <p className="source-label">Sources:</p>
                    <div className="source-chips">
                      {msg.sources.map((s, i) => (
                        <span key={i} className="source-chip">
                          Page {s.page}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        <div className="input-container">
          <div className="input-wrapper">
            <input
              type="text"
              placeholder="Ask a question about the PDF..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
            />
            <button 
              className={`mic-button ${isListening ? 'listening' : ''}`}
              onClick={toggleListening}
              type="button"
              title="Voice Search"
            >
              {isListening ? <MicOff size={20} /> : <Mic size={20} />}
            </button>
            <button
              className={`send-button ${!input.trim() || isThinking ? 'disabled' : ''}`}
              onClick={handleSendMessage}
            >
              <Send size={20} />
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
