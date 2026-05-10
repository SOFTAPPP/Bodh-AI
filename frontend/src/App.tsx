import React, { useState, useRef, useEffect, type JSX } from 'react';
import { Upload, Send, FileText, Bot, User, Loader2, CheckCircle2, Mic, MicOff, Sun, Moon, Home } from 'lucide-react';
import './App.css';

interface Message {
  id: string;
  role: 'user' | 'bot';
  content: string;
  sources?: { page: number; source: string }[];
}

function App() {
  const [view, setView] = useState<'landing' | 'features' | 'chat'>('landing');
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [currentFile, setCurrentFile] = useState<string | null>(null);
  const [indexedFiles, setIndexedFiles] = useState<string[]>([]);
  const [darkMode, setDarkMode] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const recognitionRef = useRef<any>(null);

  // Theme Toggle Effect
  useEffect(() => {
    if (darkMode) {
      document.body.classList.add('dark');
    } else {
      document.body.classList.remove('dark');
    }
  }, [darkMode]);

  const fetchFiles = async () => {
    try {
      const response = await fetch('http://localhost:8000/files');
      if (response.ok) {
        const data = await response.json();
        setIndexedFiles(data.files);
        // If no current file is selected but files exist, select the first one
        if (!currentFile && data.files.length > 0) {
          // setCurrentFile(data.files[0]);
        }
      }
    } catch (error) {
      console.error('Failed to fetch files:', error);
    }
  };

  useEffect(() => {
    fetchFiles();
  }, []);

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

  const scrollToBottom = (isSmooth = false) => {
    messagesEndRef.current?.scrollIntoView({ behavior: isSmooth ? 'smooth' : 'auto' });
  };

  useEffect(() => {
    if (view === 'chat') {
      scrollToBottom(messages.length > 0 && !isThinking);
    }
  }, [messages, isThinking, view]);

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

        const isAlreadyIndexed = data.status === 'already_indexed';
        const msg = isAlreadyIndexed
          ? `**${data.filename}** is ready.`
          : `Successfully uploaded **${data.filename}**. Continue with your query`;

        setMessages(prev => [...prev, {
          id: Date.now().toString(),
          role: 'bot',
          content: msg
        }]);

        await fetchFiles(); // Refresh list
        setView('chat');
      }
    } catch (error) {
      console.error('Upload failed:', error);
    } finally {
      setIsUploading(false);
      // Reset input so the same file can be selected again
      e.target.value = '';
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
        body: JSON.stringify({
          message: input,
          history: messages.map(m => ({ role: m.role, content: m.content })),
          active_file: currentFile
        }),
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
          if (chunk) {
            streamedContent += chunk;
            setIsThinking(false);
            setMessages(prev => {
              const last = prev[prev.length - 1];
              if (last && last.id === botMsgId) {
                const updated = [...prev];
                updated[updated.length - 1] = { ...last, content: streamedContent };
                return updated;
              }
              return prev;
            });
          }
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

  const renderMessageContent = (content: string, isThinking: boolean, role: string) => {
    if (!content && role === 'bot') {
      return (
        <div className="typing-indicator-wrapper">
          {isThinking ? (
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
      );
    }

    const lines = content.split('\n');
    const renderedElements: JSX.Element[] = [];
    let currentTable: string[][] = [];
    let isInTable = false;

    const pushTable = (index: number) => {
      if (currentTable.length > 0) {
        const tableKey = `table-${index}`;
        // Detect if it's a header or just rows
        const hasHeader = currentTable.length > 0;

        renderedElements.push(
          <div key={tableKey} className="table-container streaming-table">
            <table>
              {hasHeader && (
                <thead>
                  <tr>
                    {currentTable[0].map((cell, idx) => (
                      <th key={idx}>{renderInline(cell)}</th>
                    ))}
                  </tr>
                </thead>
              )}
              <tbody>
                {currentTable.slice(1).map((row, rowIdx) => (
                  <tr key={rowIdx}>
                    {row.map((cell, cellIdx) => (
                      <td key={cellIdx}>{renderInline(cell)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      }
      currentTable = [];
      isInTable = false;
    };

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const trimmedLine = line.trim();

      const isTableRow = trimmedLine.startsWith('|') && trimmedLine.endsWith('|');
      const isTableDivider = isTableRow && trimmedLine.includes('---');

      if (isTableRow) {
        if (!isTableDivider) {
          isInTable = true;
          const cells = trimmedLine.split('|').filter((_, idx, arr) => idx > 0 && idx < arr.length - 1).map(c => c.trim());
          currentTable.push(cells);
        }

        // If it's the last line and we're in a table, push it now
        if (i === lines.length - 1 && isInTable) {
          pushTable(i);
        }
        continue;
      } else if (isInTable) {
        pushTable(i);
      }

      if (!trimmedLine) {
        renderedElements.push(<div key={i} style={{ height: '8px' }}></div>);
        continue;
      }

      const isBullet = /^[-*•]\s+/.test(trimmedLine);
      const text = trimmedLine.replace(/^[-*•]+\s+/, '');
      const isCategoryHeader = (/^(\*\*.*?\*\*:?|.*?:\s*)$/.test(trimmedLine) && !isBullet && trimmedLine.length < 60);

      renderedElements.push(
        <div key={i} className={`${isBullet ? 'bullet-line' : 'text-line'} ${isCategoryHeader ? 'category-header' : ''}`}>
          {isBullet && <span className="bullet-dot">•</span>}
          <span>{renderInline(text)}</span>
        </div>
      );
    }

    return renderedElements;
  };

  const renderInline = (text: string) => {
    const parts = text.split(/(\*\*.*?\*\*|\[.*?\]\(.*?\))/g);
    return parts.map((part, idx) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={idx}>{part.slice(2, -2)}</strong>;
      }
      if (part.startsWith('[') && part.includes('](')) {
        const match = part.match(/\[(.*?)\]\((.*?)\)/);
        if (match) {
          return (
            <a key={idx} href={match[2]} target="_blank" rel="noopener noreferrer" className="message-link">
              {match[1]}
            </a>
          );
        }
      }
      return part;
    });
  };

  const LandingView = () => (
    <div className="landing-container view-transition">
      <nav className="landing-nav">
        <div className="logo-group">
          <Bot className="logo-icon-nav" />
          <span className="logo-text">BodhAI</span>
        </div>
        <div className="nav-links">
          <button onClick={() => setView('landing')} className={view === 'landing' ? 'active' : ''}>Home</button>
          <button onClick={() => setView('features')} className={view === 'features' ? 'active' : ''}>Features</button>
          <button onClick={() => setView('chat')} className={view === 'chat' ? 'active' : ''}>Dashboard</button>
        </div>
        <button className="theme-toggle-nav" onClick={() => setDarkMode(!darkMode)}>
          {darkMode ? <Sun size={20} /> : <Moon size={20} />}
        </button>
      </nav>

      <section className="hero-section hero-gradient">
        <div className="hero-content">
          <div className="badge animate-fadeIn">State-of-the-Art Document Intelligence</div>
          <h1 className="hero-title animate-slideUp">
            Precision Insights <br />
            From Your <span>Documents</span>
          </h1>
          <p className="hero-subtitle animate-slideUp">
            Enterprise-grade RAG engine designed for absolute accuracy in Legal,
            Medical, and Financial document intelligence.
          </p>
          <div className="hero-actions animate-slideUp">
            <button className="btn-primary" onClick={() => setView('chat')}>
              Try BodhAI Now <Send size={18} />
            </button>
            <button className="btn-secondary" onClick={() => setView('features')}>
              Explore Features
            </button>
          </div>
        </div>
        <div className="hero-visual animate-float">
          <div className="visual-card glass-card">
            <FileText size={48} className="visual-icon" />
            <div className="visual-lines">
              <div className="line long"></div>
              <div className="line short"></div>
              <div className="line mid"></div>
            </div>
            <div className="visual-badge">Precision RAG</div>
          </div>
          <div className="visual-circle animate-pulse-slow"></div>
        </div>
      </section>
    </div>
  );

  const FeaturesView = () => (
    <div className="features-container view-transition">
      <nav className="landing-nav">
        <div className="logo-group">
          <Bot className="logo-icon-nav" />
          <span className="logo-text">BodhAI</span>
        </div>
        <div className="nav-links">
          <button onClick={() => setView('landing')} className={view === 'landing' ? 'active' : ''}>Home</button>
          <button onClick={() => setView('features')} className={view === 'features' ? 'active' : ''}>Features</button>
          <button onClick={() => setView('chat')} className={view === 'chat' ? 'active' : ''}>Dashboard</button>
        </div>
        <button className="theme-toggle-nav" onClick={() => setDarkMode(!darkMode)}>
          {darkMode ? <Sun size={20} /> : <Moon size={20} />}
        </button>
      </nav>

      <header className="features-header">
        <h2>Intelligent Features</h2>
        <p>Built for precision, speed, and absolute accuracy.</p>
      </header>

      <div className="features-grid">
        {[
          { icon: <CheckCircle2 size={32} />, title: 'Zero Hallucination', desc: 'Strict grounding logic ensures every answer is backed by document facts.' },
          { icon: <Bot size={32} />, title: 'Verbatim Evidence', desc: 'Get exact quotes and chunk references for every response generated.' },
          { icon: <Mic size={32} />, title: 'Voice Intelligence', desc: 'Natural voice-to-text integration for hands-free document analysis.' },
          { icon: <Loader2 size={32} />, title: 'Rapid Ingestion', desc: 'Proprietary pipeline processes massive PDFs in seconds with local embeddings.' },
          { icon: <Sun size={32} />, title: 'Adaptive UI', desc: 'Seamlessly switch between Dark and Light modes for any environment.' },
          { icon: <FileText size={32} />, title: 'Multi-Doc RAG', desc: 'Query across multiple documents with intelligent cross-referencing.' }
        ].map((feature, i) => (
          <div key={i} className="feature-card glass-card">
            <div className="feature-icon-wrapper">{feature.icon}</div>
            <h3>{feature.title}</h3>
            <p>{feature.desc}</p>
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <div className="app-root">
      {view === 'landing' && <LandingView />}
      {view === 'features' && <FeaturesView />}
      {view === 'chat' && (
        <div className="app-container view-transition">
          <aside className="sidebar">
            <div className="logo" onClick={() => setView('landing')} style={{ cursor: 'pointer' }}>
              <div className="logo-icon">
                <Bot size={24} color="white" />
              </div>
              <h2>BodhAI</h2>
            </div>

            <div className="upload-section">
              <label className={`upload-card ${isUploading ? 'loading' : ''}`}>
                <input type="file" accept=".pdf" onChange={handleFileUpload} hidden />
                {isUploading ? (
                  <Loader2 size={32} className="upload-icon spin" />
                ) : (
                  <Upload size={32} className="upload-icon" />
                )}
                <p>{isUploading ? 'Processing...' : 'Upload Files'}</p>
                <span>Supports .pdf files</span>
              </label>
            </div>

            <div className="file-list-section">
              <h4 className="section-title">Documents</h4>
              <div className="file-list-container">
                {indexedFiles.length === 0 && !isUploading && (
                  <p className="empty-files-hint">No files indexed yet.</p>
                )}
                {indexedFiles.map((file) => (
                  <div
                    key={file}
                    className={`file-item ${currentFile === file ? 'active' : ''}`}
                    onClick={() => setCurrentFile(file)}
                  >
                    <FileText size={16} className="file-item-icon" />
                    <span className="file-item-name" title={file}>{file}</span>
                    {currentFile === file && <CheckCircle2 size={14} color="#3b82f6" />}
                  </div>
                ))}
              </div>
            </div>

            <div className="sidebar-nav">
              <button onClick={() => setView('landing')}><Home size={18} /> Home</button>
              <button onClick={() => setView('features')}><Bot size={18} /> Features</button>
            </div>

            <div className="sidebar-footer">
              <p>Demo Version 1.0</p>
            </div>
          </aside>

          <main className="chat-area">
            <header className="chat-header">
              <h3>Chat Interface</h3>
              <div className="header-actions">
                <button className="theme-toggle" onClick={() => setDarkMode(!darkMode)}>
                  {darkMode ? <Sun size={20} /> : <Moon size={20} />}
                </button>
                <div className="status-indicator">
                  <div className={`dot ${currentFile ? 'active' : ''}`}></div>
                  {currentFile ? 'AI Ready' : 'Upload a file to start'}
                </div>
              </div>
            </header>

            <div className="messages-container">
              {messages.length === 0 && (
                <div className="empty-state">
                  <div className="empty-icon"><Bot size={54} strokeWidth={1.5} /></div>
                  <h2>PDF Intelligence</h2>
                  <p>Upload your documents and let's unlock their secrets together.</p>
                </div>
              )}

              {messages.map((msg) => (
                <div key={msg.id} className={`message-wrapper ${msg.role}`}>
                  <div className="message-icon">
                    {msg.role === 'bot' ? <Bot size={20} /> : <User size={20} />}
                  </div>
                  <div className="message-content">
                    <div className={`message-bubble ${!msg.content && msg.role === 'bot' ? 'pulse' : ''}`}>
                      {renderMessageContent(msg.content, isThinking, msg.role)}
                    </div>
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
      )}
    </div>
  );
}

export default App;