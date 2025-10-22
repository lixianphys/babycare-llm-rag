'use client'

import { useState, useRef, useEffect } from 'react'
import { Sparkles, MessageSquare, RotateCcw } from 'lucide-react'
import axios from 'axios'
import { useAuth } from '../contexts/AuthContext'
import MessageBubble from '../components/MessageBubble'
import WelcomeScreen from '../components/WelcomeScreen'
import ChatInput from '../components/ChatInput'
import DocumentViewer from '../components/DocumentViewer'
import LoginButton from '../components/LoginButton'
import UserMenu from '../components/UserMenu'
import ChatHistory from '../components/ChatHistory'

interface Message {
  id: string
  content: string
  isUser: boolean
  timestamp: Date
  retrievedDocuments?: DocumentInfo[]
  retrievedCount?: number
}

interface DocumentInfo {
  content: string
  metadata: {
    source?: string
    category?: string
    [key: string]: any
  }
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function Home() {
  const { isAuthenticated, isGuest } = useAuth()
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [examples, setExamples] = useState<string[]>([])
  const [selectedDocument, setSelectedDocument] = useState<DocumentInfo | null>(null)
  const [isDocumentViewerOpen, setIsDocumentViewerOpen] = useState(false)
  const [isChatHistoryOpen, setIsChatHistoryOpen] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Load example questions on mount
  useEffect(() => {
    const loadExamples = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/examples`)
        setExamples(response.data.examples)
      } catch (error) {
        console.error('Failed to load examples:', error)
      }
    }
    loadExamples()
  }, [])

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])


  const sendMessage = async (message: string) => {
    if (!message.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      content: message,
      isUser: true,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    // Add typing indicator
    const typingMessage: Message = {
      id: (Date.now() + 1).toString(),
      content: 'Thinking...',
      isUser: false,
      timestamp: new Date()
    }
    setMessages(prev => [...prev, typingMessage])

    try {
      const response = await axios.post(`${API_BASE_URL}/chat`, {
        message: message,
        conversation_id: 'web_session',
        user_id: 'web_user'
      })

      // Remove typing indicator and add actual response
      setMessages(prev => {
        const withoutTyping = prev.filter(msg => msg.id !== typingMessage.id)
        const aiMessage: Message = {
          id: (Date.now() + 2).toString(),
          content: response.data.response,
          isUser: false,
          timestamp: new Date(),
          retrievedDocuments: response.data.retrieved_documents,
          retrievedCount: response.data.retrieved_count
        }
        return [...withoutTyping, aiMessage]
      })
    } catch (error) {
      console.error('Failed to send message:', error)
      setMessages(prev => {
        const withoutTyping = prev.filter(msg => msg.id !== typingMessage.id)
        const errorMessage: Message = {
          id: (Date.now() + 2).toString(),
          content: 'Sorry, I encountered an error. Please try again.',
          isUser: false,
          timestamp: new Date()
        }
        return [...withoutTyping, errorMessage]
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    sendMessage(input)
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage(input)
    }
  }

  const handleExampleClick = (example: string) => {
    sendMessage(example)
  }

  const clearChat = () => {
    setMessages([])
  }

  const handleDocumentClick = (document: DocumentInfo) => {
    setSelectedDocument(document)
    setIsDocumentViewerOpen(true)
  }

  const closeDocumentViewer = () => {
    setIsDocumentViewerOpen(false)
    setSelectedDocument(null)
  }


  return (
    <div className="h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
            <Sparkles className="h-5 w-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-gray-900">Baby Care Assistant</h1>
            <p className="text-sm text-gray-500">AI-powered baby care knowledge</p>
          </div>
        </div>
        <div className="flex items-center space-x-3">
          <button
            onClick={clearChat}
            className="flex items-center space-x-2 px-3 py-2 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <RotateCcw className="h-4 w-4" />
            <span>New chat</span>
          </button>
          {isAuthenticated ? (
            <UserMenu onShowHistory={() => setIsChatHistoryOpen(true)} />
          ) : (
            <div className="flex items-center space-x-2">
              <LoginButton />
              {isGuest && (
                <span className="text-sm text-gray-500">
                  Guest mode - <span className="text-blue-600">Sign in to save chat history</span>
                </span>
              )}
            </div>
          )}
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar - Hidden on mobile, visible on desktop */}
        <div className="hidden lg:block w-80 bg-white border-r border-gray-200 p-6 overflow-y-auto">
          <div className="mb-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Quick Start</h2>
            <div className="space-y-2">
              {examples.map((example, index) => (
                <button
                  key={index}
                  onClick={() => handleExampleClick(example)}
                  className="w-full text-left p-3 rounded-lg border border-gray-200 hover:border-blue-300 hover:bg-blue-50 transition-all duration-200 text-sm group"
                  disabled={isLoading}
                >
                  <div className="flex items-start space-x-2">
                    <MessageSquare className="h-4 w-4 text-gray-400 group-hover:text-blue-500 mt-0.5 flex-shrink-0" />
                    <span className="text-gray-700 group-hover:text-blue-700">{example}</span>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Main Chat Area */}
        <div className="flex-1 flex flex-col">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto">
            {messages.length === 0 && (
              <WelcomeScreen 
                examples={examples}
                onExampleClick={handleExampleClick}
                isLoading={isLoading}
              />
            )}
            
            <div className="max-w-4xl mx-auto px-4 py-6">
              {messages.map((message) => (
                <MessageBubble 
                  key={message.id} 
                  message={message} 
                  onDocumentClick={handleDocumentClick}
                />
              ))}
              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Input Area */}
          <ChatInput
            input={input}
            setInput={setInput}
            onSubmit={handleSubmit}
            onKeyPress={handleKeyPress}
            isLoading={isLoading}
          />
        </div>
      </div>

          {/* Document Viewer Modal */}
          <DocumentViewer
            isOpen={isDocumentViewerOpen}
            onClose={closeDocumentViewer}
            document={selectedDocument}
          />

          {/* Chat History Modal */}
          <ChatHistory
            isOpen={isChatHistoryOpen}
            onClose={() => setIsChatHistoryOpen(false)}
          />
        </div>
      )
    }
