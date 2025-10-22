'use client'
import { useState, useEffect } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { X, Trash2, MessageSquare } from 'lucide-react'
import axios from 'axios'
import DocumentViewer from './DocumentViewer'

interface Conversation {
  conversation_id: string
  last_updated: string
  created_at: string
  message_count: number
  last_message: {
    content: string
    type: string
  }
}

interface ChatHistoryProps {
  isOpen: boolean
  onClose: () => void
}

export default function ChatHistory({ isOpen, onClose }: ChatHistoryProps) {
  const { isAuthenticated, isGuest } = useAuth()
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [selectedConversation, setSelectedConversation] = useState<string | null>(null)
  const [messages, setMessages] = useState<any[]>([])
  const [selectedDocument, setSelectedDocument] = useState<any>(null)
  const [isDocumentViewerOpen, setIsDocumentViewerOpen] = useState(false)

  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

  useEffect(() => {
    if (isOpen && isAuthenticated) {
      loadConversations()
    }
  }, [isOpen, isAuthenticated])

  const loadConversations = async () => {
    setIsLoading(true)
    try {
      const response = await axios.get(`${API_BASE_URL}/chat/history`)
      setConversations(response.data.conversations || [])
    } catch (error) {
      console.error('Error loading conversations:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const loadMessages = async (conversationId: string) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/chat/history?conversation_id=${conversationId}`)
      setMessages(response.data.messages || [])
      setSelectedConversation(conversationId)
    } catch (error) {
      console.error('Error loading messages:', error)
    }
  }

  const deleteConversation = async (conversationId: string) => {
    if (!confirm('Are you sure you want to delete this conversation?')) {
      return
    }

    try {
      await axios.delete(`${API_BASE_URL}/chat/history/${conversationId}`)
      setConversations(conversations.filter(c => c.conversation_id !== conversationId))
      if (selectedConversation === conversationId) {
        setSelectedConversation(null)
        setMessages([])
      }
    } catch (error) {
      console.error('Error deleting conversation:', error)
    }
  }

  const deleteAllConversations = async () => {
    if (!confirm('Are you sure you want to delete ALL conversations? This cannot be undone.')) {
      return
    }

    try {
      await axios.delete(`${API_BASE_URL}/chat/history`)
      setConversations([])
      setSelectedConversation(null)
      setMessages([])
    } catch (error) {
      console.error('Error deleting all conversations:', error)
    }
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  if (!isOpen) return null

  // Show guest message if not authenticated
  if (isGuest) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
        <div className="bg-white rounded-lg shadow-xl w-full max-w-md">
          <div className="p-6 text-center">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Sign In Required</h2>
            <p className="text-gray-600 mb-6">
              You need to sign in to view and manage your chat history.
            </p>
            <div className="flex space-x-3">
              <button
                onClick={onClose}
                className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  onClose()
                  // Trigger login modal
                  window.dispatchEvent(new CustomEvent('openLogin'))
                }}
                className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                Sign In
              </button>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-4xl h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900">Chat History</h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="flex-1 flex overflow-hidden">
          {/* Conversations List */}
          <div className="w-1/3 border-r border-gray-200 flex flex-col">
            <div className="p-4 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <h3 className="font-medium text-gray-900">Conversations</h3>
                {conversations.length > 0 && (
                  <button
                    onClick={deleteAllConversations}
                    className="text-xs text-red-600 hover:text-red-700"
                  >
                    Delete All
                  </button>
                )}
              </div>
            </div>
            
            <div className="flex-1 overflow-y-auto">
              {isLoading ? (
                <div className="p-4 text-center text-gray-500">Loading...</div>
              ) : conversations.length === 0 ? (
                <div className="p-4 text-center text-gray-500">No conversations yet</div>
              ) : (
                <div className="space-y-1">
                  {conversations.map((conversation) => (
                    <div
                      key={conversation.conversation_id}
                      className={`p-3 cursor-pointer hover:bg-gray-50 border-l-4 ${
                        selectedConversation === conversation.conversation_id
                          ? 'border-blue-500 bg-blue-50'
                          : 'border-transparent'
                      }`}
                      onClick={() => loadMessages(conversation.conversation_id)}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-gray-900 truncate">
                            {conversation.last_message.content.substring(0, 50)}...
                          </p>
                          <p className="text-xs text-gray-500">
                            {conversation.message_count} messages • {formatDate(conversation.created_at)}
                          </p>
                        </div>
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            deleteConversation(conversation.conversation_id)
                          }}
                          className="text-gray-400 hover:text-red-600 ml-2"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Messages */}
          <div className="flex-1 flex flex-col">
            {selectedConversation ? (
              <>
                <div className="p-4 border-b border-gray-200">
                  <h3 className="font-medium text-gray-900">Messages</h3>
                </div>
                <div className="flex-1 overflow-y-auto p-4">
                  {messages.length === 0 ? (
                    <div className="text-center text-gray-500">No messages in this conversation</div>
                  ) : (
                    <div className="space-y-4">
                      {messages.map((message, index) => (
                        <div
                          key={index}
                          className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                        >
                          <div
                            className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                              message.type === 'user'
                                ? 'bg-blue-600 text-white'
                                : 'bg-gray-100 text-gray-900'
                            }`}
                          >
                            <p className="text-sm">{message.content}</p>
                            <p className="text-xs opacity-70 mt-1">
                              {formatDate(message.timestamp)}
                            </p>
                            
                            {/* Show retrieved documents for assistant messages */}
                            {message.type === 'assistant' && message.retrieved_documents && message.retrieved_documents.length > 0 && (
                              <div className="mt-3 pt-3 border-t border-gray-300">
                                <p className="text-xs font-medium text-gray-600 mb-2">
                                  Sources ({message.retrieved_count}):
                                </p>
                                <div className="space-y-2">
                                  {message.retrieved_documents.map((doc: any, docIndex: number) => (
                                    <div
                                      key={docIndex}
                                      className="text-xs bg-gray-50 p-2 rounded border hover:bg-gray-100 cursor-pointer transition-colors"
                                      onClick={() => {
                                        setSelectedDocument(doc)
                                        setIsDocumentViewerOpen(true)
                                      }}
                                      title="Click to read full document"
                                    >
                                      <div className="font-medium text-gray-700 mb-1">
                                        {doc.metadata?.source || 'Unknown Source'}
                                        {doc.metadata?.category && (
                                          <span className="ml-1 text-gray-500">• {doc.metadata.category}</span>
                                        )}
                                      </div>
                                      <div className="text-gray-600 line-clamp-2">
                                        {doc.content.substring(0, 100)}...
                                      </div>
                                      <div className="mt-1 text-xs text-blue-600 opacity-0 group-hover:opacity-100 transition-opacity">
                                        Click to read full document →
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div className="flex-1 flex items-center justify-center text-gray-500">
                <div className="text-center">
                  <MessageSquare className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                  <p>Select a conversation to view messages</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Document Viewer Modal */}
      <DocumentViewer
        isOpen={isDocumentViewerOpen}
        onClose={() => {
          setIsDocumentViewerOpen(false)
          setSelectedDocument(null)
        }}
        document={selectedDocument}
      />
    </div>
  )
}
