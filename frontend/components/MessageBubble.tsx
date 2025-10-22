import { Bot, User, FileText } from 'lucide-react'

interface MessageBubbleProps {
  message: {
    id: string
    content: string
    isUser: boolean
    timestamp: Date
    retrievedDocuments?: DocumentInfo[]
    retrievedCount?: number
  }
  onDocumentClick?: (document: DocumentInfo) => void
}

interface DocumentInfo {
  content: string
  metadata: {
    source?: string
    category?: string
    [key: string]: any
  }
}

export default function MessageBubble({ message, onDocumentClick }: MessageBubbleProps) {
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }

  return (
    <div className={`flex mb-6 ${message.isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`flex max-w-3xl ${message.isUser ? 'flex-row-reverse' : 'flex-row'} items-start space-x-3`}>
        {/* Avatar */}
        <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
          message.isUser 
            ? 'bg-blue-600' 
            : 'bg-gradient-to-br from-blue-500 to-purple-600'
        }`}>
          {message.isUser ? (
            <User className="h-4 w-4 text-white" />
          ) : (
            <Bot className="h-4 w-4 text-white" />
          )}
        </div>
        
        {/* Message content */}
        <div className={`flex-1 ${message.isUser ? 'text-right' : 'text-left'}`}>
          <div className={`inline-block rounded-2xl px-4 py-3 message-bubble ${
            message.isUser 
              ? 'bg-blue-600 text-white' 
              : 'bg-white border border-gray-200 shadow-sm'
          }`}>
            <p className="whitespace-pre-wrap text-sm leading-relaxed">{message.content}</p>
            
            {/* Show retrieved documents for AI messages */}
            {!message.isUser && message.retrievedDocuments && message.retrievedDocuments.length > 0 && (
              <div className="mt-4 pt-4 border-t border-gray-200">
                <div className="flex items-center space-x-2 mb-3">
                  <FileText className="h-4 w-4 text-gray-500" />
                  <span className="text-xs text-gray-600 font-medium">
                    Used {message.retrievedCount} sources
                  </span>
                </div>
                <div className="space-y-2 max-h-32 overflow-y-auto">
                  {message.retrievedDocuments.map((doc, index) => (
                    <div 
                      key={index} 
                      className="text-xs bg-gray-50 p-3 rounded-lg border hover:bg-gray-100 hover:border-blue-300 cursor-pointer transition-all duration-200 group"
                      onClick={() => onDocumentClick?.(doc)}
                    >
                      <div className="font-medium text-gray-700 mb-1 group-hover:text-blue-700">
                        {doc.metadata.source || `Source ${index + 1}`}
                      </div>
                      <div className="text-gray-600 line-clamp-2 group-hover:text-gray-800">
                        {doc.content.substring(0, 150)}...
                      </div>
                      <div className="mt-2 text-xs text-blue-600 opacity-0 group-hover:opacity-100 transition-opacity">
                        Click to read full document →
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
          
          {/* Timestamp */}
          <div className={`text-xs text-gray-500 mt-1 ${message.isUser ? 'text-right' : 'text-left'}`}>
            {formatTime(message.timestamp)}
          </div>
        </div>
      </div>
    </div>
  )
}
