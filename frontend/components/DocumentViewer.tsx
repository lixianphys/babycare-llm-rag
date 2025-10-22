import { X, FileText, ExternalLink } from 'lucide-react'
import { useEffect } from 'react'

interface DocumentViewerProps {
  isOpen: boolean
  onClose: () => void
  document: {
    content: string
    metadata: {
      source?: string
      category?: string
      [key: string]: any
    }
  } | null
}

export default function DocumentViewer({ isOpen, onClose, document }: DocumentViewerProps) {
  // Handle escape key to close
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose()
      }
    }

    if (isOpen) {
      window.document.addEventListener('keydown', handleEscape)
      return () => window.document.removeEventListener('keydown', handleEscape)
    }
  }, [isOpen, onClose])

  if (!isOpen || !document) return null

  return (
    <>
      {/* Backdrop */}
      <div 
        className="fixed inset-0 bg-black bg-opacity-50 z-40 transition-opacity"
        onClick={onClose}
      />
      
      {/* Side Panel */}
      <div className="fixed right-0 top-0 h-full w-96 bg-white shadow-2xl z-50 transform transition-transform duration-300 ease-in-out">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200 bg-gray-50">
          <div className="flex items-center space-x-2">
            <FileText className="h-5 w-5 text-gray-600" />
            <div>
              <h3 className="text-lg font-semibold text-gray-900">Document Source</h3>
              <p className="text-sm text-gray-600">{document.metadata.source || 'Unknown Source'}</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-200 rounded-lg transition-colors"
          >
            <X className="h-5 w-5 text-gray-600" />
          </button>
        </div>

        {/* Metadata */}
        <div className="p-4 border-b border-gray-200 bg-gray-50">
          <div className="space-y-2">
            {document.metadata.category && (
              <div className="flex items-center space-x-2">
                <span className="text-sm font-medium text-gray-700">Category:</span>
                <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
                  {document.metadata.category}
                </span>
              </div>
            )}
            {document.metadata.source && (
              <div className="flex items-center space-x-2">
                <span className="text-sm font-medium text-gray-700">Source:</span>
                <span className="text-sm text-gray-600">{document.metadata.source}</span>
              </div>
            )}
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          <div className="prose prose-sm max-w-none">
            <div className="text-sm leading-relaxed text-gray-800 whitespace-pre-wrap">
              {document.content}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-gray-200 bg-gray-50">
          <div className="flex items-center justify-between">
            <div className="text-xs text-gray-500">
              {document.content.length} characters
            </div>
            <button
              onClick={() => {
                // Copy to clipboard
                navigator.clipboard.writeText(document.content)
                // You could add a toast notification here
              }}
              className="flex items-center space-x-1 px-3 py-1 text-xs text-blue-600 hover:text-blue-800 hover:bg-blue-50 rounded transition-colors"
            >
              <ExternalLink className="h-3 w-3" />
              <span>Copy</span>
            </button>
          </div>
        </div>
      </div>
    </>
  )
}
