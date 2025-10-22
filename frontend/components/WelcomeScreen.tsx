import { Sparkles, MessageSquare } from 'lucide-react'

interface WelcomeScreenProps {
  examples: string[]
  onExampleClick: (example: string) => void
  isLoading: boolean
}

export default function WelcomeScreen({ examples, onExampleClick, isLoading }: WelcomeScreenProps) {
  return (
    <div className="h-full flex items-center justify-center">
      <div className="text-center max-w-md mx-auto px-6">
        <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center mx-auto mb-6">
          <Sparkles className="h-8 w-8 text-white" />
        </div>
        <h2 className="text-2xl font-semibold text-gray-900 mb-2">Welcome to Baby Care Assistant</h2>
        <p className="text-gray-600 mb-8">Ask me anything about pregnancy, baby care, nutrition, or child development.</p>
        
        {/* Example questions grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {examples.slice(0, 4).map((example, index) => (
            <button
              key={index}
              onClick={() => onExampleClick(example)}
              className="p-4 text-left rounded-xl border border-gray-200 hover:border-blue-300 hover:bg-blue-50 transition-all duration-200 group"
              disabled={isLoading}
            >
              <div className="flex items-start space-x-3">
                <MessageSquare className="h-5 w-5 text-gray-400 group-hover:text-blue-500 mt-0.5 flex-shrink-0" />
                <span className="text-sm text-gray-700 group-hover:text-blue-700">{example}</span>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
