import { useState } from 'react'
import { FileText, Eye } from 'lucide-react'
import DocumentViewer from './DocumentViewer'

interface DocumentViewerDemoProps {
  className?: string
}

export default function DocumentViewerDemo({ className = '' }: DocumentViewerDemoProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [selectedDoc, setSelectedDoc] = useState(null)

  const sampleDocuments = [
    {
      content: `# Baby Sleep Patterns

## Understanding Newborn Sleep

Newborns typically sleep 14-17 hours per day, but this sleep is distributed in short periods throughout the day and night. Here's what you need to know:

### Sleep Cycles
- Newborns have shorter sleep cycles (50-60 minutes) compared to adults (90 minutes)
- They spend more time in REM (rapid eye movement) sleep, which is lighter sleep
- Deep sleep periods are shorter and less frequent

### Common Sleep Patterns
1. **Cluster Feeding**: Babies often wake every 2-3 hours to feed
2. **Day/Night Confusion**: Newborns don't distinguish between day and night initially
3. **Growth Spurts**: Sleep patterns change during growth spurts

### Tips for Better Sleep
- Establish a consistent bedtime routine
- Keep the room dark and quiet at night
- Use white noise or gentle music
- Swaddle your baby for comfort
- Be patient - sleep patterns improve over time

Remember: Every baby is different, and sleep patterns vary significantly in the first few months.`,
      metadata: {
        source: 'Baby Sleep Guide - MotherToBaby.org',
        category: 'Sleep & Development',
        topic: 'Newborn Sleep Patterns'
      }
    },
    {
      content: `# Safe Sleep Guidelines

## Creating a Safe Sleep Environment

The American Academy of Pediatrics recommends the following safe sleep practices:

### Sleep Position
- Always place babies on their back to sleep
- Use a firm, flat sleep surface
- Avoid inclined sleepers or soft surfaces

### Sleep Environment
- Keep the crib free of loose bedding, pillows, and stuffed animals
- Use a fitted sheet only
- Maintain room temperature between 68-72°F (20-22°C)
- Ensure good air circulation

### What to Avoid
- Co-sleeping in adult beds
- Soft bedding and bumper pads
- Overheating the baby
- Smoking around the baby
- Alcohol or drug use by caregivers

### Monitoring
- Use a baby monitor if needed
- Check on your baby regularly
- Trust your instincts if something seems wrong

These guidelines help reduce the risk of Sudden Infant Death Syndrome (SIDS) and other sleep-related infant deaths.`,
      metadata: {
        source: 'Safe Sleep Guidelines - AAP',
        category: 'Safety',
        topic: 'Safe Sleep Practices'
      }
    }
  ]

  const handleDocumentClick = (doc: any) => {
    setSelectedDoc(doc)
    setIsOpen(true)
  }

  return (
    <div className={`p-6 bg-white rounded-lg border ${className}`}>
      <h3 className="text-lg font-semibold mb-4 flex items-center">
        <FileText className="h-5 w-5 mr-2" />
        Document Viewer Demo
      </h3>
      
      <p className="text-gray-600 mb-4">
        Click on any document below to open it in the side panel viewer:
      </p>

      <div className="space-y-3">
        {sampleDocuments.map((doc, index) => (
          <div 
            key={index}
            className="p-4 border border-gray-200 rounded-lg hover:border-blue-300 hover:bg-blue-50 cursor-pointer transition-all duration-200 group"
            onClick={() => handleDocumentClick(doc)}
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <h4 className="font-medium text-gray-900 group-hover:text-blue-700">
                  {doc.metadata.topic}
                </h4>
                <p className="text-sm text-gray-600 mt-1">
                  {doc.metadata.source}
                </p>
                <p className="text-xs text-gray-500 mt-2">
                  {doc.content.substring(0, 100)}...
                </p>
              </div>
              <Eye className="h-4 w-4 text-gray-400 group-hover:text-blue-500 mt-1" />
            </div>
          </div>
        ))}
      </div>

      <DocumentViewer
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
        document={selectedDoc}
      />
    </div>
  )
}
