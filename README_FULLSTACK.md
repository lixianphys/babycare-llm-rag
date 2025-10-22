# Baby Care Assistant - Full Stack Setup

This project has been migrated from Gradio to a full-stack architecture using FastAPI (backend) and Next.js (frontend).

## Architecture

```
├── main.py                 # FastAPI backend
├── babycare/              # Core chatbot logic
├── frontend/              # Next.js frontend
│   ├── app/
│   │   ├── page.tsx       # Main chat interface
│   │   ├── layout.tsx     # App layout
│   │   └── globals.css    # Global styles
│   ├── package.json       # Frontend dependencies
│   └── tailwind.config.js # Tailwind configuration
└── pyproject.toml         # Backend dependencies
```

## Quick Start

### 1. Backend (FastAPI)

```bash
# Install dependencies
uv sync

# Start the FastAPI server (recommended)
./dev_backend.sh

# Alternative: Direct Python execution
uv run python main.py

# Alternative: Production mode (no auto-reload)
./start_backend.sh
```

The API will be available at `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

### 2. Frontend (Next.js)

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will be available at `http://localhost:3000`

## API Endpoints

### Chat
- `POST /chat` - Send a message and get AI response
- `GET /examples` - Get example questions
- `GET /knowledge-base` - Get knowledge base information
- `GET /health` - Health check

### Example API Usage

```bash
# Send a chat message
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What should I feed my 6 month old baby?",
    "conversation_id": "test_session",
    "user_id": "test_user"
  }'

# Get example questions
curl "http://localhost:8000/examples"

# Health check
curl "http://localhost:8000/health"
```

## Environment Variables

Create a `.env` file in the root directory:

```env
# OpenAI API
OPENAI_API_KEY=your_openai_api_key

# Pinecone (if using vector store)
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment

# Optional: Force OpenAI embeddings for Hugging Face Spaces
FORCE_OPENAI_EMBEDDINGS=false
```

## Development

### Backend Development
```bash
# Recommended: Use the development script
./dev_backend.sh

# Alternative: Direct Python execution
uv run python main.py
```

### Frontend Development
```bash
cd frontend
npm run dev
```

## Deployment

### Backend Deployment
The FastAPI backend can be deployed to:
- **Railway**: Connect your GitHub repo
- **Render**: Deploy as a web service
- **Heroku**: Use the Procfile
- **Docker**: Use the provided Dockerfile

### Frontend Deployment
The Next.js frontend can be deployed to:
- **Vercel**: Connect your GitHub repo
- **Netlify**: Deploy from build folder
- **Railway**: Deploy as a static site

## Key Features

### Backend (FastAPI)
- ✅ RESTful API with automatic documentation
- ✅ CORS configured for frontend
- ✅ Health check endpoints
- ✅ Streaming chat responses
- ✅ Document retrieval information
- ✅ Example questions endpoint

### Frontend (Next.js)
- ✅ Modern React with TypeScript
- ✅ Tailwind CSS for styling
- ✅ Real-time chat interface
- ✅ Typing indicators
- ✅ Document display
- ✅ Example question buttons
- ✅ Responsive design

## Migration from Gradio

The following files have been removed/replaced:
- ❌ `app.py` (Gradio interface) → ✅ `main.py` (FastAPI backend)
- ❌ `styles.css` (Gradio styles) → ✅ `frontend/app/globals.css` (Tailwind)
- ❌ Gradio dependencies → ✅ FastAPI + Next.js dependencies

## Benefits of Full-Stack Architecture

1. **Separation of Concerns**: Backend handles AI logic, frontend handles UI
2. **Scalability**: Can scale backend and frontend independently
3. **Flexibility**: Easy to add new features or change UI frameworks
4. **API-First**: Backend can serve multiple frontends (web, mobile, etc.)
5. **Modern Development**: Use latest React patterns and FastAPI features
6. **Better Performance**: Optimized for production deployment
7. **Team Collaboration**: Frontend and backend developers can work independently

## Next Steps

1. **Add Authentication**: Implement user authentication
2. **Database Integration**: Add user sessions and chat history
3. **Real-time Features**: Add WebSocket support for real-time updates
4. **Mobile App**: Create React Native mobile app using the same API
5. **Analytics**: Add usage analytics and monitoring
6. **Testing**: Add comprehensive test suites for both frontend and backend
