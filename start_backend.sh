#!/bin/bash

# Start FastAPI Backend with Authentication
echo "🚀 Starting Baby Care Assistant Backend with Authentication..."
echo "Backend will be available at: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found. Please create one with your API keys."
    echo "Required variables:"
    echo "  - OPENAI_API_KEY"
    echo "  - PINECONE_API_KEY"
    echo "  - GOOGLE_CLIENT_ID"
    echo "  - GOOGLE_CLIENT_SECRET"
    echo "  - JWT_SECRET_KEY"
    echo ""
fi

# Check if MongoDB is running (optional, will use SQLite if not available)
if ! pgrep -x "mongod" > /dev/null; then
    echo "ℹ️  MongoDB not detected. Chat history will be stored in SQLite."
    echo "   For production, consider setting up MongoDB for better performance."
    echo ""
fi

# Start the FastAPI server
echo "Starting server with authentication and database support..."
uv run python -m uvicorn main:app --host 0.0.0.0 --port 8000