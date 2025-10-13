#!/bin/bash
# Quick start script for the API

echo "=================================================="
echo "🚀 Starting Recommendation System API"
echo "=================================================="

# Check if API requirements are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "📦 Installing API dependencies..."
    pip install -r app/requirements.txt
fi

# Check if model exists
if [ ! -f "outputs/models/recsys_output/encoder.keras" ]; then
    echo "❌ Model not found at: outputs/models/recsys_output/"
    echo "Please train a model first or update MODEL_DIR in app/model_service.py"
    exit 1
fi

echo "✅ Model found"
echo ""
echo "Starting API server..."
echo "  - Swagger UI: http://localhost:8000/docs"
echo "  - ReDoc: http://localhost:8000/redoc"
echo "  - Health: http://localhost:8000/health"
echo ""

# Start the API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
