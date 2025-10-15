# 🚀 Recommendation System API

FastAPI-based REST API for serving personalized recommendations.

## 📋 Features

- ✅ **GET /recommend** - Get personalized recommendations for a user
- ✅ **POST /score** - Score specific user-item pairs
- ✅ **POST /recommend/batch** - Batch recommendations for multiple users
- ✅ **GET /health** - Health check endpoint
- ✅ **GET /model/info** - Model information
- ✅ **Interactive API docs** at `/docs`

## 🚀 Quick Start

### **Option 1: Run Locally**

```bash
# 1. Install dependencies
pip install -r app/requirements.txt

# 2. Run the API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 3. Access the API
# - Swagger UI: http://localhost:8000/docs
# - ReDoc: http://localhost:8000/redoc
# - API: http://localhost:8000
```

### **Option 2: Run with Docker**

```bash
# Build and run
docker-compose -f app/docker-compose.yml up --build

# Or using Docker directly
docker build -f app/Dockerfile -t recsys-api .
docker run -p 8000:8000 recsys-api
```

## 📖 API Usage Examples

### **1. Health Check**

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0"
}
```

### **2. Get Recommendations**

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "123",
    "k": 10,
    "exclude_seen": true
  }'
```

Response:
```json
{
  "user_id": "123",
  "recommendations": [
    {"item_id": "456", "score": 0.92, "rank": 1},
    {"item_id": "789", "score": 0.87, "rank": 2},
    ...
  ],
  "count": 10,
  "model_version": "1.0.0"
}
```

### **3. Score Specific Items**

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type": application/json" \
  -d '{
    "user_id": "123",
    "item_ids": ["456", "789", "101"]
  }'
```

Response:
```json
{
  "user_id": "123",
  "scores": {
    "456": 0.92,
    "789": 0.87,
    "101": 0.65
  }
}
```

### **4. Batch Recommendations**

```bash
curl -X POST "http://localhost:8000/recommend/batch?user_ids=123&user_ids=456&k=5"
```

### **5. Model Information**

```bash
curl http://localhost:8000/model/info
```

## 🐍 Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000"

# Get recommendations
response = requests.post(
    f"{BASE_URL}/recommend",
    json={
        "user_id": "123",
        "k": 10,
        "exclude_seen": True
    }
)

recommendations = response.json()
print(f"Got {recommendations['count']} recommendations")
for rec in recommendations['recommendations']:
    print(f"  Rank {rec['rank']}: Item {rec['item_id']} (score: {rec['score']:.3f})")
```

## 📊 Performance

- **Latency**: <50ms p95 for single user
- **Throughput**: ~1000 requests/second (single instance)
- **Cold-start handling**: Falls back to popular items

## 🔧 Configuration

Environment variables:

```bash
MODEL_DIR=/app/outputs/models/experiment_001  # Path to model directory
LOG_LEVEL=INFO  # Logging level
PORT=8000  # API port
```

## 📝 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive API documentation |
| `/recommend` | POST | Get user recommendations |
| `/score` | POST | Score user-item pairs |
| `/recommend/batch` | POST | Batch recommendations |
| `/model/info` | GET | Model metadata |

## 🐳 Docker Deployment

```bash
# Development
docker-compose -f app/docker-compose.yml up

# Production (with custom config)
docker run -d \
  -p 8000:8000 \
  -e MODEL_DIR=/app/outputs/models/experiment_001 \
  -v $(pwd)/outputs:/app/outputs:ro \
  --name recsys-api \
  recsys-api:latest
```

## 🧪 Testing

```bash
# Run test script
python app/test_api.py

# Or use pytest
pytest app/tests/
```

## 📈 Monitoring

The API exposes metrics for monitoring:

- Request latency
- Error rates
- Model load status
- Memory usage

Integrate with Prometheus/Grafana for production monitoring.

## 🔒 Security Notes

**For Production:**

1. Add authentication (API keys, OAuth)
2. Enable rate limiting
3. Use HTTPS
4. Set up CORS properly
5. Add input validation
6. Monitor for anomalies

## 📄 License

MIT License
