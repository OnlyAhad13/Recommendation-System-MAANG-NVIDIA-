"""
FastAPI application for serving recommendations.

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import logging
import sys
from pathlib import Path

# Add parent directory to path to find the 'src' and 'app' modules
# Adjust this path if your project structure is different
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import the new, professional recommendation service
from app.recommendation_service import RecommendationService

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- FastAPI App Initialization ---

app = FastAPI(
    title="Recommendation System API",
    description="Production-ready recommendation API using a trained Two-Tower + DCN model.",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# This global variable will hold our loaded model service
recommendation_service: Optional[RecommendationService] = None


# --- Pydantic Models for Request/Response Validation ---

class RecommendationRequest(BaseModel):
    user_id: str = Field(..., description="User ID to get recommendations for")
    k: int = Field(10, ge=1, le=100, description="Number of recommendations to return")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "1", # Use an example user_id from your vocabulary
                "k": 10
            }
        }


class RecommendationItem(BaseModel):
    item_id: str
    score: float
    rank: int


class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[RecommendationItem]
    count: int
    model_version: str


class ScoreRequest(BaseModel):
    user_id: str
    item_ids: List[str] = Field(..., min_length=1, max_length=100)

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "1",
                "item_ids": ["25", "120", "255"] # Use item_ids from your vocabulary
            }
        }


class ScoreResponse(BaseModel):
    user_id: str
    scores: Dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None


# --- FastAPI Lifespan Events (Startup/Shutdown) ---

@app.on_event("startup")
async def startup_event():
    """Load the recommendation model when the application starts."""
    global recommendation_service
    logger.info("Starting up the recommendation service...")
    
    try:
        # NOTE: Ensure this path points to your trained model artifacts
        model_directory = "outputs/models/experiment_001"
        recommendation_service = RecommendationService(model_dir=model_directory)
        recommendation_service.load()
        logger.info("✅ Recommendation service is fully loaded and ready.")
    except Exception as e:
        logger.error(f"💥 Failed to load model during startup: {e}")
        # The service will be 'None', and endpoints will return a 503 error.


@app.on_event("shutdown")
async def shutdown_event():
    """Perform cleanup on shutdown."""
    global recommendation_service
    logger.info("Shutting down recommendation service...")
    recommendation_service = None


# --- API Endpoints ---

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check if the service is healthy and the model is loaded."""
    is_ready = recommendation_service is not None and recommendation_service.is_ready()
    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        model_loaded=is_ready,
        model_version=recommendation_service.get_model_info()['version'] if is_ready else None
    )


@app.get("/", tags=["Info"], include_in_schema=False)
async def root():
    """Root endpoint with basic API information."""
    return {
        "message": "Welcome to the Recommendation System API",
        "status": "running",
        "documentation": "/docs"
    }


@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(request: RecommendationRequest):
    """Get personalized top-K recommendations for a user."""
    if not recommendation_service or not recommendation_service.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded. Service is unavailable.")
    
    try:
        recs = recommendation_service.recommend(user_id=request.user_id, k=request.k)
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recs,
            count=len(recs),
            model_version=recommendation_service.get_model_info()['version']
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) # 404 for 'user not found'
    except Exception as e:
        logger.error(f"Error during recommendation for user '{request.user_id}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/score", response_model=ScoreResponse, tags=["Scoring"])
async def score_items(request: ScoreRequest):
    """Score a specific list of items for a given user."""
    if not recommendation_service or not recommendation_service.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded. Service is unavailable.")
    
    try:
        scores = recommendation_service.score(user_id=request.user_id, item_ids=request.item_ids)
        return ScoreResponse(user_id=request.user_id, scores=scores)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error during scoring for user '{request.user_id}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """Get metadata about the currently loaded recommendation model."""
    if not recommendation_service or not recommendation_service.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded. Service is unavailable.")
    
    return recommendation_service.get_model_info()


if __name__ == "__main__":
    import uvicorn
    # This block allows running the script directly for development
    # For production, use a process manager like Gunicorn:
    # gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)