"""
FastAPI application for serving recommendations.

Usage:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.simple_model_loader import SimpleRecommendationService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Recommendation System API",
    description="Production-ready recommendation API using Two-Tower + DCN architecture",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize recommendation service (lazy loading)
recommendation_service: Optional[SimpleRecommendationService] = None


# Request/Response Models
class RecommendationRequest(BaseModel):
    user_id: str = Field(..., description="User ID to get recommendations for")
    k: int = Field(10, ge=1, le=100, description="Number of recommendations to return")
    exclude_seen: bool = Field(True, description="Exclude items user has already interacted with")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "123",
                "k": 10,
                "exclude_seen": True
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
    item_ids: List[str] = Field(..., min_items=1, max_items=100)

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "123",
                "item_ids": ["456", "789", "101"]
            }
        }


class ScoreResponse(BaseModel):
    user_id: str
    scores: Dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None


# Startup event
@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global recommendation_service
    logger.info("Starting recommendation service...")
    
    try:
        recommendation_service = SimpleRecommendationService()
        recommendation_service.load_model()
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        # Don't fail startup, but model won't be available


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down recommendation service...")


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check if the service is healthy and model is loaded."""
    return HealthResponse(
        status="healthy" if recommendation_service and recommendation_service.is_ready() else "degraded",
        model_loaded=recommendation_service is not None and recommendation_service.is_ready(),
        model_version=recommendation_service.get_version() if recommendation_service else None
    )


# Root endpoint
@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Recommendation System API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


# Recommendation endpoint
@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(request: RecommendationRequest):
    """
    Get personalized recommendations for a user.
    
    Returns top-K items ranked by predicted user preference.
    """
    if not recommendation_service or not recommendation_service.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")
    
    try:
        recommendations = recommendation_service.recommend(
            user_id=request.user_id,
            k=request.k,
            exclude_seen=request.exclude_seen
        )
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            count=len(recommendations),
            model_version=recommendation_service.get_version()
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Scoring endpoint
@app.post("/score", response_model=ScoreResponse, tags=["Scoring"])
async def score_items(request: ScoreRequest):
    """
    Score specific user-item pairs.
    
    Useful for re-ranking or evaluating specific items for a user.
    """
    if not recommendation_service or not recommendation_service.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")
    
    try:
        scores = recommendation_service.score(
            user_id=request.user_id,
            item_ids=request.item_ids
        )
        
        return ScoreResponse(
            user_id=request.user_id,
            scores=scores
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error scoring items: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Batch recommendations endpoint
@app.post("/recommend/batch", tags=["Recommendations"])
async def get_batch_recommendations(
    user_ids: List[str] = Query(..., max_items=100),
    k: int = Query(10, ge=1, le=100)
):
    """
    Get recommendations for multiple users (batch processing).
    
    Limited to 100 users per request.
    """
    if not recommendation_service or not recommendation_service.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")
    
    try:
        results = recommendation_service.recommend_batch(user_ids=user_ids, k=k)
        
        return {
            "count": len(results),
            "recommendations": results,
            "model_version": recommendation_service.get_version()
        }
    
    except Exception as e:
        logger.error(f"Error in batch recommendations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Model info endpoint
@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """Get information about the loaded model."""
    if not recommendation_service or not recommendation_service.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return recommendation_service.get_model_info()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
