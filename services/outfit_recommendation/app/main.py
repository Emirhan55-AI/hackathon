"""
Outfit Recommendation Service - Main FastAPI Application
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import sys
import os

# Add shared modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from shared.config.settings import get_settings
from shared.config.logging import get_logger
from app.api.recommendations import router as recommendations_router
from app.api.style import router as style_router
from app.core.inference import recommendation_engine

settings = get_settings()
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Outfit Recommendation Service...")
    await recommendation_engine.initialize()
    logger.info("Outfit Recommendation Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Outfit Recommendation Service...")

# Create FastAPI app
app = FastAPI(
    title="Aura Outfit Recommendation Service",
    description="AI-powered outfit recommendation service using OutfitTransformer",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

# Include routers
app.include_router(
    recommendations_router,
    prefix="/recommendations",
    tags=["recommendations"]
)

app.include_router(
    style_router,
    prefix="/style",
    tags=["style"]
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Aura Outfit Recommendation Service",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "outfit_recommendation",
        "model_loaded": recommendation_engine.is_initialized
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8002,
        reload=settings.DEBUG
    )
