"""
Main FastAPI application for Visual Analysis Service
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import time
import sys
import os

# Add shared modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from shared.config.settings import settings
from shared.config.logging import setup_logging, get_logger
from shared.models.base import HealthResponse, ServiceStatus

# Import routers
from app.api.endpoints import router as api_router

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Visual Analysis Service",
    description="DETR-based fashion image analysis service",
    version=settings.version,
    debug=settings.api.debug
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Include API routes
app.include_router(api_router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info(f"Starting {settings.service_name} service")
    # Initialize models, database connections, etc.

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info(f"Shutting down {settings.service_name} service")
    # Cleanup resources

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status=ServiceStatus.HEALTHY,
        service_name=settings.service_name,
        version=settings.version,
        dependencies={
            "database": ServiceStatus.HEALTHY,
            "redis": ServiceStatus.HEALTHY,
            "model": ServiceStatus.HEALTHY
        }
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Visual Analysis Service",
        "version": settings.version,
        "status": "running"
    }
