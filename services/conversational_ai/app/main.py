"""
Conversational AI Service - Main FastAPI Application
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import sys
import os

# Add shared modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

try:
    from shared.config.settings import get_settings
    from shared.config.logging import get_logger
except ImportError:
    # Fallback for container environment
    import logging
    def get_settings():
        return None
    def get_logger(name):
        return logging.getLogger(name)

try:
    from app.api.chat import router as chat_router
    from app.api.knowledge import router as knowledge_router
    from app.api.personalization import router as personalization_router
    from app.core.inference import conversation_engine
except ImportError:
    # Fallback - create dummy routers
    from fastapi import APIRouter
    chat_router = APIRouter()
    knowledge_router = APIRouter()
    personalization_router = APIRouter()
    conversation_engine = None

settings = get_settings()
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Conversational AI Service...")
    await conversation_engine.initialize()
    logger.info("Conversational AI Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Conversational AI Service...")

# Create FastAPI app
app = FastAPI(
    title="Aura Conversational AI Service",
    description="AI-powered fashion conversation service using RAG and fine-tuned LLM",
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
    chat_router,
    prefix="/chat",
    tags=["chat"]
)

app.include_router(
    knowledge_router,
    prefix="/knowledge",
    tags=["knowledge"]
)

app.include_router(
    personalization_router,
    prefix="/personalization",
    tags=["personalization"]
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Aura Conversational AI Service",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "conversational_ai",
        "model_loaded": conversation_engine.is_initialized
    }

@app.websocket("/ws/chat/{user_id}")
async def websocket_chat(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    logger.info(f"WebSocket connection established for user {user_id}")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message = data.get("message", "")
            
            if not message:
                await websocket.send_json({
                    "error": "Empty message received"
                })
                continue
            
            # Process message through conversation engine
            response = await conversation_engine.process_message(
                user_id=user_id,
                message=message,
                context=data.get("context", {})
            )
            
            # Send response back to client
            await websocket.send_json({
                "response": response["response"],
                "confidence": response.get("confidence", 1.0),
                "sources": response.get("sources", []),
                "timestamp": response.get("timestamp")
            })
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket connection closed for user {user_id}")
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {str(e)}")
        await websocket.send_json({
            "error": "An error occurred processing your message"
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8003,
        reload=settings.DEBUG
    )
