"""
Conversational AI API Module
============================

Bu modül, Aura Conversational AI servisinin FastAPI endpoint'lerini içerir.

Modules:
    main: Ana FastAPI uygulaması ve endpoint'ler
    
Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8003
    python -m api.main
    
Features:
    - Hibrit QLoRA + RAG chat endpoints
    - WebSocket real-time chat support
    - Batch processing capabilities
    - Comprehensive error handling
    - Health monitoring
    
Author: Aura AI Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Aura AI Team"

# Import main FastAPI app for easy access
try:
    from .main import app, global_rag_service
    MAIN_APP_AVAILABLE = True
except ImportError as e:
    MAIN_APP_AVAILABLE = False
    app = None
    global_rag_service = None
    print(f"Warning: Main app import failed: {e}")

__all__ = [
    "app",
    "global_rag_service"
]
