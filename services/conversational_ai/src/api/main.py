"""
Aura Conversational AI Service - FastAPI Main Application
=========================================================

Bu modül, hibrit QLoRA + RAG sohbet asistanını FastAPI web sunucusu olarak sunar.
RAG Service'i HTTP API endpoints aracılığıyla erişilebilir kılar.

Features:
- Hibrit QLoRA + RAG pipeline
- Real-time chat endpoints
- Personalized fashion advice
- WebSocket support for live chat
- User wardrobe integration
- Comprehensive error handling
"""

import os
import sys
import json
import logging
import asyncio
import time
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import traceback
from contextlib import asynccontextmanager

# A1. Modül ve Kütüphane İçe Aktarma
# FastAPI ve ilgili kütüphaneler
from fastapi import (
    FastAPI, 
    HTTPException, 
    Depends,
    BackgroundTasks,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

# Pydantic models for request/response validation
from pydantic import BaseModel, Field, validator
from pydantic.json import pydantic_encoder

# HTTP ve async operations
import uvicorn

# Parent directory'yi path'e ekle (RAG service modüllerini import edebilmek için)
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Local imports - RAG Service modülü
try:
    from src.rag_service import RAGService, RAGConfig, create_rag_service
    from src.rag_config_examples import get_rag_config
    from src.rag_service_manager import create_rag_service_manager, get_rag_service_manager, RAGServiceManager
    from src.rag_service_resource import RAGServiceResource
    from src.logging_config import (
        setup_logging, 
        StructuredLoggingMiddleware,
        get_structured_logger,
        log_service_event,
        log_performance_metric,
        log_error_with_context,
        api_logger
    )
except ImportError as e:
    # Fallback import strategy
    try:
        from rag_service import RAGService, RAGConfig, create_rag_service
        from rag_config_examples import get_rag_config
        from rag_service_manager import create_rag_service_manager, get_rag_service_manager, RAGServiceManager
        from rag_service_resource import RAGServiceResource
        from logging_config import (
            setup_logging, 
            StructuredLoggingMiddleware,
            get_structured_logger,
            log_service_event,
            log_performance_metric,
            log_error_with_context,
            api_logger
        )
    except ImportError:
        # Development import
        sys.path.insert(0, str(parent_dir.parent))
        from conversational_ai.src.rag_service import RAGService, RAGConfig, create_rag_service
        from conversational_ai.src.rag_config_examples import get_rag_config
        from conversational_ai.src.rag_service_manager import create_rag_service_manager, get_rag_service_manager, RAGServiceManager
        from conversational_ai.src.rag_service_resource import RAGServiceResource
        from conversational_ai.src.logging_config import (
            setup_logging, 
            StructuredLoggingMiddleware,
            get_structured_logger,
            log_service_event,
            log_performance_metric,
            log_error_with_context,
            api_logger
        )

# Structured logging configuration
setup_logging()
logger = get_structured_logger("api", service="conversational_ai")

# Global RAG service manager
rag_service_manager: Optional[RAGServiceManager] = None

# API Configuration
API_TITLE = "Aura Conversational AI API"
API_DESCRIPTION = """
🤖 **Aura Yapay Zeka Platformu - Conversational AI Mikroservisi**

Bu API, hibrit QLoRA + RAG sistemi kullanarak kişiselleştirilmiş 
fashion sohbet asistanı hizmeti sunar.

## Özellikler:
- **Personalized Chat**: Kullanıcı gardırobuna özel öneriler
- **Fashion Expertise**: Fine-tuned LLaMA model ile moda uzmanlığı
- **Real-time Knowledge**: Vector store ile güncel gardırop bilgisi
- **Context Awareness**: Conversation history ve user preferences
- **WebSocket Support**: Real-time chat experience
- **Batch Processing**: Çoklu mesaj desteği

## Sistem Mimarisi:
```
User Query → Query Embedding → Vector Search → Context → LLM → Response
    ↓             ↓               ↓            ↓        ↓        ↓
  "Ne giysem?"  SentTrans      FAISS/Pine   Format   LLaMA   "Mavi ceket..."
```

## Desteklenen İşlemler:
- Kıyafet önerileri ve kombinasyon tavsiyeleri
- Renk uyumu ve stil danışmanlığı
- Mevsim ve etkinlik bazlı öneriler
- Gardırop analizi ve eksik parça tespiti
- Alışveriş önerileri ve trend analizi
"""
API_VERSION = "1.0.0"

# Request/Response Models
class ChatRequest(BaseModel):
    """Sohbet isteği için Pydantic modeli"""
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Kullanıcının sohbet mesajı",
        example="Bugün iş toplantısı var, ne giysem iyi olur?"
    )
    user_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Kullanıcının benzersiz kimliği",
        example="user123"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Sohbet oturumu kimliği",
        example="session_abc123"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Ek bağlam bilgileri (mevsim, etkinlik, vs.)",
        example={"season": "winter", "occasion": "business"}
    )
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty or whitespace only')
        return v.strip()


class ChatResponse(BaseModel):
    """Sohbet yanıtı için Pydantic modeli"""
    success: bool = Field(description="İşlem başarı durumu")
    response: str = Field(description="Asistan yanıtı")
    user_id: str = Field(description="Kullanıcı kimliği")
    session_id: Optional[str] = Field(description="Oturum kimliği")
    context_used: List[Dict[str, Any]] = Field(description="Kullanılan bağlam bilgileri")
    confidence: float = Field(description="Yanıt güven skoru")
    suggestions: Optional[List[str]] = Field(
        default=None,
        description="Ek öneriler veya sorular"
    )
    metadata: Dict[str, Any] = Field(description="İşlem metadata'sı")


class BatchChatRequest(BaseModel):
    """Toplu sohbet isteği"""
    messages: List[ChatRequest] = Field(
        ...,
        max_items=10,
        description="Toplu mesaj listesi (max 10)"
    )


class BatchChatResponse(BaseModel):
    """Toplu sohbet yanıtı"""
    success: bool = Field(description="Toplu işlem durumu")
    responses: List[ChatResponse] = Field(description="Yanıt listesi")
    processed_count: int = Field(description="İşlenen mesaj sayısı")
    failed_count: int = Field(description="Başarısız mesaj sayısı")
    processing_time: float = Field(description="Toplam işlem süresi")


class HealthResponse(BaseModel):
    """Sağlık kontrolü yanıtı"""
    status: str = Field(description="Servis durumu")
    service: str = Field(description="Servis adı")
    model_loaded: bool = Field(description="Model yüklenme durumu")
    version: str = Field(description="API versiyonu")
    rag_service_stats: Dict[str, Any] = Field(description="RAG servis istatistikleri")
    timestamp: str = Field(description="Zaman damgası")


class ErrorResponse(BaseModel):
    """Hata yanıtı"""
    error: str = Field(description="Hata mesajı")
    details: Optional[str] = Field(default=None, description="Hata detayları")
    request_id: Optional[str] = Field(default=None, description="İstek kimliği")
    timestamp: str = Field(description="Hata zamanı")


# Structured logging middleware has been moved to logging_config.py
# Using StructuredLoggingMiddleware instead of custom LoggingMiddleware


# A2. FastAPI Uygulaması ve RAG Servisinin Yüklenmesi
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    A2. Uygulama yaşam döngüsü yönetimi
    RAG Service Manager oluşturma ve temizleme işlemleri
    """
    global rag_service_manager
    
    log_service_event(logger, "application_startup", service="conversational_ai", version=API_VERSION)
    
    # Startup: RAG Service Manager oluşturma
    try:
        # Environment variables'dan config ayarları
        config_name = os.getenv("RAG_CONFIG", "basic")  # basic, production, fast_inference vb.
        model_path = os.getenv("FINETUNED_MODEL_PATH", "/app/models/aura_fashion_assistant")
        vector_store_path = os.getenv("VECTOR_STORE_PATH", "/app/vector_store/wardrobe_faiss.index")
        
        logger.info(f"RAG Service Manager oluşturuluyor: config={config_name}")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Vector store path: {vector_store_path}")
        
        # Config seçimi
        try:
            config = get_rag_config(config_name)
            # Override paths from environment
            config.finetuned_model_path = model_path
            config.vector_store_path = vector_store_path
        except Exception as e:
            logger.warning(f"Config yükleme hatası: {e}, basic config kullanılacak")
            config = get_rag_config("basic")
            config.finetuned_model_path = model_path
            config.vector_store_path = vector_store_path
        
        # RAG Service Manager oluştur (henüz başlatma)
        rag_service_manager = create_rag_service_manager(config)
        
        log_service_event(logger, "rag_service_manager_created", status="success")
        log_service_event(logger, "lazy_loading_configured", message="RAG Service will be initialized on first request")
        
    except Exception as e:
        log_error_with_context(
            logger, 
            e, 
            {
                "operation": "rag_service_manager_creation",
                "config_name": config_name,
                "model_path": model_path,
                "vector_store_path": vector_store_path
            }
        )
        rag_service_manager = None
    
    yield  # Uygulama çalışırken bekle
    
    # Shutdown: Temizleme işlemleri
    log_service_event(logger, "application_shutdown_started")
    
    try:
        if rag_service_manager is not None:
            await rag_service_manager.shutdown()
            rag_service_manager = None
            
        log_service_event(logger, "application_shutdown_completed", status="success")
        
    except Exception as e:
        log_error_with_context(
            logger, 
            e, 
            {"operation": "application_shutdown"}
        )


# FastAPI uygulaması oluşturma
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# A4. Middleware konfigürasyonu
# CORS middleware - Tarayıcı isteklerine izin ver
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production'da specific domain'ler belirtilmeli
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Trusted host middleware - Güvenlik için
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Production'da specific host'lar belirtilmeli
)

# Custom logging middleware
app.add_middleware(StructuredLoggingMiddleware)


# A4. Global hata işleyicileri
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP hataları için global handler"""
    logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    
    # RAG Service unavailable (503) hatalarını fallback'e yönlendir
    if exc.status_code == 503 and "RAG Service" in str(exc.detail):
        # Chat endpoint'i için fallback
        if request.url.path == "/chat":
            try:
                # Request body'yi okumaya çalış
                body = await request.body()
                if body:
                    import json
                    data = json.loads(body)
                    query = data.get("query", "Merhaba!")
                    user_id = data.get("user_id", "anonymous")
                    
                    fallback_response = generate_fallback_response(query, user_id)
                    
                    return JSONResponse(
                        status_code=200,
                        content=ChatResponse(
                            success=True,
                            response=fallback_response,
                            user_id=user_id,
                            session_id=data.get("session_id"),
                            context_used=[],
                            confidence=0.5,
                            suggestions=["Ne tür kıyafetler tercih ediyorsun?", "Hangi renkleri seviyorsun?"],
                            metadata={
                                "mode": "fallback",
                                "model_used": "fallback_chatbot",
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            }
                        ).dict()
                    )
            except Exception as fallback_error:
                logger.error(f"Fallback error: {fallback_error}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        ).dict()
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Request validation hataları için handler"""
    logger.warning(f"Validation Error: {exc.errors()}")
    
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="Request validation failed",
            details=str(exc.errors()),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Genel hatalar için global handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            details="Bir sistem hatası oluştu. Lütfen tekrar deneyin.",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        ).dict()
    )


# Utility functions

def generate_fallback_response(query: str, user_id: str) -> str:
    """
    RAG service kullanılamadığında basit fallback response'lar üretir
    
    Args:
        query: Kullanıcı sorgusu
        user_id: Kullanıcı ID'si
        
    Returns:
        str: Fallback yanıtı
    """
    import random
    
    # Basit anahtar kelime bazlı yanıtlar
    query_lower = query.lower()
    
    # Moda ile ilgili sorular
    if any(word in query_lower for word in ['ne giysem', 'ne giyeyim', 'kıyafet', 'kombin']):
        responses = [
            f"Merhaba! Şu anda gardırop analiz sistemim yükleniyor. Genel olarak, mevsime uygun rahat kıyafetler önerebilirim. Hangi mevsimde ve ne tür bir etkinlik için kıyafet arıyorsun?",
            f"Selam! Tam sistemi güncelliyorum ama yine de yardım edebilirim. Hangi renkleri seviyorsun ve rahat mı şık mi bir stil tercih ediyorsun?",
            f"Merhaba {user_id}! Sistem birazdan hazır olacak. Bu sırada, özel bir etkinlik mi var yoksa günlük kombin mi arıyorsun?"
        ]
    # Renk soruları
    elif any(word in query_lower for word in ['renk', 'color', 'hangi renk']):
        responses = [
            "Renk seçimi kişisel zevke göre değişir! Cilt tonuna ve saç rengine uygun renkler genellikle daha iyi durur. Hangi renkleri seviyorsun?",
            "Güzel bir soru! Mevsim renkleri de önemli - şu anda hangi mevsimde yaşıyorsun?",
            "Renk kombinasyonları çok eğlenceli! Tek renk mi yoksa farklı renkleri karıştırma mı tercih ediyorsun?"
        ]
    # Genel selamlaşma
    elif any(word in query_lower for word in ['merhaba', 'selam', 'hello', 'hi']):
        responses = [
            f"Merhaba {user_id}! Ben Aura, senin kişisel moda asistanın! Şu anda sistemim güncellenirken nasıl yardım edebilirim?",
            f"Selam! Aura burada 👋 Moda konusunda sana yardım etmeye hazırım. Ne tür sorular var?",
            f"Merhaba! Kişisel moda asistanın Aura ile konuşuyorsun. Hangi konuda yardıma ihtiyacın var?"
        ]
    # Teşekkür
    elif any(word in query_lower for word in ['teşekkür', 'sağol', 'thanks', 'thank you']):
        responses = [
            "Rica ederim! Moda konusunda her zaman yardım etmeye hazırım 😊",
            "Ne demek! Başka sorularını bekliyorum ✨",
            "Sevindim yardım edebildiğime! Başka neyde yardım edebilirim?"
        ]
    # Varsayılan yanıtlar
    else:
        responses = [
            f"Merhaba {user_id}! Şu anda ana sistemim güncelleniyor ama gene de yardım etmeye çalışayım. Moda ve kıyafet konularında sorularını sorabilirsin!",
            "Selam! Aura buradayım. Moda asistanı özelliklerim şu anda güncellenirken, genel moda sorularını yanıtlamaya çalışabilirim.",
            "Merhaba! Kişisel gardırop analizim şu anda hazırlanıyor. Bu arada moda tercihlerin hakkında konuşabiliriz!"
        ]
    
    return random.choice(responses)


def generate_suggestions(query: str, response: str) -> List[str]:
    """
    Yanıt bazında ek öneriler üretir
    
    Args:
        query: Kullanıcı sorgusu
        response: Asistan yanıtı
        
    Returns:
        List[str]: Öneriler listesi
    """
    suggestions = []
    
    # Query bazlı öneriler
    if "renk" in query.lower():
        suggestions.extend([
            "Hangi renkler en çok yakışıyor?",
            "Bu sezonun trend renkleri neler?"
        ])
    
    if "kombin" in query.lower() or "ne giysem" in query.lower():
        suggestions.extend([
            "Aksesuarlar nasıl tamamlanır?",
            "Hangi ayakkabı uygun olur?"
        ])
    
    if "iş" in query.lower() or "toplantı" in query.lower():
        suggestions.extend([
            "Formal giyim kuralları neler?",
            "İş gardırobumu nasıl geliştiririm?"
        ])
    
    # Response bazlı öneriler
    if "ceket" in response.lower():
        suggestions.append("Ceket bakımı nasıl yapılır?")
    
    if "ayakkabı" in response.lower():
        suggestions.append("Ayakkabı kombinleri nasıl olmalı?")
    
    return suggestions[:3]  # Max 3 öneri


# A3. API Endpoint'lerinin Tanımlanması

@app.get("/", response_model=Dict[str, str])
async def root():
    """Ana sayfa - API bilgileri"""
    return {
        "message": "🤖 Aura Conversational AI API",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
        "chat_endpoint": "/chat",
        "batch_endpoint": "/chat/batch"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    A3a. Servis sağlık kontrolü endpoint'i
    
    Bu endpoint, servisin çalışıp çalışmadığını ve RAG service'in
    yüklenip yüklenmediğini kontrol eder.
    
    Returns:
        HealthResponse: Servis durumu bilgileri
    """
    logger.debug("Health check isteği alındı")
    
    global rag_service_manager
    
    service_loaded = rag_service_manager is not None and rag_service_manager.is_ready()
    status = "OK" if service_loaded else "RAG_SERVICE_NOT_READY"
    
    # RAG service stats
    rag_stats = {}
    if rag_service_manager and rag_service_manager.is_ready():
        try:
            rag_service = rag_service_manager.get_service()
            rag_stats = rag_service.get_service_stats()
        except Exception as e:
            rag_stats = {"error": str(e)}
    elif rag_service_manager:
        # Manager var ama service hazır değil
        rag_stats = rag_service_manager.get_status()
    
    return HealthResponse(
        status=status,
        service="Conversational AI",
        model_loaded=service_loaded,
        version=API_VERSION,
        rag_service_stats=rag_stats,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    background_tasks: BackgroundTasks = None
):
    """
    A3b. Ana sohbet endpoint'i
    
    Bu endpoint, kullanıcının mesajını alır, RAG pipeline'dan geçirir
    ve kişiselleştirilmiş fashion advice yanıtı döndürür.
    
    Args:
        request: Sohbet isteği
        background_tasks: Arka plan görevleri
        
    Returns:
        ChatResponse: Sohbet yanıtı
        
    Raises:
        HTTPException: Çeşitli hata durumlarında
    """
    start_time = time.time()
    
    request_logger = logger.bind(
        user_id=request.user_id,
        query_preview=request.query[:50] + "..." if len(request.query) > 50 else request.query,
        session_id=request.session_id
    )
    
    request_logger.info("chat_request_received")
    
    try:
        # 1. RAG service manager kontrolü
        global rag_service_manager
        
        if rag_service_manager is None:
            raise HTTPException(status_code=503, detail="RAG Service Manager başlatılamadı.")
        
        # 2. Servis hazır değilse başlatmaya çalış
        if not rag_service_manager.is_ready():
            try:
                logger.info("🔄 RAG Service başlatılıyor...")
                await rag_service_manager.initialize()
            except Exception as e:
                logger.error(f"❌ RAG Service başlatma hatası: {str(e)}")
                raise HTTPException(status_code=503, detail="RAG Service başlatılamadı.")
        
        # 3. RAG service'i al
        try:
            rag_service = rag_service_manager.get_service()
        except RuntimeError as e:
            log_error_with_context(
                request_logger,
                e,
                {"operation": "get_rag_service", "fallback": "true"}
            )
            # Fallback response
            fallback_response = generate_fallback_response(request.query, request.user_id)
            processing_time = time.time() - start_time
            
            log_performance_metric(request_logger, "chat_response_time", processing_time * 1000, "ms", mode="fallback")
            
            return ChatResponse(
                success=True,
                response=fallback_response,
                user_id=request.user_id,
                session_id=request.session_id,
                context_used=[],  # Fallback'de context yok
                confidence=0.5,   # Düşük güven skoru
                suggestions=["Ne tür kıyafetler tercih ediyorsun?", "Hangi renkleri seviyorsun?", "Özel bir etkinlik mi var?"],
                metadata={
                    "mode": "fallback",
                    "processing_time": processing_time,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "model_used": "fallback_chatbot"
                }
            )
        
        # 4. RAG pipeline ile yanıt üretme (Resource Manager ile)
        request_logger.info("rag_pipeline_started")
        
        # RAGServiceResource context manager ile bellek güvenli kullanım
        with RAGServiceResource(rag_service) as managed_service:
            rag_response = managed_service.generate_response(
                query=request.query,
                user_id=request.user_id
            )
        
        # 3. Yanıt işleme
        if not rag_response.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=f"RAG pipeline hatası: {rag_response.get('error', 'Unknown error')}"
            )
        
        # 4. Ek öneriler üret
        suggestions = generate_suggestions(request.query, rag_response["response"])
        
        # 5. Confidence score hesapla
        confidence = 0.9  # Başlangıç değeri
        if rag_response.get("context_used"):
            # Context varsa confidence artır
            confidence = min(0.95, confidence + len(rag_response["context_used"]) * 0.05)
        
        # 6. Response formatla
        processing_time = time.time() - start_time
        
        response = ChatResponse(
            success=True,
            response=rag_response["response"],
            user_id=request.user_id,
            session_id=request.session_id,
            context_used=rag_response.get("context_used", []),
            confidence=confidence,
            suggestions=suggestions,
            metadata={
                "processing_time": processing_time,
                "rag_metadata": rag_response.get("metadata", {}),
                "context": request.context,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        )
        
        log_performance_metric(request_logger, "chat_response_time", processing_time * 1000, "ms", 
                             confidence=confidence, mode="rag_pipeline")
        request_logger.info("chat_response_generated", confidence=confidence, 
                           context_items=len(rag_response.get("context_used", [])))
        return response
        
    except HTTPException:
        # HTTP hataları tekrar raise et
        raise
        
    except Exception as e:
        # Beklenmeyen hatalar
        processing_time = time.time() - start_time
        log_error_with_context(
            request_logger,
            e,
            {
                "operation": "chat_processing",
                "processing_time": processing_time,
                "query_length": len(request.query)
            }
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Sohbet işlemi sırasında hata oluştu: {str(e)}"
        )


@app.post("/chat/batch", response_model=BatchChatResponse)
async def batch_chat_endpoint(
    request: BatchChatRequest,
    background_tasks: BackgroundTasks = None
):
    """
    Toplu sohbet endpoint'i
    
    Bu endpoint, birden fazla mesajı aynı anda işler.
    
    Args:
        request: Toplu sohbet isteği
        background_tasks: Arka plan görevleri
        
    Returns:
        BatchChatResponse: Toplu sohbet yanıtı
    """
    start_time = time.time()
    
    logger.info(f"Batch chat request: {len(request.messages)} messages")
    
    try:
        # RAG service manager kontrolü ve fallback
        global rag_service_manager
        
        if rag_service_manager is None:
            raise HTTPException(status_code=503, detail="RAG Service Manager başlatılamadı.")
        
        # Servis hazır değilse başlatmaya çalış
        if not rag_service_manager.is_ready():
            try:
                logger.info("🔄 RAG Service batch işlem için başlatılıyor...")
                await rag_service_manager.initialize()
            except Exception as e:
                logger.error(f"❌ RAG Service başlatma hatası: {str(e)}")
                # Batch fallback responses
                responses = []
                for message in request.messages:
                    fallback_resp = generate_fallback_response(message.query, message.user_id)
                    responses.append(ChatResponse(
                        success=True,
                        response=fallback_resp,
                        user_id=message.user_id,
                        session_id=message.session_id,
                        context_used=[],
                        confidence=0.5,
                    suggestions=["Ne tür kıyafetler tercih ediyorsun?"],
                    metadata={"mode": "fallback", "model_used": "fallback_chatbot"}
                ))
            
            return BatchChatResponse(
                success=True,
                responses=responses,
                total_processed=len(request.messages),
                failed_count=0,
                processing_time=time.time() - start_time,
                metadata={"mode": "batch_fallback"}
            )
        
        # Batch işleme
        responses = []
        failed_count = 0
        
        for i, message in enumerate(request.messages):
            logger.info(f"Processing batch message {i+1}/{len(request.messages)}")
            
            try:
                # Her mesaj için chat endpoint'ini çağır
                chat_response = await chat_endpoint(message)
                responses.append(chat_response)
                
            except Exception as e:
                logger.warning(f"Batch message {i+1} failed: {str(e)}")
                failed_count += 1
                
                # Hata response'u oluştur
                error_response = ChatResponse(
                    success=False,
                    response=f"Bu mesaj işlenirken hata oluştu: {str(e)}",
                    user_id=message.user_id,
                    session_id=message.session_id,
                    context_used=[],
                    confidence=0.0,
                    suggestions=[],
                    metadata={
                        "error": str(e),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                )
                responses.append(error_response)
        
        # Batch response oluştur
        processing_time = time.time() - start_time
        
        batch_response = BatchChatResponse(
            success=failed_count == 0,
            responses=responses,
            processed_count=len(responses),
            failed_count=failed_count,
            processing_time=processing_time
        )
        
        logger.info(f"Batch processing completed: {len(responses)} processed, {failed_count} failed")
        return batch_response
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Batch chat error: {str(e)} | Time: {processing_time:.4f}s")
        
        raise HTTPException(
            status_code=500,
            detail=f"Toplu sohbet işlemi sırasında hata oluştu: {str(e)}"
        )


@app.get("/chat/stats", response_model=Dict[str, Any])
async def get_chat_stats():
    """RAG service ve chat istatistiklerini döndürür"""
    global rag_service_manager
    
    if rag_service_manager is None or not rag_service_manager.is_ready():
        # Fallback stats
        status = "not_initialized" if rag_service_manager is None else "initializing"
        return {
            "api_version": API_VERSION,
            "service_status": status,
            "rag_service_loaded": False,
            "mode": "fallback_chatbot",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "endpoints": ["/chat", "/chat/batch", "/health", "/stats"],
            "message": f"RAG service {status}, fallback modunda çalışıyor"
        }
    
    try:
        rag_service = rag_service_manager.get_service()
        rag_stats = rag_service.get_service_stats()
        
        api_stats = {
            "api_version": API_VERSION,
            "service_status": "active",
            "endpoints": [
                {"path": "/chat", "method": "POST", "description": "Single chat message"},
                {"path": "/chat/batch", "method": "POST", "description": "Batch chat messages"},
                {"path": "/chat/stats", "method": "GET", "description": "Service statistics"},
                {"path": "/health", "method": "GET", "description": "Health check"}
            ],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return {
            "rag_service": rag_stats,
            "api_service": api_stats
        }
        
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket support for real-time chat
@app.websocket("/ws/chat/{user_id}")
async def websocket_chat(websocket: WebSocket, user_id: str):
    """
    WebSocket endpoint'i - Real-time chat
    
    Args:
        websocket: WebSocket connection
        user_id: Kullanıcı kimliği
    """
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
            
            # Create chat request
            chat_request = ChatRequest(
                query=message,
                user_id=user_id,
                session_id=data.get("session_id"),
                context=data.get("context", {})
            )
            
            try:
                # Process message
                response = await chat_endpoint(chat_request)
                
                # Send response back to client
                await websocket.send_json({
                    "success": True,
                    "response": response.response,
                    "confidence": response.confidence,
                    "suggestions": response.suggestions,
                    "context_used": len(response.context_used),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                
            except Exception as e:
                logger.error(f"WebSocket processing error: {str(e)}")
                await websocket.send_json({
                    "success": False,
                    "error": f"Mesaj işlenirken hata oluştu: {str(e)}"
                })
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket connection closed for user {user_id}")
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {str(e)}")
        try:
            await websocket.send_json({
                "error": "WebSocket connection error"
            })
        except:
            pass


# A5. Uvicorn Sunucu Başlatıcısı
if __name__ == "__main__":
    """
    A5. Ana çalıştırma bloğu
    
    Bu blok, dosya doğrudan çalıştırıldığında (python main.py) 
    uvicorn sunucusunu başlatır.
    """
    # Environment variables
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8003))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    WORKERS = int(os.getenv("WORKERS", 1))
    
    logger.info(f"🚀 Aura Conversational AI API başlatılıyor...")
    logger.info(f"📍 Host: {HOST}:{PORT}")
    logger.info(f"🐛 Debug mode: {DEBUG}")
    logger.info(f"👥 Workers: {WORKERS}")
    
    # Uvicorn config
    uvicorn_config = {
        "host": HOST,
        "port": PORT,
        "reload": DEBUG,  # Development'ta auto-reload
        "workers": WORKERS if not DEBUG else 1,  # Debug'ta tek worker
        "log_level": "info",
        "access_log": True,
        "use_colors": True,
    }
    
    try:
        # Sunucuyu başlat
        uvicorn.run("main:app", **uvicorn_config)
        
    except KeyboardInterrupt:
        logger.info("🛑 Sunucu durduruldu (Ctrl+C)")
        
    except Exception as e:
        logger.error(f"❌ Sunucu başlatma hatası: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
