"""
Structured Logging Configuration for Aura AI Platform
Bu modül, yapılandırılmış loglama (structured logging) sistemi kurar ve
HTTP request monitoring middleware'i sağlar.
"""

import logging
import structlog
import sys
import os
import time
from typing import Any, Dict, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


def setup_logging() -> None:
    """
    Yapılandırılmış loglama sistemini kurar
    Structlog ve stdlib logging'i entegre eder
    """
    # Get log level from environment
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level, logging.INFO)
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """
    FastAPI için yapılandırılmış loglama middleware'i
    Her HTTP isteğini detaylı şekilde loglar
    """
    
    def __init__(self, app, logger_name: str = "api"):
        super().__init__(app)
        self.logger = structlog.get_logger(logger_name)
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        HTTP isteklerini yakalar ve yapılandırılmış şekilde loglar
        
        Args:
            request: FastAPI Request nesnesi
            call_next: Sonraki middleware/endpoint çağrısı
            
        Returns:
            Response: HTTP yanıtı
        """
        start_time = time.time()
        
        # Request bilgilerini bind et
        request_logger = self.logger.bind(
            method=request.method,
            url=str(request.url),
            path=request.url.path,
            client_host=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent", "unknown"),
            content_type=request.headers.get("content-type", "unknown"),
            content_length=request.headers.get("content-length", 0),
            request_id=f"req_{int(start_time * 1000)}"
        )
        
        # Request başlangıcını logla
        request_logger.info("request_started")
        
        try:
            # Request'i işle
            response = await call_next(request)
            
            # Process time hesapla
            process_time = time.time() - start_time
            
            # Response bilgilerini bind et
            response_logger = request_logger.bind(
                status_code=response.status_code,
                process_time_ms=round(process_time * 1000, 2),
                response_size=response.headers.get("content-length", 0)
            )
            
            # Response'u logla
            if response.status_code >= 400:
                response_logger.warning("request_completed_with_error")
            else:
                response_logger.info("request_completed")
            
            return response
            
        except Exception as e:
            # Hata durumunu logla
            process_time = time.time() - start_time
            
            error_logger = request_logger.bind(
                error_type=type(e).__name__,
                error_message=str(e),
                process_time_ms=round(process_time * 1000, 2)
            )
            
            error_logger.error("request_failed")
            
            # Hatayı yeniden fırlat
            raise


async def logging_middleware(request: Request, call_next) -> Response:
    """
    Standalone logging middleware function
    StructuredLoggingMiddleware'in fonksiyonel versiyonu
    
    Args:
        request: FastAPI Request nesnesi
        call_next: Sonraki middleware/endpoint çağrısı
        
    Returns:
        Response: HTTP yanıtı
    """
    start_time = time.time()
    logger = structlog.get_logger("api")
    
    # Request bilgilerini bind et
    request_logger = logger.bind(
        method=request.method,
        url=str(request.url),
        path=request.url.path,
        client_host=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent", "unknown"),
        request_id=f"req_{int(start_time * 1000)}"
    )
    
    request_logger.info("request_started")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        response_logger = request_logger.bind(
            status_code=response.status_code,
            process_time_ms=round(process_time * 1000, 2)
        )
        
        response_logger.info("request_processed")
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        
        error_logger = request_logger.bind(
            error_type=type(e).__name__,
            error_message=str(e),
            process_time_ms=round(process_time * 1000, 2)
        )
        
        error_logger.error("request_failed")
        raise


def get_structured_logger(name: str, **context) -> structlog.BoundLogger:
    """
    Yapılandırılmış logger oluşturur ve opsiyonel context bind eder
    
    Args:
        name: Logger adı
        **context: Bind edilecek context key-value çiftleri
        
    Returns:
        structlog.BoundLogger: Yapılandırılmış logger
    """
    logger = structlog.get_logger(name)
    if context:
        logger = logger.bind(**context)
    return logger


def log_service_event(logger: structlog.BoundLogger, event: str, **kwargs) -> None:
    """
    Servis olaylarını standardize edilmiş formatta loglar
    
    Args:
        logger: Structlog logger instance'ı
        event: Olay adı (örn: "service_started", "model_loaded")
        **kwargs: Ek context bilgileri
    """
    logger.info(f"service_event: {event}", **kwargs)


def log_performance_metric(logger: structlog.BoundLogger, metric_name: str, value: float, unit: str = "ms", **kwargs) -> None:
    """
    Performance metriklerini loglar
    
    Args:
        logger: Structlog logger instance'ı
        metric_name: Metrik adı
        value: Metrik değeri
        unit: Birim (ms, seconds, bytes, vb.)
        **kwargs: Ek context bilgileri
    """
    logger.info(f"performance_metric: {metric_name}", 
                value=value, 
                unit=unit, 
                **kwargs)


def log_user_interaction(logger: structlog.BoundLogger, user_id: str, action: str, **kwargs) -> None:
    """
    Kullanıcı etkileşimlerini loglar
    
    Args:
        logger: Structlog logger instance'ı
        user_id: Kullanıcı kimliği
        action: Gerçekleştirilen aksiyon
        **kwargs: Ek context bilgileri
    """
    logger.info(f"user_interaction: {action}", 
                user_id=user_id, 
                **kwargs)


def log_ml_inference(logger: structlog.BoundLogger, model_name: str, input_size: int, output_size: int, latency_ms: float, **kwargs) -> None:
    """
    ML model inference olaylarını loglar
    
    Args:
        logger: Structlog logger instance'ı
        model_name: Model adı
        input_size: Input boyutu
        output_size: Output boyutu
        latency_ms: İnference süresi (milisaniye)
        **kwargs: Ek context bilgileri
    """
    logger.info(f"ml_inference: {model_name}", 
                input_size=input_size,
                output_size=output_size,
                latency_ms=latency_ms,
                **kwargs)


def log_error_with_context(logger: structlog.BoundLogger, error: Exception, context: Dict[str, Any]) -> None:
    """
    Hataları zengin context ile loglar
    
    Args:
        logger: Structlog logger instance'ı
        error: Exception nesnesi
        context: Hata context bilgileri
    """
    logger.error(f"error_occurred: {type(error).__name__}",
                 error_message=str(error),
                 **context)


# Commonly used loggers
api_logger = structlog.get_logger("api")
rag_logger = structlog.get_logger("rag_service")
auth_logger = structlog.get_logger("auth")
db_logger = structlog.get_logger("database")
