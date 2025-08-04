"""
FastAPI Main Application - DETR Visual Analysis Microservice for Aura Project
Bu modül, Aura projesinin görsel analiz mikroservisini FastAPI web sunucusu olarak sunar.
DETR tabanlı fashion analysis modelini HTTP API endpoints aracılığıyla erişilebilir kılar.
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
import tempfile
from contextlib import asynccontextmanager

# FastAPI ve ilgili kütüphaneler
from fastapi import (
    FastAPI, 
    File, 
    UploadFile, 
    HTTPException, 
    Depends, 
    BackgroundTasks,
    Request,
    Response
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

# Pydantic models for request/response validation
from pydantic import BaseModel, Field, validator
from pydantic.json import pydantic_encoder

# HTTP ve async operations
import uvicorn
import aiofiles

# PIL for image processing
from PIL import Image
import io

# Numerik kütüphaneler
import numpy as np

# Parent directory'yi path'e ekle (inference modüllerini import edebilmek için)
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Local imports - Aura project inference modules
try:
    from inference import (
        load_inference_model,
        run_inference,
        DEFAULT_CONFIDENCE_THRESHOLD,
        DEFAULT_MAX_DETECTIONS,
        FASHION_ATTRIBUTES
    )
    from data_loader import FASHIONPEDIA_CATEGORIES
    from model import TOTAL_CLASSES, FASHIONPEDIA_LABELS
except ImportError as e:
    # Fallback import strategy
    sys.path.insert(0, str(parent_dir.parent))
    try:
        from src.inference import (
            load_inference_model,
            run_inference,
            DEFAULT_CONFIDENCE_THRESHOLD,
            DEFAULT_MAX_DETECTIONS,
            FASHION_ATTRIBUTES
        )
        from src.data_loader import FASHIONPEDIA_CATEGORIES
        from src.model import TOTAL_CLASSES, FASHIONPEDIA_LABELS
    except ImportError:
        # Son çare - relatif import
        from ..inference import (
            load_inference_model,
            run_inference,
            DEFAULT_CONFIDENCE_THRESHOLD,
            DEFAULT_MAX_DETECTIONS,
            FASHION_ATTRIBUTES
        )
        from ..data_loader import FASHIONPEDIA_CATEGORIES
        from ..model import TOTAL_CLASSES, FASHIONPEDIA_LABELS

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('visual_analysis_api.log')
    ]
)
logger = logging.getLogger(__name__)

# Global variables for model and processor
global_model = None
global_processor = None
global_model_info = {}

# API Configuration
API_TITLE = "Aura Visual Analysis API"
API_DESCRIPTION = """
🔍 **Aura Yapay Zeka Platformu - Görsel Analiz Mikroservisi**

Bu API, DETR (Detection Transformer) tabanlı derin öğrenme modeli kullanarak 
fashion görüntülerinin analizini gerçekleştirir.

## Özellikler:
- **Fashion Item Detection**: Giyim eşyaları ve aksesuarları algılama
- **Attribute Analysis**: Renk, desen, stil ve malzeme analizi  
- **Segmentation**: Piksel seviyesinde segmentasyon maskeleri
- **Confidence Scoring**: Her detection için güven skorları
- **Batch Processing**: Çoklu görüntü analizi desteği

## Desteklenen Fashion Kategorileri:
- Üst giyim (gömlek, tişört, kazak, ceket vb.)
- Alt giyim (pantolon, etek, şort vb.)
- Ayakkabı ve aksesuarlar
- Desenler ve detaylar (düğme, fermuar, cep vb.)

Toplam **294 Fashionpedia kategorisi** + 1 background class = **295 sınıf**
"""
API_VERSION = "1.0.0"

# Supported image formats
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit

# API Models (Request/Response schemas)
class AnalysisRequest(BaseModel):
    """Görsel analiz isteği için Pydantic modeli"""
    confidence_threshold: float = Field(
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        ge=0.1, 
        le=1.0, 
        description="Minimum güven skoru threshold'u (0.1-1.0)"
    )
    max_detections: int = Field(
        default=DEFAULT_MAX_DETECTIONS,
        ge=1, 
        le=200, 
        description="Maksimum detection sayısı (1-200)"
    )
    return_masks: bool = Field(
        default=True,
        description="Segmentation maskelerini döndür"
    )
    include_attributes: bool = Field(
        default=True,
        description="Fashion özniteliklerini (renk, desen vb.) dahil et"
    )


class DetectionResult(BaseModel):
    """Tek bir detection sonucu"""
    label: str = Field(description="Fashion kategori etiketi")
    confidence: float = Field(description="Güven skoru (0-1)")
    bbox: List[float] = Field(description="Bounding box [x, y, width, height]")
    area: float = Field(description="Detection alanı (piksel)")
    category_id: int = Field(description="Fashionpedia kategori ID'si")
    attributes: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Fashion özellikleri (renk, desen, stil, malzeme)"
    )
    mask: Optional[List[List[int]]] = Field(
        default=None,
        description="Segmentation maskesi (opsiyonel)"
    )


class AnalysisResponse(BaseModel):
    """Görsel analiz yanıtı"""
    success: bool = Field(description="İşlem başarı durumu")
    detections: List[DetectionResult] = Field(description="Tespit edilen fashion itemlar")
    summary: Dict[str, Any] = Field(description="Analiz özeti")
    metadata: Dict[str, Any] = Field(description="İşlem metadata'sı")
    processing_time: float = Field(description="İşlem süresi (saniye)")
    model_info: Dict[str, str] = Field(description="Model bilgileri")


class HealthResponse(BaseModel):
    """Sağlık kontrolü yanıtı"""
    status: str = Field(description="Servis durumu")
    model_loaded: bool = Field(description="Model yüklenme durumu")
    version: str = Field(description="API versiyonu")
    timestamp: str = Field(description="Zaman damgası")
    supported_formats: List[str] = Field(description="Desteklenen görüntü formatları")


class ErrorResponse(BaseModel):
    """Hata yanıtı"""
    error: str = Field(description="Hata mesajı")
    details: Optional[str] = Field(default=None, description="Hata detayları")
    timestamp: str = Field(description="Hata zamanı")


# Custom middleware for logging and monitoring
class LoggingMiddleware(BaseHTTPMiddleware):
    """HTTP isteklerini loglayan middleware"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Request logging
        logger.info(f"Request: {request.method} {request.url}")
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Response logging
            logger.info(
                f"Response: {response.status_code} | "
                f"Time: {process_time:.4f}s | "
                f"Size: {response.headers.get('content-length', 'unknown')}"
            )
            
            response.headers["X-Process-Time"] = str(process_time)
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(f"Request failed: {str(e)} | Time: {process_time:.4f}s")
            raise


# Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    A2. Uygulama yaşam döngüsü yönetimi
    Başlangıçta model yükleme, kapanışta temizleme işlemleri
    """
    global global_model, global_processor, global_model_info
    
    logger.info("🚀 Aura Visual Analysis API başlatılıyor...")
    
    # Startup: Model yükleme
    try:
        model_path = os.getenv(
            "MODEL_PATH", 
            "./saved_models/detr_fashionpedia.pth"
        )
        
        logger.info(f"Model yükleniyor: {model_path}")
        
        # Eğer model dosyası yoksa, pre-trained model kullan
        if not os.path.exists(model_path):
            logger.warning(f"Model dosyası bulunamadı: {model_path}")
            logger.info("Pre-trained DETR modeli kullanılacak")
            
            # load_inference_model fonksiyonunu pre-trained model için çağır
            global_model, global_processor = load_inference_model(
                model_path="facebook/detr-resnet-50-panoptic"
            )
        else:
            global_model, global_processor = load_inference_model(model_path)
        
        # Model bilgilerini kaydet
        global_model_info = {
            "model_path": model_path,
            "total_classes": TOTAL_CLASSES,
            "categories_count": len(FASHIONPEDIA_CATEGORIES),
            "model_type": "DETR (Detection Transformer)",
            "framework": "PyTorch + Hugging Face Transformers"
        }
        
        logger.info("✅ Model başarıyla yüklendi!")
        logger.info(f"📊 Model İstatistikleri: {global_model_info}")
        
    except Exception as e:
        logger.error(f"❌ Model yükleme hatası: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Uygulamayı durdurmayalım, hata endpoint'lerinde ele alalım
        global_model = None
        global_processor = None
    
    yield  # Uygulama çalışırken bekle
    
    # Shutdown: Temizleme işlemleri
    logger.info("🛑 Aura Visual Analysis API kapatılıyor...")
    
    try:
        # Model memory'den temizle
        if global_model is not None:
            del global_model
            global_model = None
            
        if global_processor is not None:
            del global_processor
            global_processor = None
            
        # GPU memory temizle (eğer CUDA kullanılıyorsa)
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("✅ Temizleme işlemleri tamamlandı")
        
    except Exception as e:
        logger.error(f"❌ Temizleme hatası: {str(e)}")


# A2. FastAPI uygulaması oluşturma
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
app.add_middleware(LoggingMiddleware)


# A4. Global hata işleyicileri
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP hataları için global handler"""
    logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    
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
def validate_image_file(file: UploadFile) -> None:
    """
    Yüklenen dosyanın geçerli bir görüntü dosyası olup olmadığını kontrol eder
    
    Args:
        file: Yüklenen dosya
        
    Raises:
        HTTPException: Geçersiz dosya durumunda
    """
    # Dosya boyutu kontrolü
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Dosya boyutu çok büyük. Maksimum: {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    # Dosya uzantısı kontrolü
    if file.filename:
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in SUPPORTED_IMAGE_FORMATS:
            raise HTTPException(
                status_code=415,
                detail=f"Desteklenmeyen dosya formatı. Desteklenen: {list(SUPPORTED_IMAGE_FORMATS)}"
            )


def check_model_availability() -> None:
    """
    Modelin yüklü olup olmadığını kontrol eder
    
    Raises:
        HTTPException: Model yüklü değilse
    """
    if global_model is None or global_processor is None:
        raise HTTPException(
            status_code=503,
            detail="Model henüz yüklenmedi veya yükleme başarısız. Lütfen daha sonra tekrar deneyin."
        )


async def save_uploaded_file(file: UploadFile) -> str:
    """
    Yüklenen dosyayı geçici olarak kaydeder
    
    Args:
        file: Yüklenen dosya
        
    Returns:
        str: Kaydedilen dosyanın yolu
    """
    # Geçici dosya oluştur
    suffix = Path(file.filename or "image").suffix
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    return tmp_file_path


# A3. API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Ana sayfa - API bilgileri"""
    return {
        "message": "🔍 Aura Visual Analysis API",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
        "analyze_endpoint": "/analyze"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    A3a. Servis sağlık kontrolü endpoint'i
    
    Bu endpoint, servisin çalışıp çalışmadığını ve modelin 
    yüklenip yüklenmediğini kontrol eder.
    
    Returns:
        HealthResponse: Servis durumu bilgileri
    """
    logger.debug("Health check isteği alındı")
    
    model_loaded = global_model is not None and global_processor is not None
    status = "OK" if model_loaded else "MODEL_NOT_LOADED"
    
    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        version=API_VERSION,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        supported_formats=list(SUPPORTED_IMAGE_FORMATS)
    )


@app.get("/model/info", response_model=Dict[str, Any])
async def get_model_info():
    """Model hakkında detaylı bilgi döndürür"""
    check_model_availability()
    
    return {
        "model_info": global_model_info,
        "categories": FASHIONPEDIA_CATEGORIES,
        "total_categories": len(FASHIONPEDIA_CATEGORIES),
        "fashion_attributes": FASHION_ATTRIBUTES,
        "default_settings": {
            "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
            "max_detections": DEFAULT_MAX_DETECTIONS
        }
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(..., description="Analiz edilecek görüntü dosyası"),
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    max_detections: int = DEFAULT_MAX_DETECTIONS,
    return_masks: bool = True,
    include_attributes: bool = True,
    background_tasks: BackgroundTasks = None
):
    """
    A3b. Ana görüntü analizi endpoint'i
    
    Bu endpoint, kullanıcının yüklediği görüntüyü analiz eder ve
    fashion itemları, öznitelikleri ve segmentation maskelerini döndürür.
    
    Args:
        file: Yüklenen görüntü dosyası
        confidence_threshold: Minimum güven skoru
        max_detections: Maksimum detection sayısı
        return_masks: Segmentation maskelerini döndür
        include_attributes: Fashion özniteliklerini dahil et
        background_tasks: Arka plan görevleri
        
    Returns:
        AnalysisResponse: Analiz sonuçları
        
    Raises:
        HTTPException: Çeşitli hata durumlarında
    """
    start_time = time.time()
    temp_file_path = None
    
    logger.info(f"Görüntü analizi başlatılıyor: {file.filename}")
    
    try:
        # 1. Model kontrolü
        check_model_availability()
        
        # 2. Dosya doğrulama
        validate_image_file(file)
        
        # 3. Parametre doğrulama
        if not (0.1 <= confidence_threshold <= 1.0):
            raise HTTPException(
                status_code=422,
                detail="confidence_threshold 0.1-1.0 arasında olmalıdır"
            )
        
        if not (1 <= max_detections <= 200):
            raise HTTPException(
                status_code=422,
                detail="max_detections 1-200 arasında olmalıdır"
            )
        
        # 4. Dosyayı geçici olarak kaydet
        temp_file_path = await save_uploaded_file(file)
        
        # 5. Görüntüyü yükle ve doğrula
        try:
            pil_image = Image.open(temp_file_path).convert("RGB")
            original_size = pil_image.size
            logger.info(f"Görüntü yüklendi: {original_size[0]}x{original_size[1]}")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Görüntü dosyası bozuk veya okunamıyor: {str(e)}"
            )
        
        # 6. DETR inference çalıştır
        logger.info("DETR inference başlatılıyor...")
        
        inference_results = run_inference(
            model=global_model,
            image_processor=global_processor,
            image=pil_image,
            confidence_threshold=confidence_threshold,
            max_detections=max_detections,
            return_masks=return_masks,
            return_raw_output=False
        )
        
        logger.info(f"Inference tamamlandı: {len(inference_results.get('detections', []))} detection")
        
        # 7. Sonuçları API formatına dönüştür
        detections = []
        
        for det in inference_results.get('detections', []):
            detection_result = DetectionResult(
                label=det.get('label', 'unknown'),
                confidence=det.get('confidence', 0.0),
                bbox=det.get('bbox', [0, 0, 0, 0]),
                area=det.get('area', 0.0),
                category_id=det.get('category_id', 0),
                attributes=det.get('attributes') if include_attributes else None,
                mask=det.get('mask') if return_masks else None
            )
            detections.append(detection_result)
        
        # 8. Özet bilgilerini hazırla
        summary = {
            "total_detections": len(detections),
            "unique_categories": len(set(d.label for d in detections)),
            "average_confidence": sum(d.confidence for d in detections) / len(detections) if detections else 0.0,
            "image_dimensions": original_size,
            "categories_found": list(set(d.label for d in detections))
        }
        
        # Attribute summary (eğer dahil edilmişse)
        if include_attributes and detections:
            summary["attributes_summary"] = {}
            for attr_type in ["colors", "patterns", "styles", "materials"]:
                found_attrs = set()
                for det in detections:
                    if det.attributes and attr_type in det.attributes:
                        if isinstance(det.attributes[attr_type], list):
                            found_attrs.update(det.attributes[attr_type])
                        else:
                            found_attrs.add(det.attributes[attr_type])
                summary["attributes_summary"][attr_type] = list(found_attrs)
        
        # 9. Metadata hazırla
        processing_time = time.time() - start_time
        
        metadata = {
            "processing_time": processing_time,
            "image_filename": file.filename,
            "image_size": file.size,
            "parameters_used": {
                "confidence_threshold": confidence_threshold,
                "max_detections": max_detections,
                "return_masks": return_masks,
                "include_attributes": include_attributes
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        logger.info(f"Analiz tamamlandı: {processing_time:.4f}s")
        
        # 10. Response oluştur
        response = AnalysisResponse(
            success=True,
            detections=detections,
            summary=summary,
            metadata=metadata,
            processing_time=processing_time,
            model_info=global_model_info
        )
        
        # 11. Arka plan görevi: geçici dosyayı sil
        if background_tasks and temp_file_path:
            background_tasks.add_task(cleanup_temp_file, temp_file_path)
        
        return response
        
    except HTTPException:
        # HTTP hataları tekrar raise et
        raise
        
    except Exception as e:
        # Beklenmeyen hatalar
        processing_time = time.time() - start_time
        logger.error(f"Analiz hatası: {str(e)} | Time: {processing_time:.4f}s")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Görüntü analizi sırasında hata oluştu: {str(e)}"
        )
    
    finally:
        # Geçici dosyayı temizle (eğer background task kullanılmıyorsa)
        if temp_file_path and not background_tasks:
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Geçici dosya silinemedi: {e}")


@app.post("/analyze/batch", response_model=List[AnalysisResponse])
async def analyze_images_batch(
    files: List[UploadFile] = File(..., description="Analiz edilecek görüntü dosyaları listesi"),
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    max_detections: int = DEFAULT_MAX_DETECTIONS,
    return_masks: bool = False,  # Batch'te mask'ler varsayılan olarak kapalı
    include_attributes: bool = True,
    background_tasks: BackgroundTasks = None
):
    """
    Çoklu görüntü analizi endpoint'i
    
    Bu endpoint, birden fazla görüntüyü aynı anda analiz eder.
    Büyük batch'ler için performans optimizasyonu içerir.
    
    Args:
        files: Analiz edilecek görüntü dosyaları listesi
        confidence_threshold: Minimum güven skoru
        max_detections: Maksimum detection sayısı  
        return_masks: Segmentation maskelerini döndür
        include_attributes: Fashion özniteliklerini dahil et
        background_tasks: Arka plan görevleri
        
    Returns:
        List[AnalysisResponse]: Her görüntü için analiz sonuçları
    """
    logger.info(f"Batch analiz başlatılıyor: {len(files)} dosya")
    
    # Batch size kontrolü
    if len(files) > 20:  # Maksimum batch size
        raise HTTPException(
            status_code=413,
            detail="Tek seferde en fazla 20 görüntü analiz edilebilir"
        )
    
    check_model_availability()
    
    results = []
    
    for i, file in enumerate(files):
        logger.info(f"Batch analiz: {i+1}/{len(files)} - {file.filename}")
        
        try:
            # Her dosya için analyze_image fonksiyonunu çağır
            result = await analyze_image(
                file=file,
                confidence_threshold=confidence_threshold,
                max_detections=max_detections,
                return_masks=return_masks,
                include_attributes=include_attributes,
                background_tasks=background_tasks
            )
            results.append(result)
            
        except Exception as e:
            # Hatalı dosyalar için hata response'u oluştur
            logger.warning(f"Batch'te dosya hatası: {file.filename} - {str(e)}")
            
            error_result = AnalysisResponse(
                success=False,
                detections=[],
                summary={"error": str(e), "filename": file.filename},
                metadata={"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
                processing_time=0.0,
                model_info=global_model_info
            )
            results.append(error_result)
    
    logger.info(f"Batch analiz tamamlandı: {len(results)} sonuç")
    return results


# Utility endpoints
@app.get("/categories", response_model=Dict[str, Any])
async def get_categories():
    """Desteklenen fashion kategorilerini döndürür"""
    return {
        "categories": FASHIONPEDIA_CATEGORIES,
        "total_count": len(FASHIONPEDIA_CATEGORIES),
        "attributes": FASHION_ATTRIBUTES
    }


@app.get("/stats", response_model=Dict[str, Any])
async def get_api_stats():
    """API istatistiklerini döndürür"""
    # Bu örnekte basit istatistikler, production'da gerçek metrics kullanılabilir
    return {
        "api_version": API_VERSION,
        "model_loaded": global_model is not None,
        "supported_formats": list(SUPPORTED_IMAGE_FORMATS),
        "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
        "max_batch_size": 20,
        "endpoints": [
            {"path": "/", "method": "GET", "description": "Root endpoint"},
            {"path": "/health", "method": "GET", "description": "Health check"},
            {"path": "/analyze", "method": "POST", "description": "Single image analysis"},
            {"path": "/analyze/batch", "method": "POST", "description": "Batch image analysis"},
            {"path": "/categories", "method": "GET", "description": "Fashion categories"},
            {"path": "/model/info", "method": "GET", "description": "Model information"}
        ]
    }


# Background task functions
def cleanup_temp_file(file_path: str):
    """Geçici dosyayı temizleyen arka plan görevi"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.debug(f"Geçici dosya silindi: {file_path}")
    except Exception as e:
        logger.warning(f"Geçici dosya silinemedi {file_path}: {e}")


# A5. Uvicorn sunucu başlatıcısı
if __name__ == "__main__":
    """
    A5. Ana çalıştırma bloğu
    
    Bu blok, dosya doğrudan çalıştırıldığında (python main.py) 
    uvicorn sunucusunu başlatır.
    """
    # Environment variables
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    WORKERS = int(os.getenv("WORKERS", 1))
    
    logger.info(f"🚀 Aura Visual Analysis API başlatılıyor...")
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
