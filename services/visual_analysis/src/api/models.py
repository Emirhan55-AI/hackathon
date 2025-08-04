"""
Pydantic Models for Aura Visual Analysis API
Bu modül, API request/response şemalarını tanımlayan Pydantic modellerini içerir.
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class ImageFormat(str, Enum):
    """Desteklenen görüntü formatları enum'u"""
    JPG = "jpg"
    JPEG = "jpeg"
    PNG = "png"
    BMP = "bmp"
    TIFF = "tiff"
    WEBP = "webp"


class ConfidenceLevel(str, Enum):
    """Önceden tanımlanmış güven seviyesi presetleri"""
    LOW = "low"      # 0.3
    MEDIUM = "medium" # 0.5  
    HIGH = "high"    # 0.7
    VERY_HIGH = "very_high" # 0.9


class AnalysisRequest(BaseModel):
    """Görsel analiz isteği için Pydantic modeli"""
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.1, 
        le=1.0, 
        description="Minimum güven skoru threshold'u (0.1-1.0)"
    )
    max_detections: int = Field(
        default=50,
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
    confidence_preset: Optional[ConfidenceLevel] = Field(
        default=None,
        description="Önceden tanımlanmış güven seviyesi (threshold'u override eder)"
    )
    
    @validator('confidence_threshold')
    def validate_confidence(cls, v, values):
        """Confidence threshold doğrulama"""
        if 'confidence_preset' in values and values['confidence_preset']:
            # Preset kullanılıyorsa threshold'u ignore et
            preset_values = {
                ConfidenceLevel.LOW: 0.3,
                ConfidenceLevel.MEDIUM: 0.5,
                ConfidenceLevel.HIGH: 0.7,
                ConfidenceLevel.VERY_HIGH: 0.9
            }
            return preset_values[values['confidence_preset']]
        return v


class BoundingBox(BaseModel):
    """Bounding box koordinatları"""
    x: float = Field(description="Sol üst köşe X koordinatı")
    y: float = Field(description="Sol üst köşe Y koordinatı") 
    width: float = Field(description="Genişlik")
    height: float = Field(description="Yükseklik")
    
    @validator('x', 'y', 'width', 'height')
    def validate_positive(cls, v):
        if v < 0:
            raise ValueError("Koordinat değerleri negatif olamaz")
        return v


class FashionAttributes(BaseModel):
    """Fashion öznitelikleri modeli"""
    colors: Optional[List[str]] = Field(default=None, description="Tespit edilen renkler")
    patterns: Optional[List[str]] = Field(default=None, description="Tespit edilen desenler")
    styles: Optional[List[str]] = Field(default=None, description="Tespit edilen stiller")
    materials: Optional[List[str]] = Field(default=None, description="Tespit edilen malzemeler")
    dominant_color: Optional[str] = Field(default=None, description="Baskın renk")
    style_category: Optional[str] = Field(default=None, description="Ana stil kategorisi")


class DetectionResult(BaseModel):
    """Tek bir detection sonucu"""
    label: str = Field(description="Fashion kategori etiketi")
    confidence: float = Field(ge=0.0, le=1.0, description="Güven skoru (0-1)")
    bbox: List[float] = Field(description="Bounding box [x, y, width, height]")
    bbox_normalized: Optional[BoundingBox] = Field(
        default=None, 
        description="Normalized bounding box"
    )
    area: float = Field(ge=0.0, description="Detection alanı (piksel)")
    category_id: int = Field(description="Fashionpedia kategori ID'si")
    attributes: Optional[FashionAttributes] = Field(
        default=None,
        description="Fashion özellikleri"
    )
    mask: Optional[List[List[int]]] = Field(
        default=None,
        description="Segmentation maskesi (opsiyonel)"
    )
    mask_encoded: Optional[str] = Field(
        default=None,
        description="Base64 encoded mask (daha kompakt)"
    )


class AnalysisSummary(BaseModel):
    """Analiz özeti"""
    total_detections: int = Field(description="Toplam detection sayısı")
    unique_categories: int = Field(description="Benzersiz kategori sayısı")
    average_confidence: float = Field(description="Ortalama güven skoru")
    confidence_distribution: Optional[Dict[str, int]] = Field(
        default=None,
        description="Güven skoru dağılımı"
    )
    image_dimensions: List[int] = Field(description="Görüntü boyutları [width, height]")
    categories_found: List[str] = Field(description="Bulunan kategoriler")
    attributes_summary: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="Öznitelik özeti"
    )


class ProcessingMetadata(BaseModel):
    """İşlem metadata'sı"""
    processing_time: float = Field(description="İşlem süresi (saniye)")
    image_filename: Optional[str] = Field(default=None, description="Görüntü dosya adı")
    image_size: Optional[int] = Field(default=None, description="Görüntü dosya boyutu (byte)")
    parameters_used: Dict[str, Any] = Field(description="Kullanılan parametreler")
    timestamp: str = Field(description="İşlem zamanı")
    model_version: Optional[str] = Field(default=None, description="Model versiyonu")
    api_version: Optional[str] = Field(default=None, description="API versiyonu")


class AnalysisResponse(BaseModel):
    """Görsel analiz yanıtı"""
    success: bool = Field(description="İşlem başarı durumu")
    detections: List[DetectionResult] = Field(description="Tespit edilen fashion itemlar")
    summary: AnalysisSummary = Field(description="Analiz özeti")
    metadata: ProcessingMetadata = Field(description="İşlem metadata'sı")
    model_info: Dict[str, Any] = Field(description="Model bilgileri")
    
    class Config:
        """Pydantic konfigürasyonu"""
        schema_extra = {
            "example": {
                "success": True,
                "detections": [
                    {
                        "label": "shirt, blouse",
                        "confidence": 0.89,
                        "bbox": [100.5, 150.2, 200.8, 300.1],
                        "area": 60240.8,
                        "category_id": 1,
                        "attributes": {
                            "colors": ["blue", "white"],
                            "patterns": ["striped"],
                            "styles": ["casual"],
                            "materials": ["cotton"],
                            "dominant_color": "blue"
                        }
                    }
                ],
                "summary": {
                    "total_detections": 1,
                    "unique_categories": 1,
                    "average_confidence": 0.89,
                    "image_dimensions": [800, 600],
                    "categories_found": ["shirt, blouse"]
                },
                "metadata": {
                    "processing_time": 2.34,
                    "image_filename": "fashion_image.jpg",
                    "parameters_used": {
                        "confidence_threshold": 0.7,
                        "max_detections": 50
                    },
                    "timestamp": "2025-08-03 10:30:45"
                }
            }
        }


class HealthResponse(BaseModel):
    """Sağlık kontrolü yanıtı"""
    status: str = Field(description="Servis durumu")
    model_loaded: bool = Field(description="Model yüklenme durumu")
    version: str = Field(description="API versiyonu")
    timestamp: str = Field(description="Zaman damgası")
    supported_formats: List[str] = Field(description="Desteklenen görüntü formatları")
    system_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Sistem bilgileri"
    )


class ErrorResponse(BaseModel):
    """Hata yanıtı"""
    error: str = Field(description="Hata mesajı")
    error_code: Optional[str] = Field(default=None, description="Hata kodu")
    details: Optional[str] = Field(default=None, description="Hata detayları")
    timestamp: str = Field(description="Hata zamanı")
    request_id: Optional[str] = Field(default=None, description="İstek ID'si")


class BatchAnalysisRequest(BaseModel):
    """Batch analiz isteği"""
    confidence_threshold: float = Field(default=0.7, ge=0.1, le=1.0)
    max_detections: int = Field(default=50, ge=1, le=200)
    return_masks: bool = Field(default=False)
    include_attributes: bool = Field(default=True)
    parallel_processing: bool = Field(
        default=True,
        description="Paralel işleme kullan"
    )
    max_files: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Maksimum dosya sayısı"
    )


class BatchAnalysisResponse(BaseModel):
    """Batch analiz yanıtı"""
    success: bool = Field(description="Genel başarı durumu")
    total_files: int = Field(description="Toplam dosya sayısı")
    successful_analyses: int = Field(description="Başarılı analiz sayısı")
    failed_analyses: int = Field(description="Başarısız analiz sayısı")
    results: List[AnalysisResponse] = Field(description="Her dosya için sonuç")
    batch_summary: Dict[str, Any] = Field(description="Batch özeti")
    total_processing_time: float = Field(description="Toplam işlem süresi")


class ModelInfo(BaseModel):
    """Model bilgileri"""
    model_name: str = Field(description="Model adı")
    model_type: str = Field(description="Model tipi")
    framework: str = Field(description="Kullanılan framework")
    total_classes: int = Field(description="Toplam sınıf sayısı")
    categories_count: int = Field(description="Kategori sayısı")
    model_size: Optional[str] = Field(default=None, description="Model boyutu")
    training_dataset: Optional[str] = Field(default=None, description="Eğitim veri seti")
    performance_metrics: Optional[Dict[str, float]] = Field(
        default=None,
        description="Performans metrikleri"
    )


class CategoryInfo(BaseModel):
    """Kategori bilgileri"""
    category_id: int = Field(description="Kategori ID'si")
    category_name: str = Field(description="Kategori adı")
    parent_category: Optional[str] = Field(default=None, description="Ana kategori")
    subcategories: Optional[List[str]] = Field(default=None, description="Alt kategoriler")
    typical_attributes: Optional[List[str]] = Field(
        default=None,
        description="Tipik öznitelikler"
    )


class APIStats(BaseModel):
    """API istatistikleri"""
    api_version: str = Field(description="API versiyonu")
    model_loaded: bool = Field(description="Model yüklenme durumu")
    supported_formats: List[str] = Field(description="Desteklenen formatlar")
    max_file_size_mb: int = Field(description="Maksimum dosya boyutu (MB)")
    max_batch_size: int = Field(description="Maksimum batch boyutu")
    endpoints: List[Dict[str, str]] = Field(description="Mevcut endpoint'ler")
    system_resources: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Sistem kaynakları"
    )
    uptime: Optional[str] = Field(default=None, description="Çalışma süresi")
