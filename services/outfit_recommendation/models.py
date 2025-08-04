"""
Pydantic Models for OutfitTransformer API - Aura Project
Bu modül, OutfitTransformer API'si için Pydantic response/request model'larını içerir.
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator
from pydantic.types import UUID4
import time

# Request Models

class OutfitCompatibilityRequest(BaseModel):
    """Outfit uyumluluk analizi request modeli"""
    item_ids: List[str] = Field(..., min_items=2, max_items=6, description="Analiz edilecek item ID'leri")
    return_detailed_scores: bool = Field(default=False, description="Detaylı skorları döndür")
    include_graph_analysis: bool = Field(default=False, description="Graph analizi ekle")
    
    @validator('item_ids')
    def validate_item_ids(cls, v):
        if not v:
            raise ValueError("Item IDs boş olamaz")
        
        # Duplicate check
        if len(v) != len(set(v)):
            raise ValueError("Duplicate item ID'ler")
        
        # ID format check (basit)
        for item_id in v:
            if not isinstance(item_id, str) or len(item_id.strip()) == 0:
                raise ValueError(f"Geçersiz item ID: {item_id}")
        
        return v

    class Config:
        schema_extra = {
            "example": {
                "item_ids": ["item_001", "item_002", "item_003"],
                "return_detailed_scores": True,
                "include_graph_analysis": False
            }
        }


class OutfitRecommendationRequest(BaseModel):
    """Outfit öneri request modeli"""
    seed_item_ids: List[str] = Field(..., min_items=1, description="Başlangıç item'ları")
    target_categories: Optional[List[str]] = Field(default=None, description="Hedef kategoriler")
    occasion: Optional[str] = Field(default=None, description="Durum (casual, formal, party, etc.)")
    season: Optional[str] = Field(default=None, description="Mevsim (spring, summer, fall, winter)")
    max_recommendations: int = Field(default=5, ge=1, le=20, description="Maksimum öneri sayısı")
    
    @validator('occasion')
    def validate_occasion(cls, v):
        if v is not None:
            valid_occasions = ["casual", "formal", "business", "party", "sport", "beach", "travel"]
            if v.lower() not in valid_occasions:
                raise ValueError(f"Geçersiz occasion. Geçerli değerler: {valid_occasions}")
        return v.lower() if v else None
    
    @validator('season')
    def validate_season(cls, v):
        if v is not None:
            valid_seasons = ["spring", "summer", "fall", "winter", "all"]
            if v.lower() not in valid_seasons:
                raise ValueError(f"Geçersiz season. Geçerli değerler: {valid_seasons}")
        return v.lower() if v else None

    class Config:
        schema_extra = {
            "example": {
                "seed_item_ids": ["item_001"],
                "target_categories": ["tops", "bottoms", "shoes"],
                "occasion": "casual",
                "season": "summer",
                "max_recommendations": 5
            }
        }


class ItemRecommendationRequest(BaseModel):
    """Item öneri request modeli"""
    item_id: str = Field(..., description="Hedef item ID")
    target_categories: Optional[List[str]] = Field(default=None, description="Önerilecek kategoriler")
    max_recommendations: int = Field(default=10, ge=1, le=50, description="Maksimum öneri sayısı")
    
    @validator('item_id')
    def validate_item_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Item ID boş olamaz")
        return v.strip()

    class Config:
        schema_extra = {
            "example": {
                "item_id": "item_001",
                "target_categories": ["bottoms", "shoes"],
                "max_recommendations": 10
            }
        }


class AddItemRequest(BaseModel):
    """Item ekleme request modeli"""
    item_id: str = Field(..., description="Unique item ID")
    category: str = Field(..., description="Item kategorisi")
    color: Optional[str] = Field(default=None, description="Item rengi")
    style: Optional[str] = Field(default=None, description="Item stili")
    price: Optional[float] = Field(default=None, ge=0, description="Item fiyatı")
    brand: Optional[str] = Field(default=None, description="Item markası")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Ek metadata")
    
    @validator('category')
    def validate_category(cls, v):
        valid_categories = [
            "tops", "bottoms", "dresses", "shoes", "accessories", 
            "outerwear", "swimwear", "activewear", "intimates", "sleepwear"
        ]
        if v.lower() not in valid_categories:
            raise ValueError(f"Geçersiz kategori. Geçerli değerler: {valid_categories}")
        return v.lower()

    class Config:
        schema_extra = {
            "example": {
                "item_id": "new_item_001",
                "category": "tops",
                "color": "blue",
                "style": "casual",
                "price": 29.99,
                "brand": "Fashion Brand",
                "metadata": {"material": "cotton", "size": "M"}
            }
        }


# Response Models

class FashionRulesResult(BaseModel):
    """Fashion kuralları analiz sonucu"""
    passed_rules: List[str] = Field(default=[], description="Geçen kurallar")
    failed_rules: List[str] = Field(default=[], description="Başarısız kurallar") 
    warnings: List[str] = Field(default=[], description="Uyarılar")
    overall_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Genel kural skoru")


class DetailedScores(BaseModel):
    """Detaylı skorlar"""
    compatibility_probs: List[float] = Field(description="Uyumluluk olasılıkları")
    outfit_scores: List[float] = Field(description="Outfit skorları")
    predicted_label: int = Field(description="Predicted label (0: incompatible, 1: compatible)")


class GraphAnalysis(BaseModel):
    """Graph analiz sonucu"""
    num_nodes: int = Field(description="Node sayısı")
    num_edges: int = Field(description="Edge sayısı")
    density: float = Field(description="Graph yoğunluğu")
    is_connected: bool = Field(description="Bağlantılı graph mı")
    average_compatibility: float = Field(description="Ortalama uyumluluk")
    num_components: Optional[int] = Field(default=None, description="Component sayısı")
    largest_component_size: Optional[int] = Field(default=None, description="En büyük component boyutu")
    most_central_item: Optional[Dict[str, Any]] = Field(default=None, description="En merkezi item")


class OutfitCompatibilityResponse(BaseModel):
    """Outfit uyumluluk analizi response"""
    outfit_id: str = Field(description="Outfit ID")
    item_ids: List[str] = Field(description="Item ID'leri")
    is_compatible: bool = Field(description="Uyumlu mu")
    compatibility_score: float = Field(ge=0.0, le=1.0, description="Uyumluluk skoru")
    outfit_score: float = Field(description="Outfit skoru")
    recommendation: str = Field(description="Öneri (compatible/incompatible)")
    detailed_scores: Optional[DetailedScores] = Field(default=None, description="Detaylı skorlar")
    fashion_rules: FashionRulesResult = Field(description="Fashion kuralları analizi")
    graph_analysis: Optional[GraphAnalysis] = Field(default=None, description="Graph analizi")
    processing_time_ms: float = Field(description="İşlem süresi (ms)")
    
    class Config:
        schema_extra = {
            "example": {
                "outfit_id": "outfit_item_001_item_002",
                "item_ids": ["item_001", "item_002", "item_003"],
                "is_compatible": True,
                "compatibility_score": 0.85,
                "outfit_score": 0.78,
                "recommendation": "compatible",
                "fashion_rules": {
                    "passed_rules": ["category_balance", "color_harmony"],
                    "failed_rules": [],
                    "warnings": [],
                    "overall_score": 1.0
                },
                "processing_time_ms": 125.5
            }
        }


class ItemDetail(BaseModel):
    """Item detay bilgisi"""
    item_id: str = Field(description="Item ID")
    category: str = Field(description="Kategori")
    color: Optional[str] = Field(description="Renk")
    style: Optional[str] = Field(description="Stil")
    price: Optional[float] = Field(description="Fiyat")
    brand: Optional[str] = Field(description="Marka")
    image_path: Optional[str] = Field(description="Görüntü path")
    similarity_score: Optional[float] = Field(description="Benzerlik skoru")
    compatibility_with_target: Optional[float] = Field(description="Hedef ile uyumluluk")
    recommendation_reason: Optional[str] = Field(description="Öneri sebebi")
    metadata: Optional[Dict[str, Any]] = Field(description="Ek bilgiler")


class OutfitRecommendation(BaseModel):
    """Outfit önerisi"""
    outfit_id: str = Field(description="Outfit ID")
    items: List[ItemDetail] = Field(description="Outfit item'ları")
    compatibility: OutfitCompatibilityResponse = Field(description="Uyumluluk analizi")
    occasion: Optional[str] = Field(description="Durum")
    season: Optional[str] = Field(description="Mevsim")
    generation_method: str = Field(description="Oluşturma yöntemi")


class OutfitRecommendationResponse(BaseModel):
    """Outfit öneri response"""
    seed_item_ids: List[str] = Field(description="Başlangıç item'ları")
    recommendations: List[OutfitRecommendation] = Field(description="Outfit önerileri")
    total_recommendations: int = Field(description="Toplam öneri sayısı")
    filters_applied: Dict[str, Any] = Field(description="Uygulanan filtreler")
    processing_time_ms: float = Field(description="İşlem süresi (ms)")
    
    class Config:
        schema_extra = {
            "example": {
                "seed_item_ids": ["item_001"],
                "recommendations": [
                    {
                        "outfit_id": "rec_outfit_001",
                        "items": [
                            {
                                "item_id": "item_001",
                                "category": "tops",
                                "color": "blue",
                                "style": "casual"
                            }
                        ],
                        "compatibility": {
                            "compatibility_score": 0.89,
                            "is_compatible": True
                        },
                        "occasion": "casual",
                        "generation_method": "transformer_based"
                    }
                ],
                "total_recommendations": 1,
                "filters_applied": {
                    "target_categories": ["bottoms", "shoes"],
                    "occasion": "casual"
                },
                "processing_time_ms": 234.6
            }
        }


class ItemRecommendationResponse(BaseModel):
    """Item öneri response"""
    item_id: str = Field(description="Hedef item ID")
    recommendations: List[ItemDetail] = Field(description="Item önerileri")
    total_recommendations: int = Field(description="Toplam öneri sayısı")
    filters_applied: Dict[str, Any] = Field(description="Uygulanan filtreler")
    
    class Config:
        schema_extra = {
            "example": {
                "item_id": "item_001",
                "recommendations": [
                    {
                        "item_id": "item_002",
                        "category": "bottoms",
                        "compatibility_with_target": 0.82,
                        "recommendation_reason": "Compatible bottoms for tops"
                    }
                ],
                "total_recommendations": 1,
                "filters_applied": {
                    "target_categories": ["bottoms"]
                }
            }
        }


class AddItemResponse(BaseModel):
    """Item ekleme response"""
    success: bool = Field(description="Başarılı mı")
    item_id: str = Field(description="Eklenen item ID")
    item_data: ItemDetail = Field(description="Item verileri")
    message: str = Field(description="Durum mesajı")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "item_id": "new_item_001",
                "item_data": {
                    "item_id": "new_item_001",
                    "category": "tops",
                    "color": "blue",
                    "style": "casual"
                },
                "message": "Item başarıyla eklendi"
            }
        }


class OutfitAnalysisResponse(BaseModel):
    """Detaylı outfit analiz response"""
    outfit_id: str = Field(description="Outfit ID")
    item_ids: List[str] = Field(description="Item ID'leri")
    items_detail: List[ItemDetail] = Field(description="Item detayları")
    compatibility_analysis: OutfitCompatibilityResponse = Field(description="Uyumluluk analizi")
    graph_analysis: GraphAnalysis = Field(description="Graph analizi")
    recommendations: Dict[str, List[str]] = Field(description="İyileştirme önerileri")
    timestamp: float = Field(description="Analiz zamanı")
    
    class Config:
        schema_extra = {
            "example": {
                "outfit_id": "analysis_outfit_001",
                "item_ids": ["item_001", "item_002"],
                "items_detail": [],
                "compatibility_analysis": {
                    "compatibility_score": 0.75,
                    "is_compatible": True
                },
                "graph_analysis": {
                    "num_nodes": 2,
                    "num_edges": 1,
                    "density": 1.0,
                    "is_connected": True,
                    "average_compatibility": 0.75
                },
                "recommendations": {
                    "improvement_suggestions": [],
                    "alternative_items": [],
                    "styling_tips": []
                },
                "timestamp": 1234567890.0
            }
        }


class DatabaseStatsResponse(BaseModel):
    """Veritabanı istatistikleri response"""
    total_items: int = Field(description="Toplam item sayısı")
    categories: Dict[str, int] = Field(default={}, description="Kategori dağılımı")
    colors: Dict[str, int] = Field(default={}, description="Renk dağılımı")
    styles: Dict[str, int] = Field(default={}, description="Stil dağılımı")
    has_faiss_index: bool = Field(description="FAISS index var mı")
    embedding_dimension: int = Field(description="Embedding boyutu")
    timestamp: float = Field(description="İstatistik zamanı")
    
    class Config:
        schema_extra = {
            "example": {
                "total_items": 1500,
                "categories": {
                    "tops": 300,
                    "bottoms": 250,
                    "shoes": 200
                },
                "colors": {
                    "black": 200,
                    "white": 180,
                    "blue": 150
                },
                "styles": {
                    "casual": 600,
                    "formal": 400,
                    "trendy": 300
                },
                "has_faiss_index": True,
                "embedding_dimension": 512,
                "timestamp": 1234567890.0
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(description="Sistem durumu")
    timestamp: float = Field(description="Kontrol zamanı")
    version: str = Field(description="API versiyonu")
    model_loaded: bool = Field(description="Model yüklü mü")
    demo_mode: bool = Field(description="Demo mode aktif mi")
    auth_enabled: bool = Field(description="Authentication aktif mi")
    database_stats: Optional[Dict[str, Any]] = Field(default=None, description="Database istatistikleri")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": 1234567890.0,
                "version": "1.0.0",
                "model_loaded": True,
                "demo_mode": False,
                "auth_enabled": False,
                "database_stats": {
                    "total_items": 1500,
                    "has_faiss_index": True
                }
            }
        }


class ErrorResponse(BaseModel):
    """Hata response"""
    error: str = Field(description="Hata kodu")
    message: str = Field(description="Hata mesajı")
    status_code: int = Field(description="HTTP status kodu")
    timestamp: float = Field(description="Hata zamanı")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Hata detayları")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "VALIDATION_ERROR",
                "message": "Invalid item ID format",
                "status_code": 400,
                "timestamp": 1234567890.0,
                "details": {
                    "field": "item_ids",
                    "value": "invalid_id"
                }
            }
        }


# Additional utility models

class BatchCompatibilityRequest(BaseModel):
    """Batch outfit uyumluluk request"""
    outfit_combinations: List[List[str]] = Field(..., description="Outfit kombinasyonları")
    return_detailed_scores: bool = Field(default=False, description="Detaylı skorları döndür")
    
    @validator('outfit_combinations')
    def validate_combinations(cls, v):
        if not v:
            raise ValueError("En az 1 outfit kombinasyonu gerekli")
        
        for i, combination in enumerate(v):
            if len(combination) < 2:
                raise ValueError(f"Kombinasyon {i}: En az 2 item gerekli")
            if len(combination) > 6:
                raise ValueError(f"Kombinasyon {i}: Maksimum 6 item destekleniyor")
        
        return v

    class Config:
        schema_extra = {
            "example": {
                "outfit_combinations": [
                    ["item_001", "item_002"],
                    ["item_003", "item_004", "item_005"]
                ],
                "return_detailed_scores": False
            }
        }


class BatchCompatibilityResponse(BaseModel):
    """Batch outfit uyumluluk response"""
    results: List[OutfitCompatibilityResponse] = Field(description="Batch sonuçları")
    total_processed: int = Field(description="İşlenen toplam kombinasyon")
    processing_time_ms: float = Field(description="Toplam işlem süresi")
    
    class Config:
        schema_extra = {
            "example": {
                "results": [
                    {
                        "outfit_id": "batch_outfit_001",
                        "compatibility_score": 0.85,
                        "is_compatible": True
                    }
                ],
                "total_processed": 1,
                "processing_time_ms": 456.7
            }
        }


# Enum-like constants for validation
VALID_CATEGORIES = [
    "tops", "bottoms", "dresses", "shoes", "accessories",
    "outerwear", "swimwear", "activewear", "intimates", "sleepwear"
]

VALID_COLORS = [
    "black", "white", "gray", "brown", "beige", "navy", "blue", "light_blue",
    "red", "pink", "purple", "green", "yellow", "orange", "gold", "silver",
    "multicolor", "unknown"
]

VALID_STYLES = [
    "casual", "formal", "business", "trendy", "classic", "bohemian", "vintage",
    "sporty", "elegant", "edgy", "minimalist", "romantic", "unknown"
]

VALID_OCCASIONS = [
    "casual", "formal", "business", "party", "sport", "beach", "travel",
    "date", "work", "weekend", "evening", "daytime"
]

VALID_SEASONS = [
    "spring", "summer", "fall", "winter", "all"
]


# Model validation utilities
def validate_item_id_format(item_id: str) -> bool:
    """Item ID format validation"""
    if not item_id or not isinstance(item_id, str):
        return False
    
    # Basit format check
    if len(item_id.strip()) == 0:
        return False
    
    # Special characters check (isteğe bağlı)
    import re
    if not re.match(r'^[a-zA-Z0-9_-]+$', item_id):
        return False
    
    return True


def validate_category(category: str) -> bool:
    """Kategori validation"""
    return category.lower() in VALID_CATEGORIES


def validate_color(color: str) -> bool:
    """Renk validation"""
    return color.lower() in VALID_COLORS


def validate_style(style: str) -> bool:
    """Stil validation"""
    return style.lower() in VALID_STYLES
