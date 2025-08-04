"""
OutfitTransformer FastAPI Application - Aura Project
Bu modül, OutfitTransformer tabanlı outfit recommendation microservice'ini sağlar.
"""

import os
import logging
import time
import uuid
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import asyncio
import json

# FastAPI imports
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Pydantic models
from pydantic import BaseModel, Field, validator
from pydantic.types import UUID4

# Core libraries
import torch
import numpy as np
from PIL import Image
import io

# Local imports
from inference import (
    OutfitRecommendationEngine,
    load_outfit_engine,
    create_demo_items_database
)
from models import (
    OutfitCompatibilityRequest,
    OutfitCompatibilityResponse,
    OutfitRecommendationRequest,
    OutfitRecommendationResponse,
    ItemRecommendationRequest,
    ItemRecommendationResponse,
    AddItemRequest,
    AddItemResponse,
    OutfitAnalysisResponse,
    HealthResponse,
    DatabaseStatsResponse,
    ErrorResponse
)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App configuration
APP_CONFIG = {
    "title": "Aura Outfit Recommendation API",
    "description": "AI-powered fashion outfit recommendation and compatibility analysis using OutfitTransformer",
    "version": "1.0.0",
    "model_path": os.getenv("MODEL_PATH", "./models/outfit_transformer_best.pt"),
    "items_database_path": os.getenv("ITEMS_DB_PATH", "./data/items_database.json"),
    "upload_dir": "./uploads",
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "allowed_image_extensions": {".jpg", ".jpeg", ".png", ".bmp", ".tiff"},
    "enable_auth": os.getenv("ENABLE_AUTH", "false").lower() == "true",
    "demo_mode": os.getenv("DEMO_MODE", "false").lower() == "true"
}

# Global engine instance
recommendation_engine: Optional[OutfitRecommendationEngine] = None

# FastAPI app
app = FastAPI(
    title=APP_CONFIG["title"],
    description=APP_CONFIG["description"],
    version=APP_CONFIG["version"],
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production'da specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)


# Dependency functions
async def get_auth_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Authentication dependency"""
    if not APP_CONFIG["enable_auth"]:
        return None
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    # Simple token validation (production'da JWT veya database check)
    valid_tokens = ["aura-demo-token", "development-token"]
    if credentials.credentials not in valid_tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    
    return credentials.credentials


async def get_recommendation_engine():
    """Recommendation engine dependency"""
    global recommendation_engine
    
    if recommendation_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation engine not initialized"
        )
    
    return recommendation_engine


# Utility functions
def validate_image_file(file: UploadFile) -> bool:
    """Görüntü dosyası validation"""
    if not file.content_type.startswith("image/"):
        return False
    
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in APP_CONFIG["allowed_image_extensions"]:
        return False
    
    return True


async def save_uploaded_file(file: UploadFile) -> str:
    """Upload edilen dosyayı kaydet"""
    # Upload directory oluştur
    upload_dir = Path(APP_CONFIG["upload_dir"])
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Unique filename
    file_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix.lower()
    filename = f"{file_id}{file_extension}"
    file_path = upload_dir / filename
    
    # File'ı kaydet
    content = await file.read()
    
    if len(content) > APP_CONFIG["max_file_size"]:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size: {APP_CONFIG['max_file_size']} bytes"
        )
    
    with open(file_path, "wb") as f:
        f.write(content)
    
    return str(file_path)


# Startup/Shutdown events
@app.on_event("startup")
async def startup_event():
    """Application startup"""
    global recommendation_engine
    
    logger.info("OutfitTransformer API başlatılıyor...")
    
    try:
        # Model path check
        model_path = APP_CONFIG["model_path"]
        
        if APP_CONFIG["demo_mode"]:
            logger.info("Demo mode aktif")
            
            # Demo model ve database oluştur (eğer yoksa)
            if not Path(model_path).exists():
                logger.warning(f"Model bulunamadı: {model_path}")
                logger.info("Demo mode'da mock model kullanılacak")
            
            # Demo items database
            demo_db_path = "./data/demo_items_database.json"
            if not Path(demo_db_path).exists():
                Path(demo_db_path).parent.mkdir(parents=True, exist_ok=True)
                create_demo_items_database("./demo_images", demo_db_path)
                APP_CONFIG["items_database_path"] = demo_db_path
        
        # Engine yükle
        if Path(model_path).exists():
            recommendation_engine = load_outfit_engine(
                model_path=model_path,
                items_database_path=APP_CONFIG["items_database_path"],
                config={"device": "auto"}
            )
            logger.info("OutfitTransformer engine yüklendi")
        else:
            logger.warning("Model yüklenemedi, mock engine kullanılacak")
            # Mock engine (development için)
            recommendation_engine = None
    
    except Exception as e:
        logger.error(f"Startup hatası: {e}")
        # Production'da critical error, development'ta continue
        if not APP_CONFIG["demo_mode"]:
            raise


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info("OutfitTransformer API kapatılıyor...")
    
    global recommendation_engine
    recommendation_engine = None


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Sistem sağlık kontrolü"""
    global recommendation_engine
    
    status_info = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": APP_CONFIG["version"],
        "model_loaded": recommendation_engine is not None,
        "demo_mode": APP_CONFIG["demo_mode"],
        "auth_enabled": APP_CONFIG["enable_auth"]
    }
    
    if recommendation_engine:
        try:
            db_stats = recommendation_engine.get_database_stats()
            status_info["database_stats"] = db_stats
        except Exception as e:
            logger.warning(f"Database stats alınamadı: {e}")
            status_info["database_stats"] = {"error": str(e)}
    
    return HealthResponse(**status_info)


# Main API endpoints
@app.post("/outfit/compatibility", response_model=OutfitCompatibilityResponse, tags=["Outfit Analysis"])
async def predict_outfit_compatibility(
    request: OutfitCompatibilityRequest,
    engine: OutfitRecommendationEngine = Depends(get_recommendation_engine),
    token: str = Depends(get_auth_token)
):
    """
    Outfit uyumluluğunu analiz eder
    
    Verilen item ID'leri için outfit uyumluluğunu predict eder ve
    detaylı analiz sonuçları döndürür.
    """
    try:
        logger.info(f"Compatibility analizi: {request.item_ids}")
        
        # Validation
        if len(request.item_ids) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="En az 2 item ID gerekli"
            )
        
        if len(request.item_ids) > 6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maksimum 6 item destekleniyor"
            )
        
        # Compatibility prediction
        result = engine.predict_outfit_compatibility(
            item_ids=request.item_ids,
            return_scores=request.return_detailed_scores
        )
        
        # Graph analysis (eğer istenirse)
        graph_analysis = None
        if request.include_graph_analysis:
            try:
                graph_analysis = engine.analyze_outfit_graph(request.item_ids)
            except Exception as e:
                logger.warning(f"Graph analizi hatası: {e}")
        
        response = OutfitCompatibilityResponse(
            outfit_id=result["outfit_id"],
            item_ids=result["item_ids"],
            is_compatible=result["is_compatible"],
            compatibility_score=result["compatibility_score"],
            outfit_score=result["outfit_score"],
            recommendation=result["recommendation"],
            detailed_scores=result.get("detailed_scores"),
            fashion_rules=result["fashion_rules"],
            graph_analysis=graph_analysis,
            processing_time_ms=0  # Basit implementation
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Compatibility prediction hatası: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction hatası: {str(e)}"
        )


@app.post("/outfit/recommendations", response_model=OutfitRecommendationResponse, tags=["Outfit Recommendations"])
async def generate_outfit_recommendations(
    request: OutfitRecommendationRequest,
    engine: OutfitRecommendationEngine = Depends(get_recommendation_engine),
    token: str = Depends(get_auth_token)
):
    """
    Outfit önerileri oluşturur
    
    Verilen seed item'lara göre uyumlu outfit kombinasyonları önerir.
    """
    try:
        logger.info(f"Outfit önerileri: {request.seed_item_ids}")
        
        # Validation
        if not request.seed_item_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="En az 1 seed item gerekli"
            )
        
        # Generate recommendations
        recommendations = engine.generate_outfit_recommendations(
            seed_item_ids=request.seed_item_ids,
            target_categories=request.target_categories,
            occasion=request.occasion,
            season=request.season,
            max_outfits=request.max_recommendations
        )
        
        response = OutfitRecommendationResponse(
            seed_item_ids=request.seed_item_ids,
            recommendations=recommendations,
            total_recommendations=len(recommendations),
            filters_applied={
                "target_categories": request.target_categories,
                "occasion": request.occasion,
                "season": request.season
            },
            processing_time_ms=0
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Outfit recommendation hatası: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recommendation hatası: {str(e)}"
        )


@app.post("/items/recommendations", response_model=ItemRecommendationResponse, tags=["Item Recommendations"])
async def get_item_recommendations(
    request: ItemRecommendationRequest,
    engine: OutfitRecommendationEngine = Depends(get_recommendation_engine),
    token: str = Depends(get_auth_token)
):
    """
    Item önerileri
    
    Belirli bir item için uyumlu diğer item'ları önerir.
    """
    try:
        logger.info(f"Item önerileri: {request.item_id}")
        
        recommendations = engine.get_item_recommendations(
            item_id=request.item_id,
            categories=request.target_categories,
            top_k=request.max_recommendations
        )
        
        response = ItemRecommendationResponse(
            item_id=request.item_id,
            recommendations=recommendations,
            total_recommendations=len(recommendations),
            filters_applied={
                "target_categories": request.target_categories
            }
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Item recommendation hatası: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Item recommendation hatası: {str(e)}"
        )


@app.post("/items/add", response_model=AddItemResponse, tags=["Item Management"])
async def add_item_to_database(
    item_id: str,
    category: str,
    color: Optional[str] = None,
    style: Optional[str] = None,
    price: Optional[float] = None,
    brand: Optional[str] = None,
    image: UploadFile = File(...),
    engine: OutfitRecommendationEngine = Depends(get_recommendation_engine),
    token: str = Depends(get_auth_token)
):
    """
    Yeni item ekler
    
    Görüntü upload ederek veritabanına yeni item ekler.
    """
    try:
        logger.info(f"Yeni item ekleniyor: {item_id}")
        
        # Image validation
        if not validate_image_file(image):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Geçersiz görüntü formatı"
            )
        
        # Save image
        image_path = await save_uploaded_file(image)
        
        # Add to database
        item_data = engine.add_item_to_database(
            item_id=item_id,
            image_path=image_path,
            category=category,
            color=color,
            style=style,
            price=price,
            brand=brand,
            metadata={"upload_timestamp": time.time()}
        )
        
        response = AddItemResponse(
            success=True,
            item_id=item_id,
            item_data=item_data,
            message="Item başarıyla eklendi"
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Item ekleme hatası: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Item ekleme hatası: {str(e)}"
        )


@app.get("/database/stats", response_model=DatabaseStatsResponse, tags=["Database"])
async def get_database_statistics(
    engine: OutfitRecommendationEngine = Depends(get_recommendation_engine),
    token: str = Depends(get_auth_token)
):
    """Veritabanı istatistikleri"""
    try:
        stats = engine.get_database_stats()
        
        response = DatabaseStatsResponse(
            **stats,
            timestamp=time.time()
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Database stats hatası: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database stats hatası: {str(e)}"
        )


@app.get("/outfit/analyze/{outfit_id}", response_model=OutfitAnalysisResponse, tags=["Outfit Analysis"])
async def analyze_outfit(
    outfit_id: str,
    item_ids: str,  # Comma-separated item IDs
    engine: OutfitRecommendationEngine = Depends(get_recommendation_engine),
    token: str = Depends(get_auth_token)
):
    """
    Detaylı outfit analizi
    
    Outfit'in uyumluluğunu, graph analizini ve fashion rule'larını analiz eder.
    """
    try:
        # Parse item IDs
        item_id_list = [item_id.strip() for item_id in item_ids.split(",")]
        
        if len(item_id_list) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="En az 2 item ID gerekli"
            )
        
        # Compatibility analysis
        compatibility = engine.predict_outfit_compatibility(
            item_ids=item_id_list,
            return_scores=True
        )
        
        # Graph analysis
        graph_analysis = engine.analyze_outfit_graph(item_id_list)
        
        # Item details
        items_detail = []
        for item_id in item_id_list:
            if item_id in engine.item_database:
                item_data = engine.item_database[item_id].copy()
                # Remove embedding (too large for response)
                item_data.pop("image_embedding", None)
                items_detail.append(item_data)
        
        response = OutfitAnalysisResponse(
            outfit_id=outfit_id,
            item_ids=item_id_list,
            items_detail=items_detail,
            compatibility_analysis=compatibility,
            graph_analysis=graph_analysis,
            recommendations={
                "improvement_suggestions": [],
                "alternative_items": [],
                "styling_tips": []
            },
            timestamp=time.time()
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Outfit analizi hatası: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Outfit analizi hatası: {str(e)}"
        )


# Demo endpoints (development/testing için)
@app.get("/demo/sample-outfits", tags=["Demo"])
async def get_sample_outfits(
    engine: OutfitRecommendationEngine = Depends(get_recommendation_engine)
):
    """Demo outfit örnekleri"""
    if not APP_CONFIG["demo_mode"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Demo mode aktif değil"
        )
    
    # Sample outfit combinations
    db_stats = engine.get_database_stats()
    total_items = db_stats.get("total_items", 0)
    
    if total_items < 4:
        return {"message": "Yeterli item yok", "total_items": total_items}
    
    # Create sample outfits
    all_item_ids = list(engine.item_database.keys())
    sample_outfits = []
    
    for i in range(min(3, total_items // 3)):
        start_idx = i * 3
        outfit_items = all_item_ids[start_idx:start_idx + 3]
        
        try:
            compatibility = engine.predict_outfit_compatibility(outfit_items)
            sample_outfits.append({
                "outfit_id": f"demo_outfit_{i+1}",
                "items": outfit_items,
                "compatibility": compatibility
            })
        except Exception as e:
            logger.warning(f"Demo outfit hatası: {e}")
    
    return {
        "sample_outfits": sample_outfits,
        "total_samples": len(sample_outfits),
        "database_stats": db_stats
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTP_ERROR",
            message=exc.detail,
            status_code=exc.status_code,
            timestamp=time.time()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="INTERNAL_SERVER_ERROR",
            message="Internal server error occurred",
            status_code=500,
            timestamp=time.time()
        ).dict()
    )


# Main execution
if __name__ == "__main__":
    import uvicorn
    
    # Development server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
