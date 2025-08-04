"""
Vector Store Builder for Aura Fashion Assistant RAG System
===========================================================

Bu modül, Aura projesinin sohbet asistanı için RAG (Retrieval-Augmented Generation) 
sisteminin vektör veritabanını oluşturur.

Kullanıcının gardırop verilerini alır, sentence-transformers ile vektörlere dönüştürür
ve FAISS veya Pinecone vektör veritabanında saklar.

Özellikler:
- Sentence Transformers ile semantic embeddings
- FAISS ve Pinecone vektör veritabanı desteği
- Yapılandırılmış gardırop verisi işleme
- Visual Analysis servis entegrasyonu
- Verimli batch processing
- Metadata filtreleme desteği

Author: Aura AI Team
Version: 1.0.0
Date: 2025-08-03
"""

import os
import sys
import json
import logging
import argparse
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime

# Core libraries
import numpy as np
import pandas as pd

# Embedding libraries
from sentence_transformers import SentenceTransformer
import torch

# Vector store libraries
import faiss
try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    warnings.warn("Pinecone not available. Only FAISS will be supported.")

# Utility libraries
from loguru import logger
import yaml
from tqdm import tqdm
import uuid

# Configuration
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# A1. MODÜL VE KÜTÜPHANELERİ İÇE AKTARMA - TAMAMLANDI
# =============================================================================

# =============================================================================
# A2. KONFİGÜRASYON VE ARGÜMANLARI TANIMLAMA
# =============================================================================

@dataclass
class VectorStoreConfig:
    """Vector store oluşturma için konfigürasyon sınıfı"""
    
    # Input/Output Configuration
    input_data_path: str = "./data/user_wardrobe.json"
    output_dir: str = "./vector_stores"
    
    # Embedding Configuration
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384  # all-MiniLM-L6-v2 için
    max_seq_length: int = 512
    
    # Vector Store Configuration
    vector_store_type: str = "faiss"  # "faiss" veya "pinecone"
    faiss_index_type: str = "IndexFlatIP"  # Inner Product (cosine similarity)
    
    # Pinecone Configuration (eğer kullanılıyorsa)
    pinecone_api_key: str = ""
    pinecone_environment: str = "us-west1-gcp-free"
    pinecone_index_name: str = "aura-wardrobe"
    pinecone_metric: str = "cosine"
    
    # Processing Configuration
    batch_size: int = 32
    normalize_embeddings: bool = True
    include_metadata: bool = True
    
    # Filtering Configuration
    min_confidence: float = 0.5  # Visual analysis minimum confidence
    supported_categories: List[str] = None
    
    def __post_init__(self):
        """Post-initialization setup"""
        if self.supported_categories is None:
            self.supported_categories = [
                "shirt", "t-shirt", "blouse", "sweater", "jacket", "coat",
                "dress", "skirt", "pants", "jeans", "shorts", 
                "shoes", "boots", "sneakers", "bag", "handbag", "accessories"
            ]


def parse_arguments() -> argparse.Namespace:
    """
    A2a. Komut satırı argümanlarını tanımla ve parse et
    
    Returns:
        argparse.Namespace: Parse edilmiş argümanlar
    """
    parser = argparse.ArgumentParser(
        description="Aura Fashion Assistant - Vector Store Builder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output arguments
    parser.add_argument(
        "--input_data_path",
        type=str,
        default="./data/user_wardrobe.json",
        help="Path to user wardrobe data (JSON file or directory)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./vector_stores",
        help="Output directory for vector store files"
    )
    
    # Embedding arguments
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence transformer model name"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length for embeddings"
    )
    
    # Vector store arguments
    parser.add_argument(
        "--vector_store_type",
        type=str,
        choices=["faiss", "pinecone"],
        default="faiss",
        help="Type of vector store to use"
    )
    parser.add_argument(
        "--faiss_index_type",
        type=str,
        choices=["IndexFlatIP", "IndexFlatL2", "IndexIVFFlat"],
        default="IndexFlatIP",
        help="FAISS index type"
    )
    
    # Pinecone arguments
    parser.add_argument(
        "--pinecone_api_key",
        type=str,
        default="",
        help="Pinecone API key (or set PINECONE_API_KEY env var)"
    )
    parser.add_argument(
        "--pinecone_environment",
        type=str,
        default="us-west1-gcp-free",
        help="Pinecone environment"
    )
    parser.add_argument(
        "--pinecone_index_name",
        type=str,
        default="aura-wardrobe",
        help="Pinecone index name"
    )
    
    # Processing arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding generation"
    )
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for visual analysis results"
    )
    
    # Technical arguments
    parser.add_argument(
        "--normalize_embeddings",
        action="store_true",
        default=True,
        help="Normalize embeddings for cosine similarity"
    )
    parser.add_argument(
        "--include_metadata",
        action="store_true",
        default=True,
        help="Include metadata in vector store"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode with verbose logging"
    )
    
    return parser.parse_args()


def setup_logging(debug: bool = False) -> None:
    """
    A2b. Logging sistemini yapılandır
    
    Args:
        debug: Debug seviyesinde loglama aktif edilsin mi
    """
    # Remove default logger
    logger.remove()
    
    # Console logging
    log_level = "DEBUG" if debug else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # File logging
    log_file = f"vector_store_build_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days"
    )
    
    logger.info("🚀 Aura Vector Store Builder başlatılıyor...")
    logger.info(f"📝 Log dosyası: {log_file}")


# =============================================================================
# A3. EMBEDDING MODELİNİ YÜKLEME
# =============================================================================

def load_embedding_model(config: VectorStoreConfig) -> SentenceTransformer:
    """
    A3a. Sentence Transformer embedding modelini yükle
    
    Args:
        config: Vector store konfigürasyonu
        
    Returns:
        SentenceTransformer: Yüklenmiş embedding modeli
    """
    logger.info(f"📥 Embedding modeli yükleniyor: {config.embedding_model_name}")
    
    try:
        # Sentence Transformer modelini yükle
        model = SentenceTransformer(
            config.embedding_model_name,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Model konfigürasyonlarını ayarla
        if hasattr(model, 'max_seq_length'):
            model.max_seq_length = config.max_seq_length
        
        logger.success(f"✅ Embedding modeli yüklendi: {config.embedding_model_name}")
        
        # Model bilgilerini göster
        logger.info(f"📊 Model bilgileri:")
        logger.info(f"  - Model name: {config.embedding_model_name}")
        logger.info(f"  - Embedding dimension: {model.get_sentence_embedding_dimension()}")
        logger.info(f"  - Max sequence length: {getattr(model, 'max_seq_length', 'N/A')}")
        logger.info(f"  - Device: {model.device}")
        
        # Embedding dimension'ı config'e kaydet
        config.embedding_dimension = model.get_sentence_embedding_dimension()
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Embedding modeli yükleme hatası: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


# =============================================================================
# A4. KULLANICI GARDIROBı VERISINI YÜKLEME
# =============================================================================

def load_wardrobe_data(config: VectorStoreConfig) -> List[Dict[str, Any]]:
    """
    A4a. Kullanıcı gardırop verisini yükle ve işle
    
    Args:
        config: Vector store konfigürasyonu
        
    Returns:
        List[Dict[str, Any]]: İşlenmiş gardırop verisi
    """
    logger.info(f"📊 Gardırop verisi yükleniyor: {config.input_data_path}")
    
    try:
        input_path = Path(config.input_data_path)
        
        if not input_path.exists():
            # Örnek veri oluştur
            logger.warning(f"⚠️ Veri dosyası bulunamadı: {input_path}")
            logger.info("📝 Örnek gardırop verisi oluşturuluyor...")
            
            sample_data = create_sample_wardrobe_data()
            
            # Örnek veriyi kaydet
            input_path.parent.mkdir(parents=True, exist_ok=True)
            with open(input_path, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f, indent=2, ensure_ascii=False)
            
            logger.success(f"✅ Örnek veri oluşturuldu: {input_path}")
            return sample_data
        
        # Dosyayı yükle
        if input_path.is_file():
            # Tek dosya
            if input_path.suffix.lower() == '.json':
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                raise ValueError(f"Desteklenmeyen dosya formatı: {input_path.suffix}")
                
        elif input_path.is_dir():
            # Dizindeki tüm JSON dosyalarını yükle
            data = []
            json_files = list(input_path.glob("*.json"))
            
            if not json_files:
                raise FileNotFoundError(f"Dizinde JSON dosyası bulunamadı: {input_path}")
            
            for json_file in json_files:
                logger.info(f"  📁 Yükleniyor: {json_file.name}")
                with open(json_file, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                    if isinstance(file_data, list):
                        data.extend(file_data)
                    else:
                        data.append(file_data)
        
        logger.info(f"📊 Ham veri boyutu: {len(data)} item")
        
        # Veriyi işle ve filtrele
        processed_data = process_wardrobe_data(data, config)
        
        logger.success(f"✅ Gardırop verisi yüklendi: {len(processed_data)} işlenmiş item")
        
        return processed_data
        
    except Exception as e:
        logger.error(f"❌ Veri yükleme hatası: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


def create_sample_wardrobe_data() -> List[Dict[str, Any]]:
    """
    A4b. Örnek gardırop verisi oluştur (Visual Analysis formatında)
    
    Returns:
        List[Dict[str, Any]]: Örnek gardırop verisi
    """
    sample_items = [
        {
            "id": str(uuid.uuid4()),
            "user_id": "user_123",
            "image_path": "/images/blue_jeans.jpg",
            "visual_analysis": {
                "detections": [
                    {
                        "label": "jeans",
                        "confidence": 0.95,
                        "bbox": [10, 20, 200, 400],
                        "attributes": {
                            "colors": ["blue", "denim"],
                            "patterns": ["solid"],
                            "styles": ["casual", "straight-leg"],
                            "materials": ["denim", "cotton"]
                        }
                    }
                ],
                "summary": {
                    "total_detections": 1,
                    "categories_found": ["jeans"]
                }
            },
            "user_tags": ["casual", "everyday", "comfortable"],
            "brand": "Levi's",
            "size": "M",
            "color_primary": "blue",
            "season": ["spring", "fall", "winter"],
            "occasion": ["casual", "work"],
            "purchase_date": "2024-01-15",
            "last_worn": "2024-07-20"
        },
        {
            "id": str(uuid.uuid4()),
            "user_id": "user_123",
            "image_path": "/images/white_shirt.jpg",
            "visual_analysis": {
                "detections": [
                    {
                        "label": "shirt",
                        "confidence": 0.92,
                        "bbox": [15, 10, 180, 250],
                        "attributes": {
                            "colors": ["white"],
                            "patterns": ["solid"],
                            "styles": ["business", "button-down"],
                            "materials": ["cotton", "polyester"]
                        }
                    }
                ],
                "summary": {
                    "total_detections": 1,
                    "categories_found": ["shirt"]
                }
            },
            "user_tags": ["business", "formal", "versatile"],
            "brand": "Brooks Brothers",
            "size": "L",
            "color_primary": "white",
            "season": ["spring", "summer", "fall"],
            "occasion": ["business", "formal", "date"],
            "purchase_date": "2024-02-10",
            "last_worn": "2024-07-25"
        },
        {
            "id": str(uuid.uuid4()),
            "user_id": "user_123",
            "image_path": "/images/black_dress.jpg",
            "visual_analysis": {
                "detections": [
                    {
                        "label": "dress",
                        "confidence": 0.98,
                        "bbox": [5, 5, 190, 380],
                        "attributes": {
                            "colors": ["black"],
                            "patterns": ["solid"],
                            "styles": ["cocktail", "sleeveless"],
                            "materials": ["polyester", "spandex"]
                        }
                    }
                ],
                "summary": {
                    "total_detections": 1,
                    "categories_found": ["dress"]
                }
            },
            "user_tags": ["elegant", "special occasions", "favorite"],
            "brand": "Zara",
            "size": "S",
            "color_primary": "black",
            "season": ["summer", "spring"],
            "occasion": ["formal", "party", "date"],
            "purchase_date": "2024-03-20",
            "last_worn": "2024-07-15"
        },
        {
            "id": str(uuid.uuid4()),
            "user_id": "user_123",
            "image_path": "/images/sneakers.jpg",
            "visual_analysis": {
                "detections": [
                    {
                        "label": "sneakers",
                        "confidence": 0.89,
                        "bbox": [20, 50, 160, 120],
                        "attributes": {
                            "colors": ["white", "black"],
                            "patterns": ["solid", "logo"],
                            "styles": ["athletic", "casual"],
                            "materials": ["leather", "rubber"]
                        }
                    }
                ],
                "summary": {
                    "total_detections": 1,
                    "categories_found": ["sneakers"]
                }
            },
            "user_tags": ["comfortable", "sporty", "everyday"],
            "brand": "Nike",
            "size": "9",
            "color_primary": "white",
            "season": ["spring", "summer", "fall"],
            "occasion": ["casual", "sport", "travel"],
            "purchase_date": "2024-01-05",
            "last_worn": "2024-07-30"
        },
        {
            "id": str(uuid.uuid4()),
            "user_id": "user_123",
            "image_path": "/images/blazer.jpg",
            "visual_analysis": {
                "detections": [
                    {
                        "label": "jacket",
                        "confidence": 0.94,
                        "bbox": [8, 12, 185, 280],
                        "attributes": {
                            "colors": ["navy", "blue"],
                            "patterns": ["solid"],
                            "styles": ["blazer", "tailored"],
                            "materials": ["wool", "polyester"]
                        }
                    }
                ],
                "summary": {
                    "total_detections": 1,
                    "categories_found": ["jacket"]
                }
            },
            "user_tags": ["professional", "versatile", "investment piece"],
            "brand": "Hugo Boss",
            "size": "M",
            "color_primary": "navy",
            "season": ["fall", "winter", "spring"],
            "occasion": ["business", "formal", "smart casual"],
            "purchase_date": "2024-04-12",
            "last_worn": "2024-07-28"
        }
    ]
    
    return sample_items


def process_wardrobe_data(data: List[Dict[str, Any]], config: VectorStoreConfig) -> List[Dict[str, Any]]:
    """
    A4c. Ham gardırop verisini işle ve filtrele
    
    Args:
        data: Ham gardırop verisi
        config: Vector store konfigürasyonu
        
    Returns:
        List[Dict[str, Any]]: İşlenmiş ve filtrelenmiş veri
    """
    logger.info("🧹 Gardırop verisi işleniyor...")
    
    processed_items = []
    
    for item in data:
        try:
            # ID kontrolü
            if 'id' not in item:
                item['id'] = str(uuid.uuid4())
            
            # Visual analysis kontrolü
            if 'visual_analysis' not in item or not item['visual_analysis']:
                logger.warning(f"⚠️ Visual analysis verisi eksik: {item.get('id', 'unknown')}")
                continue
            
            detections = item['visual_analysis'].get('detections', [])
            if not detections:
                logger.warning(f"⚠️ Detection verisi eksik: {item.get('id', 'unknown')}")
                continue
            
            # Confidence filtreleme
            valid_detections = [
                det for det in detections 
                if det.get('confidence', 0) >= config.min_confidence
            ]
            
            if not valid_detections:
                logger.debug(f"⚠️ Düşük confidence, filtrelendi: {item.get('id', 'unknown')}")
                continue
            
            # Kategori filtreleme
            item_categories = [det.get('label', '').lower() for det in valid_detections]
            supported_found = any(
                cat in config.supported_categories 
                for cat in item_categories
            )
            
            if not supported_found:
                logger.debug(f"⚠️ Desteklenmeyen kategori, filtrelendi: {item.get('id', 'unknown')}")
                continue
            
            # Veriyi zenginleştir
            item['processed_detections'] = valid_detections
            item['processed_categories'] = item_categories
            
            processed_items.append(item)
            
        except Exception as e:
            logger.warning(f"⚠️ Item işleme hatası: {str(e)} - {item.get('id', 'unknown')}")
            continue
    
    logger.info(f"🧹 Veri işleme tamamlandı:")
    logger.info(f"  - Ham itemlar: {len(data)}")
    logger.info(f"  - İşlenmiş itemlar: {len(processed_items)}")
    logger.info(f"  - Filtrelenen itemlar: {len(data) - len(processed_items)}")
    
    return processed_items


# =============================================================================
# A5. VEKTÖR VERİTABANINI (FAISS VEYA PINECONE) BAŞLATMA
# =============================================================================

def initialize_vector_store(config: VectorStoreConfig) -> Union[faiss.Index, Any]:
    """
    A5a. Vector store'u başlat (FAISS veya Pinecone)
    
    Args:
        config: Vector store konfigürasyonu
        
    Returns:
        Union[faiss.Index, Any]: İnitialize edilmiş vector store
    """
    logger.info(f"🔧 Vector store başlatılıyor: {config.vector_store_type}")
    
    if config.vector_store_type == "faiss":
        return initialize_faiss_store(config)
    elif config.vector_store_type == "pinecone":
        return initialize_pinecone_store(config)
    else:
        raise ValueError(f"Desteklenmeyen vector store tipi: {config.vector_store_type}")


def initialize_faiss_store(config: VectorStoreConfig) -> faiss.Index:
    """
    A5b. FAISS vector store'unu başlat
    
    Args:
        config: Vector store konfigürasyonu
        
    Returns:
        faiss.Index: FAISS indeksi
    """
    logger.info("🔧 FAISS vector store başlatılıyor...")
    
    try:
        # FAISS indeks tipine göre indeks oluştur
        if config.faiss_index_type == "IndexFlatIP":
            # Inner Product (cosine similarity için normalize gerekli)
            base_index = faiss.IndexFlatIP(config.embedding_dimension)
        elif config.faiss_index_type == "IndexFlatL2":
            # L2 distance
            base_index = faiss.IndexFlatL2(config.embedding_dimension)
        elif config.faiss_index_type == "IndexIVFFlat":
            # IVF (daha büyük veri setleri için)
            quantizer = faiss.IndexFlatL2(config.embedding_dimension)
            base_index = faiss.IndexIVFFlat(quantizer, config.embedding_dimension, 100)
        else:
            raise ValueError(f"Desteklenmeyen FAISS indeks tipi: {config.faiss_index_type}")
        
        # ID mapping için IndexIDMap kullan
        index = faiss.IndexIDMap(base_index)
        
        logger.success(f"✅ FAISS indeksi oluşturuldu:")
        logger.info(f"  - Index type: {config.faiss_index_type}")
        logger.info(f"  - Dimension: {config.embedding_dimension}")
        logger.info(f"  - Is trained: {index.is_trained}")
        
        return index
        
    except Exception as e:
        logger.error(f"❌ FAISS indeks oluşturma hatası: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


def initialize_pinecone_store(config: VectorStoreConfig) -> Any:
    """
    A5c. Pinecone vector store'unu başlat
    
    Args:
        config: Vector store konfigürasyonu
        
    Returns:
        Any: Pinecone indeksi
    """
    if not PINECONE_AVAILABLE:
        raise ImportError("Pinecone paketi kurulu değil. pip install pinecone-client komutu ile kurun.")
    
    logger.info("🔧 Pinecone vector store başlatılıyor...")
    
    try:
        # API key'i al
        api_key = config.pinecone_api_key or os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("Pinecone API key bulunamadı. --pinecone_api_key argümanı veya PINECONE_API_KEY environment variable ayarlayın.")
        
        # Pinecone'u başlat
        pinecone.init(
            api_key=api_key,
            environment=config.pinecone_environment
        )
        
        # İndeks listesini kontrol et
        existing_indexes = pinecone.list_indexes()
        
        if config.pinecone_index_name not in existing_indexes:
            # Yeni indeks oluştur
            logger.info(f"🔧 Pinecone indeksi oluşturuluyor: {config.pinecone_index_name}")
            
            pinecone.create_index(
                name=config.pinecone_index_name,
                dimension=config.embedding_dimension,
                metric=config.pinecone_metric
            )
            
            # İndeksin hazır olmasını bekle
            import time
            time.sleep(10)
            
            logger.success(f"✅ Pinecone indeksi oluşturuldu: {config.pinecone_index_name}")
        else:
            logger.info(f"📂 Mevcut Pinecone indeksi kullanılıyor: {config.pinecone_index_name}")
        
        # İndekse bağlan
        index = pinecone.Index(config.pinecone_index_name)
        
        # İndeks bilgilerini göster
        stats = index.describe_index_stats()
        logger.info(f"📊 Pinecone indeks bilgileri:")
        logger.info(f"  - Index name: {config.pinecone_index_name}")
        logger.info(f"  - Dimension: {config.embedding_dimension}")
        logger.info(f"  - Metric: {config.pinecone_metric}")
        logger.info(f"  - Total vectors: {stats.get('total_vector_count', 0)}")
        
        return index
        
    except Exception as e:
        logger.error(f"❌ Pinecone indeks oluşturma hatası: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


# =============================================================================
# A6. VERİLERİ EMBEDDING'LEME VE VERİTABANINA EKLEME
# =============================================================================

def generate_embeddings_and_add_to_store(
    items: List[Dict[str, Any]],
    embedding_model: SentenceTransformer,
    vector_store: Union[faiss.Index, Any],
    config: VectorStoreConfig
) -> None:
    """
    A6a. Gardırop itemlarını embedding'le ve vector store'a ekle
    
    Args:
        items: İşlenmiş gardırop itemları
        embedding_model: Sentence transformer modeli
        vector_store: Vector store (FAISS veya Pinecone)
        config: Vector store konfigürasyonu
    """
    logger.info(f"🔄 {len(items)} item için embedding'ler oluşturuluyor...")
    
    try:
        # Metinleri hazırla
        texts = []
        ids = []
        metadata_list = []
        
        for item in items:
            # Item için açıklayıcı metin oluştur
            description = create_item_description(item)
            texts.append(description)
            ids.append(item['id'])
            
            # Metadata hazırla
            if config.include_metadata:
                metadata = create_item_metadata(item)
                metadata_list.append(metadata)
        
        logger.info(f"📝 {len(texts)} metin açıklaması hazırlandı")
        
        # Batch'ler halinde embedding'leri oluştur
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), config.batch_size), desc="Embedding generation"):
            batch_texts = texts[i:i + config.batch_size]
            
            # Embeddings oluştur
            batch_embeddings = embedding_model.encode(
                batch_texts,
                convert_to_numpy=True,
                normalize_embeddings=config.normalize_embeddings,
                show_progress_bar=False
            )
            
            all_embeddings.append(batch_embeddings)
        
        # Tüm embedding'leri birleştir
        embeddings = np.vstack(all_embeddings)
        logger.success(f"✅ Embeddings oluşturuldu: {embeddings.shape}")
        
        # Vector store'a ekle
        if config.vector_store_type == "faiss":
            add_to_faiss_store(vector_store, embeddings, ids, metadata_list, config)
        elif config.vector_store_type == "pinecone":
            add_to_pinecone_store(vector_store, embeddings, ids, metadata_list, config)
        
        logger.success(f"✅ {len(items)} item vector store'a eklendi")
        
    except Exception as e:
        logger.error(f"❌ Embedding ve ekleme hatası: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


def create_item_description(item: Dict[str, Any]) -> str:
    """
    A6b. Gardırop item'ı için açıklayıcı metin oluştur
    
    Args:
        item: Gardırop item'ı
        
    Returns:
        str: Açıklayıcı metin
    """
    description_parts = []
    
    # Visual analysis detayları
    detections = item.get('processed_detections', [])
    for detection in detections:
        label = detection.get('label', '')
        attributes = detection.get('attributes', {})
        
        # Temel item açıklaması
        if label:
            description_parts.append(f"This is a {label}")
        
        # Renk bilgisi
        colors = attributes.get('colors', [])
        if colors:
            color_text = ", ".join(colors)
            description_parts.append(f"in {color_text} color")
        
        # Stil bilgisi
        styles = attributes.get('styles', [])
        if styles:
            style_text = ", ".join(styles)
            description_parts.append(f"with {style_text} style")
        
        # Malzeme bilgisi
        materials = attributes.get('materials', [])
        if materials:
            material_text = ", ".join(materials)
            description_parts.append(f"made of {material_text}")
        
        # Desen bilgisi
        patterns = attributes.get('patterns', [])
        if patterns and patterns != ['solid']:
            pattern_text = ", ".join(patterns)
            description_parts.append(f"featuring {pattern_text} pattern")
    
    # Kullanıcı etiketleri
    user_tags = item.get('user_tags', [])
    if user_tags:
        tags_text = ", ".join(user_tags)
        description_parts.append(f"Described as {tags_text}")
    
    # Marka bilgisi
    brand = item.get('brand')
    if brand:
        description_parts.append(f"Brand: {brand}")
    
    # Mevsim bilgisi
    seasons = item.get('season', [])
    if seasons:
        season_text = ", ".join(seasons)
        description_parts.append(f"Suitable for {season_text}")
    
    # Etkinlik bilgisi
    occasions = item.get('occasion', [])
    if occasions:
        occasion_text = ", ".join(occasions)
        description_parts.append(f"Perfect for {occasion_text} occasions")
    
    # Metni birleştir
    description = ". ".join(description_parts) + "."
    
    return description


def create_item_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    A6c. Item için metadata oluştur
    
    Args:
        item: Gardırop item'ı
        
    Returns:
        Dict[str, Any]: Metadata
    """
    metadata = {
        "id": item.get('id'),
        "user_id": item.get('user_id'),
        "image_path": item.get('image_path'),
        "brand": item.get('brand'),
        "size": item.get('size'),
        "color_primary": item.get('color_primary'),
        "purchase_date": item.get('purchase_date'),
        "last_worn": item.get('last_worn'),
    }
    
    # Kategoriler
    categories = item.get('processed_categories', [])
    if categories:
        metadata['categories'] = categories
    
    # Kullanıcı etiketleri
    user_tags = item.get('user_tags', [])
    if user_tags:
        metadata['user_tags'] = user_tags
    
    # Mevsim ve etkinlik
    seasons = item.get('season', [])
    if seasons:
        metadata['seasons'] = seasons
    
    occasions = item.get('occasion', [])
    if occasions:
        metadata['occasions'] = occasions
    
    # Visual analysis özeti
    if 'visual_analysis' in item:
        va_summary = item['visual_analysis'].get('summary', {})
        metadata['total_detections'] = va_summary.get('total_detections', 0)
        metadata['categories_found'] = va_summary.get('categories_found', [])
    
    # None değerleri temizle
    metadata = {k: v for k, v in metadata.items() if v is not None}
    
    return metadata


def add_to_faiss_store(
    index: faiss.Index,
    embeddings: np.ndarray,
    ids: List[str],
    metadata_list: List[Dict[str, Any]],
    config: VectorStoreConfig
) -> None:
    """
    A6d. FAISS store'a embedding'leri ekle
    
    Args:
        index: FAISS indeksi
        embeddings: Embedding vektörleri
        ids: Item ID'leri
        metadata_list: Metadata listesi
        config: Vector store konfigürasyonu
    """
    logger.info("💾 FAISS store'a vektörler ekleniyor...")
    
    try:
        # ID'leri integer'a dönüştür (FAISS requirement)
        # UUID'leri hash'le
        int_ids = np.array([hash(id_str) % (2**63 - 1) for id_str in ids], dtype=np.int64)
        
        # Vektörleri ekle
        index.add_with_ids(embeddings.astype(np.float32), int_ids)
        
        logger.success(f"✅ FAISS'e {len(embeddings)} vektör eklendi")
        logger.info(f"📊 FAISS indeks bilgileri:")
        logger.info(f"  - Total vectors: {index.ntotal}")
        logger.info(f"  - Is trained: {index.is_trained}")
        
    except Exception as e:
        logger.error(f"❌ FAISS'e ekleme hatası: {str(e)}")
        raise


def add_to_pinecone_store(
    index: Any,
    embeddings: np.ndarray,
    ids: List[str],
    metadata_list: List[Dict[str, Any]],
    config: VectorStoreConfig
) -> None:
    """
    A6e. Pinecone store'a embedding'leri ekle
    
    Args:
        index: Pinecone indeksi
        embeddings: Embedding vektörleri
        ids: Item ID'leri
        metadata_list: Metadata listesi
        config: Vector store konfigürasyonu
    """
    logger.info("💾 Pinecone store'a vektörler ekleniyor...")
    
    try:
        # Pinecone format'ına dönüştür
        vectors_to_upsert = []
        
        for i, (id_str, embedding) in enumerate(zip(ids, embeddings)):
            vector_data = {
                "id": id_str,
                "values": embedding.tolist()
            }
            
            # Metadata ekle
            if config.include_metadata and i < len(metadata_list):
                # Pinecone metadata constraints'leri dikkate al
                metadata = {}
                for key, value in metadata_list[i].items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value
                    elif isinstance(value, list):
                        # List'leri string'e dönüştür
                        metadata[key] = ", ".join(map(str, value))
                
                vector_data["metadata"] = metadata
            
            vectors_to_upsert.append(vector_data)
        
        # Batch'ler halinde upsert
        batch_size = 100  # Pinecone limit
        
        for i in tqdm(range(0, len(vectors_to_upsert), batch_size), desc="Pinecone upsert"):
            batch = vectors_to_upsert[i:i + batch_size]
            index.upsert(vectors=batch)
        
        logger.success(f"✅ Pinecone'e {len(embeddings)} vektör eklendi")
        
        # İndeks istatistikleri
        stats = index.describe_index_stats()
        logger.info(f"📊 Pinecone indeks bilgileri:")
        logger.info(f"  - Total vectors: {stats.get('total_vector_count', 0)}")
        
    except Exception as e:
        logger.error(f"❌ Pinecone'e ekleme hatası: {str(e)}")
        raise


# =============================================================================
# A7. VEKTÖR VERİTABANINI KAYDETME
# =============================================================================

def save_vector_store(
    vector_store: Union[faiss.Index, Any],
    config: VectorStoreConfig,
    metadata: Dict[str, Any]
) -> None:
    """
    A7a. Vector store'u kaydet
    
    Args:
        vector_store: Vector store (FAISS veya Pinecone)
        config: Vector store konfigürasyonu
        metadata: Kaydedilecek metadata
    """
    logger.info("💾 Vector store kaydediliyor...")
    
    if config.vector_store_type == "faiss":
        save_faiss_store(vector_store, config, metadata)
    elif config.vector_store_type == "pinecone":
        save_pinecone_config(config, metadata)
    
    logger.success("✅ Vector store başarıyla kaydedildi")


def save_faiss_store(
    index: faiss.Index,
    config: VectorStoreConfig,
    metadata: Dict[str, Any]
) -> None:
    """
    A7b. FAISS store'u dosyaya kaydet
    
    Args:
        index: FAISS indeksi
        config: Vector store konfigürasyonu
        metadata: Kaydedilecek metadata
    """
    try:
        # Output directory oluştur
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # FAISS indeksini kaydet
        index_path = output_dir / "faiss_index.index"
        faiss.write_index(index, str(index_path))
        
        logger.success(f"✅ FAISS indeksi kaydedildi: {index_path}")
        
        # Metadata'yı kaydet
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        logger.success(f"✅ Metadata kaydedildi: {metadata_path}")
        
        # Configuration'ı kaydet
        config_path = output_dir / "vector_store_config.json"
        config_dict = asdict(config)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False, default=str)
        
        logger.success(f"✅ Configuration kaydedildi: {config_path}")
        
        # Kullanım kılavuzu oluştur
        readme_path = output_dir / "README.md"
        create_usage_guide(readme_path, config, metadata)
        
        logger.info(f"📁 Vector store dosyaları:")
        for file_path in output_dir.iterdir():
            if file_path.is_file():
                file_size = file_path.stat().st_size / 1024**2  # MB
                logger.info(f"  - {file_path.name}: {file_size:.2f} MB")
        
    except Exception as e:
        logger.error(f"❌ FAISS kaydetme hatası: {str(e)}")
        raise


def save_pinecone_config(config: VectorStoreConfig, metadata: Dict[str, Any]) -> None:
    """
    A7c. Pinecone konfigürasyonunu kaydet
    
    Args:
        config: Vector store konfigürasyonu
        metadata: Kaydedilecek metadata
    """
    try:
        # Output directory oluştur
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Pinecone connection info
        pinecone_info = {
            "index_name": config.pinecone_index_name,
            "environment": config.pinecone_environment,
            "metric": config.pinecone_metric,
            "dimension": config.embedding_dimension,
            "created_at": datetime.now().isoformat()
        }
        
        pinecone_path = output_dir / "pinecone_config.json"
        with open(pinecone_path, 'w', encoding='utf-8') as f:
            json.dump(pinecone_info, f, indent=2, ensure_ascii=False)
        
        logger.success(f"✅ Pinecone config kaydedildi: {pinecone_path}")
        
        # Metadata'yı kaydet
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        logger.success(f"✅ Metadata kaydedildi: {metadata_path}")
        
        # Kullanım kılavuzu oluştur
        readme_path = output_dir / "README.md"
        create_usage_guide(readme_path, config, metadata)
        
    except Exception as e:
        logger.error(f"❌ Pinecone config kaydetme hatası: {str(e)}")
        raise


def create_usage_guide(readme_path: Path, config: VectorStoreConfig, metadata: Dict[str, Any]) -> None:
    """
    A7d. Vector store kullanım kılavuzu oluştur
    
    Args:
        readme_path: README dosya yolu
        config: Vector store konfigürasyonu
        metadata: Vector store metadata'sı
    """
    guide_content = f"""# Aura Fashion Assistant - Vector Store

Bu vector store, Aura moda asistanının RAG sistemi için oluşturulmuştur.

## Vector Store Bilgileri

- **Type**: {config.vector_store_type}
- **Embedding Model**: {config.embedding_model_name}
- **Dimension**: {config.embedding_dimension}
- **Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Items**: {metadata.get('total_items', 'N/A')}

## Dosyalar

### FAISS Store (eğer kullanılıyorsa)
- `faiss_index.index`: FAISS indeks dosyası
- `metadata.json`: Item metadata'ları
- `vector_store_config.json`: Vector store konfigürasyonu

### Pinecone Store (eğer kullanılıyorsa)
- `pinecone_config.json`: Pinecone bağlantı bilgileri
- `metadata.json`: Item metadata'ları

## Kullanım

### FAISS Store Yükleme

```python
import faiss
import json

# İndeksi yükle
index = faiss.read_index("faiss_index.index")

# Metadata'yı yükle
with open("metadata.json", "r") as f:
    metadata = json.load(f)

# Arama yap
query_vector = ...  # Embedding vektörü
k = 5  # Döndürülecek sonuç sayısı
distances, indices = index.search(query_vector, k)
```

### Pinecone Store Kullanma

```python
import pinecone
import json

# Config'i yükle
with open("pinecone_config.json", "r") as f:
    config = json.load(f)

# Pinecone'a bağlan
pinecone.init(api_key="your-api-key", environment=config["environment"])
index = pinecone.Index(config["index_name"])

# Arama yap
query_vector = ...  # Embedding vektörü
results = index.query(vector=query_vector, top_k=5, include_metadata=True)
```

## Metadata Yapısı

Her gardırop item'ı için aşağıdaki metadata saklanır:

```json
{{
    "id": "unique-item-id",
    "user_id": "user-123",
    "categories": ["shirt", "casual"],
    "brand": "Brand Name",
    "color_primary": "blue",
    "seasons": ["spring", "fall"],
    "occasions": ["casual", "work"],
    "user_tags": ["favorite", "comfortable"]
}}
```

## Yeniden Oluşturma

Vector store'u yeniden oluşturmak için:

```bash
python src/build_vector_store.py \\
    --input_data_path ./data/user_wardrobe.json \\
    --vector_store_type {config.vector_store_type} \\
    --embedding_model_name {config.embedding_model_name} \\
    --output_dir {config.output_dir}
```

---
Generated by Aura Vector Store Builder v1.0.0
"""
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    logger.success(f"✅ Kullanım kılavuzu oluşturuldu: {readme_path}")


# =============================================================================
# MAIN FUNCTION VE SCRIPT RUNNER
# =============================================================================

def main():
    """
    Ana vector store oluşturma pipeline'ı
    
    Bu fonksiyon tüm adımları sırasıyla çalıştırır:
    1. Arguments parsing ve configuration
    2. Embedding modeli yükleme
    3. Gardırop verisini yükleme
    4. Vector store başlatma
    5. Embedding'leme ve ekleme
    6. Kaydetme
    """
    try:
        # A2. Argümanları parse et ve configuration oluştur
        args = parse_arguments()
        setup_logging(debug=args.debug)
        
        logger.info("🎯 Aura Fashion Assistant Vector Store Builder")
        logger.info("=" * 60)
        
        # Configuration objesi oluştur
        config = VectorStoreConfig(
            input_data_path=args.input_data_path,
            output_dir=args.output_dir,
            embedding_model_name=args.embedding_model_name,
            max_seq_length=args.max_seq_length,
            vector_store_type=args.vector_store_type,
            faiss_index_type=args.faiss_index_type,
            pinecone_api_key=args.pinecone_api_key,
            pinecone_environment=args.pinecone_environment,
            pinecone_index_name=args.pinecone_index_name,
            batch_size=args.batch_size,
            min_confidence=args.min_confidence,
            normalize_embeddings=args.normalize_embeddings,
            include_metadata=args.include_metadata,
        )
        
        logger.info("📋 Configuration:")
        logger.info(f"  - Input data: {config.input_data_path}")
        logger.info(f"  - Output dir: {config.output_dir}")
        logger.info(f"  - Embedding model: {config.embedding_model_name}")
        logger.info(f"  - Vector store: {config.vector_store_type}")
        logger.info(f"  - Batch size: {config.batch_size}")
        
        # System info
        logger.info("🖥️ System Information:")
        logger.info(f"  - PyTorch version: {torch.__version__}")
        logger.info(f"  - CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"  - GPU name: {torch.cuda.get_device_name(0)}")
        
        # A3. Embedding modeli yükleme
        logger.info("\n" + "="*60)
        logger.info("ADIM 1: EMBEDDING MODELİ YÜKLEME")
        logger.info("="*60)
        
        embedding_model = load_embedding_model(config)
        
        # A4. Gardırop verisini yükleme
        logger.info("\n" + "="*60)
        logger.info("ADIM 2: GARDıROP VERISINİ YÜKLEME")
        logger.info("="*60)
        
        wardrobe_items = load_wardrobe_data(config)
        
        if not wardrobe_items:
            logger.error("❌ İşlenecek gardırop verisi bulunamadı")
            return
        
        # A5. Vector store başlatma
        logger.info("\n" + "="*60)
        logger.info("ADIM 3: VECTOR STORE BAŞLATMA")
        logger.info("="*60)
        
        vector_store = initialize_vector_store(config)
        
        # A6. Embedding'leme ve ekleme
        logger.info("\n" + "="*60)
        logger.info("ADIM 4: EMBEDDING'LEME VE EKLEME")
        logger.info("="*60)
        
        generate_embeddings_and_add_to_store(
            wardrobe_items,
            embedding_model,
            vector_store,
            config
        )
        
        # A7. Kaydetme
        logger.info("\n" + "="*60)
        logger.info("ADIM 5: VECTOR STORE KAYDETME")
        logger.info("="*60)
        
        # Metadata oluştur
        metadata = {
            "created_at": datetime.now().isoformat(),
            "total_items": len(wardrobe_items),
            "embedding_model": config.embedding_model_name,
            "embedding_dimension": config.embedding_dimension,
            "vector_store_type": config.vector_store_type,
            "min_confidence": config.min_confidence,
            "categories_processed": list(set(
                cat for item in wardrobe_items 
                for cat in item.get('processed_categories', [])
            )),
            "config": asdict(config)
        }
        
        save_vector_store(vector_store, config, metadata)
        
        # Final
        logger.info("\n" + "="*60)
        logger.success("🎉 VECTOR STORE OLUŞTURMA TAMAMLANDI!")
        logger.info("="*60)
        
        output_path = Path(config.output_dir)
        logger.info(f"📁 Vector store lokasyonu: {output_path}")
        logger.info(f"📊 İşlenen itemlar: {len(wardrobe_items)}")
        logger.info(f"🔍 Vector store türü: {config.vector_store_type}")
        logger.info("📖 Kullanım için README.md dosyasını inceleyin")
        logger.info("🚀 Vector store artık RAG sisteminde kullanıma hazır!")
        
        # GPU memory temizleme
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 GPU memory temizlendi")
        
    except KeyboardInterrupt:
        logger.warning("⚠️ Vector store oluşturma kullanıcı tarafından durduruldu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ Pipeline hatası: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Cleanup on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sys.exit(1)


if __name__ == "__main__":
    """
    Script doğrudan çalıştırıldığında main fonksiyonunu çağır
    
    Kullanım örnekleri:
    
    # FAISS ile temel kullanım
    python src/build_vector_store.py \\
        --input_data_path ./data/user_wardrobe.json \\
        --vector_store_type faiss \\
        --output_dir ./vector_stores
    
    # Pinecone ile kullanım
    python src/build_vector_store.py \\
        --input_data_path ./data/user_wardrobe.json \\
        --vector_store_type pinecone \\
        --pinecone_api_key your-api-key \\
        --pinecone_index_name aura-wardrobe
    
    # Özel embedding modeli ile
    python src/build_vector_store.py \\
        --embedding_model_name sentence-transformers/all-mpnet-base-v2 \\
        --batch_size 16 \\
        --min_confidence 0.7
    """
    main()
