"""
Polyvore Dataset Loader for OutfitTransformer Model Training - Aura Project
Bu modül, Polyvore Outfits veri setini yüklemek ve OutfitTransformer modeli için ön işleme yapmaktan sorumludur.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from pathlib import Path
import pickle
import random

# Core libraries
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset, WeightedRandomSampler
import torchvision.transforms as transforms

# PIL for image processing
from PIL import Image, ImageDraw

# Hugging Face libraries
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoImageProcessor

# Numerical computation
import numpy as np
import pandas as pd

# Graph processing for outfit compatibility
import networkx as nx

# Text processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# JSON processing
import jsonlines

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Polyvore dataset constants
POLYVORE_DATASET_NAME = "polyvore_outfits"  # Local or custom dataset name
IMAGE_SIZE = 224  # Standard fashion image size for transformers
MAX_ITEMS_PER_OUTFIT = 10  # Maximum number of items in an outfit

# OutfitTransformer specific constants
OUTFIT_TRANSFORMER_IMAGE_SIZE = (224, 224)  # Input image size
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization mean
IMAGENET_STD = [0.229, 0.224, 0.225]   # ImageNet normalization std

# Valid dataset splits for Polyvore
VALID_SPLITS = ["train", "validation", "test"]

# Fashion category hierarchies for OutfitTransformer
FASHION_CATEGORY_HIERARCHY = {
    "tops": ["shirt", "blouse", "t-shirt", "tank", "sweater", "hoodie", "jacket", "blazer", "coat"],
    "bottoms": ["jeans", "pants", "shorts", "skirt", "leggings", "dress"],
    "shoes": ["sneakers", "boots", "sandals", "heels", "flats", "loafers"],
    "accessories": ["bag", "purse", "backpack", "hat", "scarf", "belt", "jewelry", "watch", "sunglasses"],
    "outerwear": ["jacket", "coat", "blazer", "cardigan", "vest"]
}

# Outfit compatibility rules (basic rules for fashion matching)
COMPATIBILITY_RULES = {
    "formal": {
        "tops": ["blouse", "shirt", "blazer"],
        "bottoms": ["pants", "skirt", "dress"],
        "shoes": ["heels", "loafers", "flats"],
        "colors": ["black", "white", "gray", "navy", "beige"]
    },
    "casual": {
        "tops": ["t-shirt", "tank", "sweater", "hoodie"],
        "bottoms": ["jeans", "shorts", "leggings"],
        "shoes": ["sneakers", "boots", "sandals"],
        "colors": ["any"]
    },
    "business": {
        "tops": ["shirt", "blouse", "blazer"],
        "bottoms": ["pants", "skirt", "dress"],
        "shoes": ["heels", "loafers"],
        "colors": ["black", "white", "gray", "navy"]
    }
}


def load_polyvore_dataset(data_dir: str, split: str = "train") -> Union[Dataset, Dict[str, Any]]:
    """
    A2. Polyvore Outfits veri setini yükler
    
    Bu fonksiyon, OutfitTransformer model eğitimi için Polyvore veri setinin 
    belirtilen bölümünü (train/validation/test) yükler.
    
    Args:
        data_dir (str): Polyvore veri setinin bulunduğu dizin yolu
        split (str): Yüklenecek veri seti bölümü ('train', 'validation', 'test')
        
    Returns:
        Union[Dataset, Dict[str, Any]]: Yüklenen Polyvore veri seti
        
    Raises:
        ValueError: Geçersiz split değeri verildiğinde
        FileNotFoundError: Veri dosyaları bulunamadığında
        
    Example:
        >>> dataset = load_polyvore_dataset("/path/to/polyvore", split="train")
        >>> print(f"Loaded {len(dataset['outfits'])} outfits")
    """
    # Split doğrulama
    if split not in VALID_SPLITS:
        raise ValueError(
            f"Geçersiz split '{split}'. Geçerli değerler: {VALID_SPLITS}"
        )
    
    logger.info(f"Polyvore veri seti yükleniyor - split: {split}")
    
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Veri dizini bulunamadı: {data_dir}")
    
    try:
        # Polyvore JSON dosyalarını yükle
        outfits_file = data_path / f"{split}_outfits.json"
        items_file = data_path / f"{split}_items.json"
        
        if not outfits_file.exists():
            raise FileNotFoundError(f"Outfit dosyası bulunamadı: {outfits_file}")
        
        if not items_file.exists():
            raise FileNotFoundError(f"Items dosyası bulunamadı: {items_file}")
        
        # JSON dosyalarını yükle
        logger.info(f"Outfits yükleniyor: {outfits_file}")
        with open(outfits_file, 'r', encoding='utf-8') as f:
            outfits_data = json.load(f)
        
        logger.info(f"Items yükleniyor: {items_file}")
        with open(items_file, 'r', encoding='utf-8') as f:
            items_data = json.load(f)
        
        # Items'ı ID'ye göre indexle
        items_by_id = {item['item_id']: item for item in items_data}
        
        # Dataset yapısını oluştur
        dataset = {
            "outfits": outfits_data,
            "items": items_by_id,
            "split": split,
            "num_outfits": len(outfits_data),
            "num_items": len(items_data)
        }
        
        logger.info(f"Başarıyla yüklendi: {dataset['num_outfits']} outfit, {dataset['num_items']} item")
        return dataset
        
    except Exception as e:
        error_msg = f"Polyvore veri seti yüklenemedi: {str(e)}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg) from e


def preprocess_outfit_image(image: Union[Image.Image, str, np.ndarray], 
                           processor: Optional[AutoImageProcessor] = None,
                           target_size: Tuple[int, int] = OUTFIT_TRANSFORMER_IMAGE_SIZE) -> torch.Tensor:
    """
    A3. OutfitTransformer modeli için görüntü ön işleme fonksiyonu
    
    Bu fonksiyon, girdi görüntüsünü OutfitTransformer modelinin beklediği formata getirir.
    
    Args:
        image (Union[Image.Image, str, np.ndarray]): İşlenecek görüntü
        processor (Optional[AutoImageProcessor]): Vision processor
        target_size (Tuple[int, int]): Hedef görüntü boyutu
        
    Returns:
        torch.Tensor: İşlenmiş görüntü tensörü [C, H, W] formatında
        
    Raises:
        ValueError: Görüntü None veya geçersiz ise
        TypeError: Desteklenmeyen görüntü tipi
        OSError: Görüntü dosyası bulunamadığında
        
    Example:
        >>> from PIL import Image
        >>> img = Image.open("fashion_item.jpg")
        >>> processed = preprocess_outfit_image(img)
        >>> print(f"Processed shape: {processed.shape}")  # [3, 224, 224]
    """
    # Girdi doğrulama
    if image is None:
        raise ValueError("Görüntü None olamaz")
    
    # PIL Image'a dönüştürme
    pil_image = _convert_to_pil(image)
    
    # Processor kullan (eğer verilmişse)
    if processor is not None:
        try:
            processed = processor(images=pil_image, return_tensors="pt")
            return processed["pixel_values"].squeeze(0)  # [1, 3, H, W] -> [3, H, W]
        except Exception as e:
            logger.warning(f"Processor kullanılamadı, manual preprocessing yapılacak: {e}")
    
    # Manual preprocessing
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    try:
        processed_tensor = transform(pil_image)
        logger.debug(f"Görüntü işlendi: {processed_tensor.shape}")
        return processed_tensor
        
    except Exception as e:
        error_msg = f"Görüntü işleme hatası: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def create_outfit_compatibility_graph(outfits: List[Dict[str, Any]], 
                                     items: Dict[str, Dict[str, Any]]) -> nx.Graph:
    """
    A4. Outfit uyumluluk grafiği oluşturur
    
    Bu fonksiyon, outfitler arasındaki item uyumluluğunu temsil eden bir graf oluşturur.
    Bu graf, OutfitTransformer'ın öğrenmesi için uyumluluk bilgisi sağlar.
    
    Args:
        outfits (List[Dict[str, Any]]): Outfit listesi
        items (Dict[str, Dict[str, Any]]): Item ID'den item bilgisine mapping
        
    Returns:
        nx.Graph: Outfit uyumluluk grafiği
        
    Example:
        >>> graph = create_outfit_compatibility_graph(outfits, items)
        >>> print(f"Graph has {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    """
    logger.info("Outfit uyumluluk grafiği oluşturuluyor...")
    
    graph = nx.Graph()
    
    # Her outfit için item'lar arasında edge oluştur
    for outfit in outfits:
        outfit_items = outfit.get('items', [])
        
        # Outfit içindeki her item çifti arasında uyumluluk edge'i ekle
        for i, item_id_1 in enumerate(outfit_items):
            if item_id_1 not in items:
                continue
                
            item_1 = items[item_id_1]
            
            # Tek item için node ekle
            if not graph.has_node(item_id_1):
                graph.add_node(item_id_1, **item_1)
            
            for j, item_id_2 in enumerate(outfit_items[i+1:], i+1):
                if item_id_2 not in items:
                    continue
                    
                item_2 = items[item_id_2]
                
                # İkinci item için node ekle
                if not graph.has_node(item_id_2):
                    graph.add_node(item_id_2, **item_2)
                
                # Uyumluluk edge'i ekle
                if graph.has_edge(item_id_1, item_id_2):
                    # Mevcut edge'in weight'ini artır
                    graph[item_id_1][item_id_2]['weight'] += 1
                else:
                    # Yeni edge ekle
                    compatibility_score = calculate_item_compatibility(item_1, item_2)
                    graph.add_edge(item_id_1, item_id_2, 
                                 weight=1, 
                                 compatibility=compatibility_score)
    
    logger.info(f"Graf oluşturuldu: {graph.number_of_nodes()} node, {graph.number_of_edges()} edge")
    return graph


def calculate_item_compatibility(item1: Dict[str, Any], item2: Dict[str, Any]) -> float:
    """
    İki fashion item arasındaki uyumluluğu hesaplar
    
    Args:
        item1, item2: Fashion item'ları
        
    Returns:
        float: Uyumluluk skoru (0-1)
    """
    compatibility_score = 0.5  # Base score
    
    # Kategori uyumluluğu
    cat1 = item1.get('category', '').lower()
    cat2 = item2.get('category', '').lower()
    
    # Aynı kategori ise uyumsuz (aynı tip giysi)
    if cat1 == cat2:
        compatibility_score -= 0.3
    
    # Farklı kategoriler ise uyumlu
    if _are_categories_compatible(cat1, cat2):
        compatibility_score += 0.3
    
    # Renk uyumluluğu
    color1 = item1.get('color', '').lower()
    color2 = item2.get('color', '').lower()
    
    if _are_colors_compatible(color1, color2):
        compatibility_score += 0.2
    
    # 0-1 arasında sınırla
    return max(0.0, min(1.0, compatibility_score))


def _are_categories_compatible(cat1: str, cat2: str) -> bool:
    """Kategorilerin uyumlu olup olmadığını kontrol eder"""
    # Temel uyumluluk kuralları
    compatible_pairs = [
        ("top", "bottom"), ("shirt", "pants"), ("blouse", "skirt"),
        ("dress", "shoes"), ("top", "shoes"), ("bottom", "shoes"),
        ("accessories", "top"), ("accessories", "bottom")
    ]
    
    for pair in compatible_pairs:
        if (cat1 in pair[0] and cat2 in pair[1]) or (cat1 in pair[1] and cat2 in pair[0]):
            return True
    
    return False


def _are_colors_compatible(color1: str, color2: str) -> bool:
    """Renklerin uyumlu olup olmadığını kontrol eder"""
    # Temel renk uyumluluk kuralları
    neutral_colors = {"black", "white", "gray", "beige", "navy"}
    
    # Nötr renkler herkesle uyumlu
    if color1 in neutral_colors or color2 in neutral_colors:
        return True
    
    # Monokromatik (aynı renk)
    if color1 == color2:
        return True
    
    # Diğer uyumluluklar (basitleştirilmiş)
    compatible_combinations = [
        ("blue", "white"), ("red", "black"), ("green", "brown"),
        ("pink", "gray"), ("yellow", "blue")
    ]
    
    for combo in compatible_combinations:
        if (color1 in combo and color2 in combo):
            return True
    
    return False


def extract_outfit_features(outfit: Dict[str, Any], 
                          items: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    A5. Outfit'ten feature'ları çıkarır
    
    Bu fonksiyon, bir outfit'ten OutfitTransformer'ın kullanabileceği 
    feature'ları (renk paleti, kategori dağılımı, stil vs.) çıkarır.
    
    Args:
        outfit (Dict[str, Any]): Outfit bilgileri
        items (Dict[str, Dict[str, Any]]): Item mapping
        
    Returns:
        Dict[str, Any]: Outfit feature'ları
        
    Example:
        >>> features = extract_outfit_features(outfit, items)
        >>> print(f"Outfit style: {features['style']}")
    """
    outfit_items = outfit.get('items', [])
    
    # Feature containers
    categories = []
    colors = []
    styles = []
    prices = []
    
    for item_id in outfit_items:
        if item_id in items:
            item = items[item_id]
            
            # Kategori
            if 'category' in item:
                categories.append(item['category'])
            
            # Renk
            if 'color' in item:
                colors.append(item['color'])
            
            # Stil
            if 'style' in item:
                styles.append(item['style'])
            
            # Fiyat
            if 'price' in item:
                try:
                    prices.append(float(item['price']))
                except (ValueError, TypeError):
                    pass
    
    # Feature özeti
    features = {
        "num_items": len(outfit_items),
        "categories": list(set(categories)),
        "dominant_colors": _get_dominant_elements(colors),
        "style_distribution": _get_style_distribution(styles),
        "price_range": {
            "min": min(prices) if prices else 0,
            "max": max(prices) if prices else 0,
            "avg": sum(prices) / len(prices) if prices else 0
        },
        "outfit_type": _infer_outfit_type(categories, styles),
        "season": outfit.get('season', 'unknown'),
        "occasion": outfit.get('occasion', 'unknown')
    }
    
    return features


def _get_dominant_elements(elements: List[str], top_k: int = 3) -> List[str]:
    """Liste içindeki en sık görülen elemanları döndürür"""
    if not elements:
        return []
    
    element_counts = {}
    for element in elements:
        element_counts[element] = element_counts.get(element, 0) + 1
    
    # En sık görülenleri sırala
    sorted_elements = sorted(element_counts.items(), key=lambda x: x[1], reverse=True)
    
    return [element for element, count in sorted_elements[:top_k]]


def _get_style_distribution(styles: List[str]) -> Dict[str, float]:
    """Stil dağılımını hesaplar"""
    if not styles:
        return {}
    
    style_counts = {}
    for style in styles:
        style_counts[style] = style_counts.get(style, 0) + 1
    
    total = len(styles)
    return {style: count / total for style, count in style_counts.items()}


def _infer_outfit_type(categories: List[str], styles: List[str]) -> str:
    """Kategori ve stillerden outfit tipini çıkarır"""
    # Basit kural tabanlı çıkarım
    if any("dress" in cat.lower() for cat in categories):
        return "dress_outfit"
    
    has_formal_items = any(style in ["formal", "business", "elegant"] for style in styles)
    if has_formal_items:
        return "formal"
    
    has_casual_items = any(style in ["casual", "sporty", "comfortable"] for style in styles)
    if has_casual_items:
        return "casual"
    
    return "unknown"


def _convert_to_pil(image: Union[Image.Image, str, np.ndarray]) -> Image.Image:
    """
    Farklı formatlardan PIL Image'a dönüştürme yardımcı fonksiyonu
    
    Args:
        image: Dönüştürülecek görüntü
        
    Returns:
        PIL.Image.Image: RGB formatında PIL görüntüsü
        
    Raises:
        TypeError: Desteklenmeyen görüntü tipi
        OSError: Dosya bulunamadığında
    """
    if isinstance(image, str):
        # Dosya yolu ise yükle
        if not os.path.exists(image):
            raise OSError(f"Görüntü dosyası bulunamadı: {image}")
        pil_image = Image.open(image)
    elif isinstance(image, np.ndarray):
        # NumPy array ise dönüştür
        if image.ndim == 2:
            # Grayscale
            pil_image = Image.fromarray(image, mode="L")
        elif image.ndim == 3:
            # RGB veya BGR
            if image.shape[2] == 3:
                pil_image = Image.fromarray(image, mode="RGB")
            else:
                raise ValueError(f"Desteklenmeyen array şekli: {image.shape}")
        else:
            raise ValueError(f"Desteklenmeyen array boyutu: {image.ndim}")
    elif isinstance(image, Image.Image):
        pil_image = image
    else:
        raise TypeError(f"Desteklenmeyen görüntü tipi: {type(image)}")
    
    # RGB formatına çevir
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    
    return pil_image


class PolyvoreOutfitDataset(TorchDataset):
    """
    PyTorch Dataset wrapper for Polyvore outfits data
    
    Bu sınıf, Polyvore outfit veri setini PyTorch DataLoader ile 
    kullanılabilir hale getirir.
    """
    
    def __init__(self, 
                 dataset: Dict[str, Any],
                 image_dir: str,
                 processor: Optional[AutoImageProcessor] = None,
                 transforms: Optional[transforms.Compose] = None,
                 max_items: int = MAX_ITEMS_PER_OUTFIT):
        """
        Dataset initialize et
        
        Args:
            dataset: Polyvore dataset dict
            image_dir: Görüntü dosyalarının bulunduğu dizin
            processor: Vision processor
            transforms: Opsiyonel transform'lar
            max_items: Outfit başına maksimum item sayısı
        """
        self.outfits = dataset['outfits']
        self.items = dataset['items']
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.transforms = transforms
        self.max_items = max_items
        
        # Valid outfitleri filtrele (göründklerini ve yeterli item'a sahip olanları)
        self.valid_outfits = self._filter_valid_outfits()
        
        logger.info(f"Dataset oluşturuldu: {len(self.valid_outfits)} geçerli outfit")
        
    def _filter_valid_outfits(self) -> List[Dict[str, Any]]:
        """Geçerli outfitleri filtreler"""
        valid_outfits = []
        
        for outfit in self.outfits:
            outfit_items = outfit.get('items', [])
            
            # En az 2 item olmalı
            if len(outfit_items) < 2:
                continue
            
            # Item'ların görüntüleri mevcut olmalı
            valid_items = []
            for item_id in outfit_items:
                if item_id in self.items:
                    item = self.items[item_id]
                    image_path = self.image_dir / f"{item_id}.jpg"
                    if image_path.exists():
                        valid_items.append(item_id)
            
            # En az 2 geçerli item olmalı
            if len(valid_items) >= 2:
                outfit_copy = outfit.copy()
                outfit_copy['items'] = valid_items[:self.max_items]  # Maksimum item sayısı
                valid_outfits.append(outfit_copy)
        
        return valid_outfits
        
    def __len__(self) -> int:
        """Dataset uzunluğu"""
        return len(self.valid_outfits)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Belirtilen index'teki outfit'i döndür
        
        Args:
            idx: Outfit indeksi
            
        Returns:
            Dict: İşlenmiş outfit verileri
        """
        outfit = self.valid_outfits[idx]
        outfit_items = outfit['items']
        
        # Item görüntülerini yükle ve işle
        item_images = []
        item_features = []
        
        for item_id in outfit_items:
            item = self.items[item_id]
            
            # Görüntüyü yükle
            image_path = self.image_dir / f"{item_id}.jpg"
            
            try:
                image = Image.open(image_path).convert("RGB")
                
                # Görüntüyü işle
                if self.transforms:
                    processed_image = self.transforms(image)
                else:
                    processed_image = preprocess_outfit_image(image, self.processor)
                
                item_images.append(processed_image)
                
                # Item feature'larını çıkar
                features = {
                    'category': item.get('category', ''),
                    'color': item.get('color', ''),
                    'style': item.get('style', ''),
                    'price': float(item.get('price', 0)) if item.get('price') else 0.0
                }
                item_features.append(features)
                
            except Exception as e:
                logger.warning(f"Item {item_id} yüklenemedi: {e}")
                # Dummy tensör oluştur
                dummy_image = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
                item_images.append(dummy_image)
                item_features.append({'category': '', 'color': '', 'style': '', 'price': 0.0})
        
        # Outfit feature'larını çıkar
        outfit_features = extract_outfit_features(outfit, self.items)
        
        # Padding (eğer max_items'dan az item varsa)
        while len(item_images) < self.max_items:
            dummy_image = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
            item_images.append(dummy_image)
            item_features.append({'category': '', 'color': '', 'style': '', 'price': 0.0})
        
        # Tensörlere dönüştür
        item_images_tensor = torch.stack(item_images)  # [max_items, 3, H, W]
        
        return {
            "outfit_id": outfit.get('outfit_id', idx),
            "item_images": item_images_tensor,
            "item_features": item_features,
            "outfit_features": outfit_features,
            "num_items": len(outfit_items)
        }


def create_outfit_dataloader(dataset: Dict[str, Any],
                           image_dir: str,
                           batch_size: int = 16,
                           shuffle: bool = True,
                           num_workers: int = 4,
                           processor: Optional[AutoImageProcessor] = None) -> DataLoader:
    """
    Polyvore outfit veri seti için PyTorch DataLoader oluşturur
    
    Args:
        dataset: Polyvore dataset
        image_dir: Görüntü dizini
        batch_size: Batch boyutu
        shuffle: Veriyi karıştır
        num_workers: Worker thread sayısı
        processor: Vision processor
        
    Returns:
        DataLoader: PyTorch DataLoader
    """
    outfit_dataset = PolyvoreOutfitDataset(
        dataset=dataset,
        image_dir=image_dir,
        processor=processor
    )
    
    return DataLoader(
        dataset=outfit_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_outfits
    )


def _collate_outfits(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Outfit batch'ini collate eden fonksiyon
    
    Args:
        batch: Outfit listesi
        
    Returns:
        Dict: Collate edilmiş batch
    """
    # Batch içindeki tüm alanları birleştir
    outfit_ids = [item['outfit_id'] for item in batch]
    item_images = torch.stack([item['item_images'] for item in batch])  # [B, max_items, 3, H, W]
    num_items = torch.tensor([item['num_items'] for item in batch])
    
    # Item features ve outfit features'ı ayrı ayrı toplayalım
    item_features_batch = [item['item_features'] for item in batch]
    outfit_features_batch = [item['outfit_features'] for item in batch]
    
    return {
        "outfit_ids": outfit_ids,
        "item_images": item_images,
        "item_features": item_features_batch,
        "outfit_features": outfit_features_batch,
        "num_items": num_items
    }


def get_polyvore_transforms(is_training: bool = True) -> transforms.Compose:
    """
    Polyvore veri seti için uygun transform'ları döndürür
    
    Args:
        is_training: Eğitim için mi yoksa inference için mi
        
    Returns:
        transforms.Compose: Composed transform'lar
    """
    if is_training:
        # Eğitim için data augmentation ekle
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.1, 
                contrast=0.1, 
                saturation=0.1, 
                hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        # Inference için sadece temel preprocessing
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


def validate_outfit_data(outfit: Dict[str, Any], items: Dict[str, Dict[str, Any]]) -> bool:
    """
    Outfit verisinin gerekli alanları içerip içermediğini doğrular
    
    Args:
        outfit: Outfit verisi
        items: Items mapping
        
    Returns:
        bool: Outfit geçerli ise True, değilse False
    """
    required_fields = ["items"]
    
    # Gerekli alanları kontrol et
    for field in required_fields:
        if field not in outfit:
            logger.warning(f"Outfit'te gerekli alan eksik: {field}")
            return False
    
    # Item'ları doğrula
    outfit_items = outfit["items"]
    if not isinstance(outfit_items, list) or len(outfit_items) < 2:
        logger.warning("Outfit en az 2 item içermeli")
        return False
    
    # Item'ların varlığını kontrol et
    valid_items = 0
    for item_id in outfit_items:
        if item_id in items:
            valid_items += 1
    
    if valid_items < 2:
        logger.warning("Outfit'te en az 2 geçerli item olmalı")
        return False
    
    return True


# Test fonksiyonu
def test_polyvore_data_loader():
    """
    Polyvore data loader fonksiyonlarını test eder
    
    Bu fonksiyon, tüm temel fonksiyonların çalışıp çalışmadığını kontrol eder.
    """
    logger.info("Polyvore data loader test başlatılıyor...")
    
    try:
        # 1. Fake data ile test
        logger.info("1. Fake Polyvore data ile test...")
        
        # Mock outfits ve items oluştur
        mock_items = {
            "item_1": {"category": "shirt", "color": "blue", "style": "casual", "price": "29.99"},
            "item_2": {"category": "jeans", "color": "blue", "style": "casual", "price": "59.99"},
            "item_3": {"category": "sneakers", "color": "white", "style": "sporty", "price": "89.99"}
        }
        
        mock_outfits = [
            {"outfit_id": "outfit_1", "items": ["item_1", "item_2", "item_3"], "season": "spring"},
            {"outfit_id": "outfit_2", "items": ["item_1", "item_2"], "season": "summer"}
        ]
        
        mock_dataset = {
            "outfits": mock_outfits,
            "items": mock_items,
            "split": "test",
            "num_outfits": 2,
            "num_items": 3
        }
        
        # 2. Feature extraction testi
        logger.info("2. Feature extraction testi...")
        features = extract_outfit_features(mock_outfits[0], mock_items)
        logger.info(f"Outfit features: {features}")
        
        # 3. Compatibility graph testi
        logger.info("3. Compatibility graph testi...")
        graph = create_outfit_compatibility_graph(mock_outfits, mock_items)
        logger.info(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # 4. Image preprocessing testi
        logger.info("4. Image preprocessing testi...")
        test_image = Image.new("RGB", (300, 300), color="red")
        processed = preprocess_outfit_image(test_image)
        logger.info(f"Processed image shape: {processed.shape}")
        
        logger.info("Tüm testler başarılı!")
        
    except Exception as e:
        logger.error(f"Test hatası: {e}")
        raise


if __name__ == "__main__":
    # Test çalıştır
    test_polyvore_data_loader()
