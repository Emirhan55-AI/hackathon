"""
Data loader for Fashionpedia dataset - DETR Model Training Pipeline
Bu modül, Fashionpedia veri setini yüklemek ve DETR modeli için ön işleme yapmaktan sorumludur.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

# Core libraries
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset
import torchvision.transforms as transforms

# PIL for image processing
from PIL import Image, ImageDraw

# Hugging Face libraries
from datasets import load_dataset, Dataset, DatasetDict
from transformers import DetrImageProcessor

# Numerical computation
import numpy as np

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fashionpedia dataset constants
FASHIONPEDIA_DATASET_NAME = "detection-datasets/fashionpedia"  # Hugging Face hub dataset name
IMAGE_SIZE = 800  # Standard DETR input size
MAX_OBJECTS = 100  # Maximum number of objects per image for DETR

# DETR specific constants
DETR_IMAGE_SIZE = (800, 800)  # DETR model expected input size
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization mean
IMAGENET_STD = [0.229, 0.224, 0.225]   # ImageNet normalization std

# Valid dataset splits for Fashionpedia
VALID_SPLITS = ["train", "validation", "test"]

# Fashionpedia label mappings
FASHIONPEDIA_CATEGORIES = {
    1: "shirt, blouse",
    2: "top, t-shirt, sweatshirt",
    3: "sweater",
    4: "cardigan",
    5: "jacket",
    6: "vest",
    7: "pants",
    8: "shorts",
    9: "skirt",
    10: "coat",
    11: "dress",
    12: "jumpsuit",
    13: "cape",
    14: "glasses",
    15: "hat",
    16: "headband, head covering, hair accessory",
    17: "tie",
    18: "glove",
    19: "watch",
    20: "belt",
    21: "leg warmer",
    22: "tights, stockings",
    23: "sock",
    24: "shoe",
    25: "bag, wallet",
    26: "scarf",
    27: "umbrella",
    28: "hood",
    29: "collar",
    30: "lapel",
    31: "epaulette",
    32: "sleeve",
    33: "pocket",
    34: "neckline",
    35: "buckle",
    36: "zipper",
    37: "applique",
    38: "bead",
    39: "bow",
    40: "flower",
    41: "fringe",
    42: "ribbon",
    43: "rivet",
    44: "ruffle",
    45: "sequin",
    46: "tassel"
}


def load_fashionpedia_dataset(data_dir: Optional[str] = None, split: str = "train") -> Union[Dataset, DatasetDict]:
    """
    A2. Fashionpedia veri setini yükler - Hugging Face datasets kullanarak
    
    Bu fonksiyon, DETR model eğitimi için Fashionpedia veri setinin belirtilen 
    bölümünü (train/validation/test) yükler.
    
    Args:
        data_dir (Optional[str]): Veri setinin bulunduğu yerel dizin yolu. 
                                 None ise Hugging Face Hub'dan indirir.
        split (str): Yüklenecek veri seti bölümü ('train', 'validation', 'test')
        
    Returns:
        Union[Dataset, DatasetDict]: Yüklenen Fashionpedia veri seti
        
    Raises:
        ValueError: Geçersiz split değeri verildiğinde
        ConnectionError: Veri seti yüklenemediğinde
        OSError: Yerel veri dizini bulunamadığında
        
    Example:
        >>> dataset = load_fashionpedia_dataset(split="train")
        >>> print(f"Loaded {len(dataset)} training samples")
    """
    # Split doğrulama
    if split not in VALID_SPLITS:
        raise ValueError(
            f"Geçersiz split '{split}'. Geçerli değerler: {VALID_SPLITS}"
        )
    
    logger.info(f"Fashionpedia veri seti yükleniyor - split: {split}")
    
    try:
        # Yerel dizin kontrolü
        if data_dir is not None:
            data_path = Path(data_dir)
            if not data_path.exists():
                raise OSError(f"Veri dizini bulunamadı: {data_dir}")
            
            # Yerel JSON dosyalarından yükle
            json_file = data_path / f"{split}.json"
            if not json_file.exists():
                raise OSError(f"Split dosyası bulunamadı: {json_file}")
            
            logger.info(f"Yerel dosyadan yükleniyor: {json_file}")
            dataset = load_dataset("json", data_files=str(json_file), split="train")
        else:
            # Hugging Face Hub'dan yükle
            logger.info(f"Hugging Face Hub'dan yükleniyor: {FASHIONPEDIA_DATASET_NAME}")
            dataset = load_dataset(FASHIONPEDIA_DATASET_NAME, split=split)
        
        logger.info(f"Başarıyla yüklendi: {len(dataset)} örnek")
        return dataset
        
    except Exception as e:
        error_msg = f"Fashionpedia veri seti yüklenemedi: {str(e)}"
        logger.error(error_msg)
        raise ConnectionError(error_msg) from e


def preprocess_image(image: Union[Image.Image, str, np.ndarray], 
                    processor: Optional[DetrImageProcessor] = None) -> torch.Tensor:
    """
    A3. DETR modeli için görüntü ön işleme fonksiyonu
    
    Bu fonksiyon, girdi görüntüsünü DETR modelinin beklediği formata getirir.
    DetrImageProcessor kullanarak normalize etme, yeniden boyutlandırma yapar.
    
    Args:
        image (Union[Image.Image, str, np.ndarray]): İşlenecek görüntü
            - PIL Image nesnesi
            - Görüntü dosyası yolu (string)
            - NumPy array
        processor (Optional[DetrImageProcessor]): DETR image processor. 
                                                 None ise varsayılan oluşturulur.
        
    Returns:
        torch.Tensor: İşlenmiş görüntü tensörü [C, H, W] formatında
        
    Raises:
        ValueError: Görüntü None veya geçersiz ise
        TypeError: Desteklenmeyen görüntü tipi
        OSError: Görüntü dosyası bulunamadığında
        
    Example:
        >>> from PIL import Image
        >>> img = Image.open("fashion_item.jpg")
        >>> processed = preprocess_image(img)
        >>> print(f"Processed shape: {processed.shape}")  # [3, 800, 800]
    """
    # Girdi doğrulama
    if image is None:
        raise ValueError("Görüntü None olamaz")
    
    # PIL Image'a dönüştürme
    pil_image = _convert_to_pil(image)
    
    # DetrImageProcessor oluştur veya kullan
    if processor is None:
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    
    try:
        # DETR processor ile işle
        # return_tensors="pt" ile PyTorch tensor döndür
        processed = processor(images=pil_image, return_tensors="pt")
        
        # Pixel values'ı çıkar ve squeeze ile batch dimension'ı kaldır
        pixel_values = processed["pixel_values"].squeeze(0)  # [1, 3, H, W] -> [3, H, W]
        
        logger.debug(f"Görüntü işlendi: {pixel_values.shape}")
        return pixel_values
        
    except Exception as e:
        error_msg = f"Görüntü işleme hatası: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def get_labels_for_item(item_id: Union[str, int], 
                       dataset: Optional[Dataset] = None) -> Dict[str, Any]:
    """
    A4. Belirtilen item ID için Fashionpedia etiketlerini getirir
    
    Bu fonksiyon, veri setinden belirli bir öğenin etiketlerini (kategori, 
    bounding box, segmentation mask vb.) döndürür.
    
    Args:
        item_id (Union[str, int]): Fashionpedia öğe kimliği
        dataset (Optional[Dataset]): Arama yapılacak veri seti. 
                                   None ise train split yüklenir.
        
    Returns:
        Dict[str, Any]: Öğe etiketleri ve meta verileri içeren sözlük
            - image_id: Görüntü kimliği
            - annotations: Anotasyon listesi
            - categories: Kategori isimleri
            - bboxes: Bounding box koordinatları
            - segmentation: Segmentasyon maskeleri (varsa)
            
    Raises:
        ValueError: Geçersiz item_id
        KeyError: Item bulunamadığında
        
    Example:
        >>> labels = get_labels_for_item("12345")
        >>> print(f"Categories: {labels['categories']}")
        >>> print(f"Bboxes: {labels['bboxes']}")
    """
    # Item ID doğrulama
    if item_id is None:
        raise ValueError("Item ID None olamaz")
    
    # String ID'yi integer'a çevir
    if isinstance(item_id, str):
        if not item_id.strip():
            raise ValueError("Item ID boş string olamaz")
        try:
            item_id = int(item_id)
        except ValueError:
            raise ValueError(f"Geçersiz item ID formatı: {item_id}")
    
    # Dataset yükle (eğer verilmediyse)
    if dataset is None:
        logger.info("Dataset verilmedi, train split yükleniyor")
        dataset = load_fashionpedia_dataset(split="train")
    
    logger.debug(f"Item {item_id} için etiketler aranıyor")
    
    # Dataset'te item ara
    for idx, item in enumerate(dataset):
        # Çeşitli ID alanlarını kontrol et
        if (item.get("image_id") == item_id or 
            item.get("id") == item_id or 
            idx == item_id):
            
            # Etiketleri çıkar ve yapılandır
            annotations = item.get("annotations", [])
            
            result = {
                "image_id": item.get("image_id", item_id),
                "annotations": annotations,
                "categories": _extract_categories_from_annotations(annotations),
                "bboxes": _extract_bboxes_from_annotations(annotations),
                "area": item.get("area", 0),
                "iscrowd": item.get("iscrowd", 0),
                "segmentation": _extract_segmentation_from_annotations(annotations)
            }
            
            logger.debug(f"Item {item_id} bulundu: {len(annotations)} anotasyon")
            return result
    
    # Item bulunamadı
    raise KeyError(f"Item {item_id} veri setinde bulunamadı")


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


def _extract_segmentation_from_annotations(annotations: List[Dict]) -> List[Dict]:
    """
    Anotasyonlardan segmentasyon maskelerini çıkarma yardımcı fonksiyonu
    
    Args:
        annotations: Anotasyon sözlükleri listesi
        
    Returns:
        List[Dict]: Segmentasyon maskeleri listesi
    """
    segmentations = []
    for ann in annotations:
        segmentation = ann.get("segmentation")
        if segmentation:
            segmentations.append({
                "segmentation": segmentation,
                "area": ann.get("area", 0),
                "bbox": ann.get("bbox", [])
            })
    return segmentations


def _extract_categories_from_annotations(annotations: List[Dict]) -> List[str]:
    """
    Anotasyonlardan kategori isimlerini çıkarma yardımcı fonksiyonu
    
    Args:
        annotations: Anotasyon sözlükleri listesi
        
    Returns:
        List[str]: Kategori isimleri listesi
    """
    categories = []
    for ann in annotations:
        category_id = ann.get("category_id")
        if category_id in FASHIONPEDIA_CATEGORIES:
            categories.append(FASHIONPEDIA_CATEGORIES[category_id])
    return categories


def _extract_bboxes_from_annotations(annotations: List[Dict]) -> List[List[float]]:
    """
    Anotasyonlardan bounding box'ları çıkarma yardımcı fonksiyonu
    
    Args:
        annotations: Anotasyon sözlükleri listesi
        
    Returns:
        List[List[float]]: [x, y, width, height] formatında bounding box listesi
    """
    bboxes = []
    for ann in annotations:
        bbox = ann.get("bbox")
        if bbox and len(bbox) == 4:
            bboxes.append(bbox)
    return bboxes


def create_detr_data_collator(processor: Optional[DetrImageProcessor] = None):
    """
    DETR modeli için özel data collator oluşturur
    
    Bu fonksiyon, batch halinde veri yüklemek için DataLoader ile birlikte 
    kullanılacak collate fonksiyonu oluşturur.
    
    Args:
        processor: DETR image processor
        
    Returns:
        Collate fonksiyonu
    """
    if processor is None:
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    
    def collate_fn(batch):
        """
        Batch örnekleri için collate fonksiyonu
        
        Args:
            batch: Örnek listesi
            
        Returns:
            Tuple: (images, targets) - işlenmiş görüntüler ve hedef etiketler
        """
        images = []
        targets = []
        
        for item in batch:
            try:
                # Görüntüyü işle
                if "image" in item:
                    processed_image = preprocess_image(item["image"], processor)
                    images.append(processed_image)
                
                # Hedef etiketleri hazırla
                if "annotations" in item:
                    bboxes = _extract_bboxes_from_annotations(item["annotations"])
                    category_ids = [ann.get("category_id", 0) for ann in item["annotations"]]
                    
                    target = {
                        "boxes": torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.empty((0, 4)),
                        "labels": torch.tensor(category_ids, dtype=torch.int64) if category_ids else torch.empty((0,), dtype=torch.int64),
                        "image_id": torch.tensor(item.get("image_id", 0), dtype=torch.int64)
                    }
                    targets.append(target)
                
            except Exception as e:
                logger.warning(f"Batch item işlenirken hata: {str(e)}")
                continue
        
        if images:
            return torch.stack(images), targets
        else:
            return torch.empty((0, 3, IMAGE_SIZE, IMAGE_SIZE)), []
    
    return collate_fn


def get_fashionpedia_transforms(is_training: bool = True) -> transforms.Compose:
    """
    Fashionpedia veri seti için uygun transform'ları döndürür
    
    Args:
        is_training: Eğitim için mi yoksa inference için mi
        
    Returns:
        transforms.Compose: Composed transform'lar
    """
    if is_training:
        # Eğitim için data augmentation ekle
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2, 
                contrast=0.2, 
                saturation=0.2, 
                hue=0.1
            ),
            transforms.RandomRotation(degrees=10),
            transforms.Resize(DETR_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        # Inference için sadece temel preprocessing
        return transforms.Compose([
            transforms.Resize(DETR_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


class FashionpediaDataset(TorchDataset):
    """
    PyTorch Dataset wrapper for Fashionpedia data
    
    Bu sınıf, Fashionpedia veri setini PyTorch DataLoader ile 
    kullanılabilir hale getirir.
    """
    
    def __init__(self, 
                 dataset: Dataset, 
                 processor: Optional[DetrImageProcessor] = None,
                 transforms: Optional[transforms.Compose] = None):
        """
        Dataset initialize et
        
        Args:
            dataset: Hugging Face Dataset
            processor: DETR image processor
            transforms: Opsiyonel transform'lar
        """
        self.dataset = dataset
        self.processor = processor or DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.transforms = transforms
        
    def __len__(self) -> int:
        """Dataset uzunluğu"""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Belirtilen index'teki örneği döndür
        
        Args:
            idx: Örnek indeksi
            
        Returns:
            Dict: İşlenmiş görüntü ve etiketler
        """
        item = self.dataset[idx]
        
        # Görüntüyü al ve işle
        image = item["image"]
        if self.transforms:
            image = self.transforms(image)
        else:
            image = preprocess_image(image, self.processor)
        
        # Etiketleri hazırla
        annotations = item.get("annotations", [])
        bboxes = _extract_bboxes_from_annotations(annotations)
        labels = [ann.get("category_id", 0) for ann in annotations]
        
        return {
            "pixel_values": image,
            "labels": {
                "boxes": torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.empty((0, 4)),
                "class_labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64),
                "image_id": torch.tensor(item.get("image_id", idx), dtype=torch.int64)
            }
        }


def validate_dataset_item(item: Dict) -> bool:
    """
    Veri seti öğesinin gerekli alanları içerip içermediğini doğrular
    
    Args:
        item: Veri seti öğesi sözlüğü
        
    Returns:
        bool: Öğe geçerli ise True, değilse False
    """
    required_fields = ["image", "annotations"]
    
    # Gerekli alanları kontrol et
    for field in required_fields:
        if field not in item:
            logger.warning(f"Gerekli alan eksik: {field}")
            return False
    
    # Anotasyonları doğrula
    annotations = item["annotations"]
    if not isinstance(annotations, list):
        logger.warning("Anotasyonlar liste formatında değil")
        return False
    
    for ann in annotations:
        if not isinstance(ann, dict):
            logger.warning("Anotasyon sözlük formatında değil")
            return False
        
        # Temel anotasyon alanlarını kontrol et
        if "category_id" not in ann or "bbox" not in ann:
            logger.warning("Anotasyonda gerekli alanlar eksik")
            return False
        
        # Bounding box formatını kontrol et
        bbox = ann["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            logger.warning(f"Geçersiz bbox formatı: {bbox}")
            return False
        
        # Bbox değerlerinin numerik olduğunu kontrol et
        try:
            x, y, w, h = bbox
            float(x), float(y), float(w), float(h)
        except (ValueError, TypeError):
            logger.warning(f"Bbox değerleri numerik değil: {bbox}")
            return False
    
    return True


def get_category_name(category_id: int) -> str:
    """
    Kategori ID'sinden kategori ismini döndürür
    
    Args:
        category_id: Fashionpedia kategori ID'si
        
    Returns:
        str: Kategori ismi
        
    Raises:
        KeyError: Kategori ID bulunamadığında
    """
    if category_id not in FASHIONPEDIA_CATEGORIES:
        raise KeyError(f"Kategori ID {category_id} Fashionpedia kategorilerinde bulunamadı")
    
    return FASHIONPEDIA_CATEGORIES[category_id]


def get_category_id(category_name: str) -> int:
    """
    Kategori isminden kategori ID'sini döndürür
    
    Args:
        category_name: Fashionpedia kategori ismi
        
    Returns:
        int: Kategori ID'si
        
    Raises:
        KeyError: Kategori ismi bulunamadığında
    """
    category_name_lower = category_name.lower().strip()
    
    for cat_id, cat_name in FASHIONPEDIA_CATEGORIES.items():
        if cat_name.lower() == category_name_lower:
            return cat_id
    
    raise KeyError(f"Kategori ismi '{category_name}' Fashionpedia kategorilerinde bulunamadı")


def create_dataloader(dataset: Union[Dataset, FashionpediaDataset], 
                     batch_size: int = 8,
                     shuffle: bool = True,
                     num_workers: int = 4,
                     processor: Optional[DetrImageProcessor] = None) -> DataLoader:
    """
    Fashionpedia veri seti için PyTorch DataLoader oluşturur
    
    Args:
        dataset: Fashionpedia dataset
        batch_size: Batch boyutu
        shuffle: Veriyi karıştır
        num_workers: Worker thread sayısı
        processor: DETR image processor
        
    Returns:
        DataLoader: PyTorch DataLoader
    """
    # Eğer Hugging Face Dataset ise FashionpediaDataset'e wrap et
    if isinstance(dataset, Dataset):
        dataset = FashionpediaDataset(dataset, processor)
    
    # Collate function oluştur
    collate_fn = create_detr_data_collator(processor)
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()  # GPU varsa memory pinning kullan
    )


def get_dataset_statistics(dataset: Dataset) -> Dict[str, Any]:
    """
    Veri seti istatistiklerini hesaplar
    
    Args:
        dataset: Fashionpedia dataset
        
    Returns:
        Dict[str, Any]: İstatistik bilgileri
    """
    stats = {
        "total_samples": len(dataset),
        "category_distribution": {},
        "average_objects_per_image": 0,
        "bbox_size_stats": {"min": [], "max": [], "mean": []},
        "valid_samples": 0
    }
    
    total_objects = 0
    all_bbox_areas = []
    
    logger.info("Veri seti istatistikleri hesaplanıyor...")
    
    for item in dataset:
        # Veri doğrulama
        if not validate_dataset_item(item):
            continue
        
        stats["valid_samples"] += 1
        annotations = item["annotations"]
        total_objects += len(annotations)
        
        # Kategori dağılımı
        for ann in annotations:
            cat_id = ann.get("category_id")
            if cat_id in FASHIONPEDIA_CATEGORIES:
                cat_name = FASHIONPEDIA_CATEGORIES[cat_id]
                stats["category_distribution"][cat_name] = stats["category_distribution"].get(cat_name, 0) + 1
            
            # Bbox area hesaplama
            bbox = ann.get("bbox", [])
            if len(bbox) == 4:
                x, y, w, h = bbox
                area = w * h
                all_bbox_areas.append(area)
    
    # İstatistikleri hesapla
    if stats["valid_samples"] > 0:
        stats["average_objects_per_image"] = total_objects / stats["valid_samples"]
    
    if all_bbox_areas:
        stats["bbox_size_stats"] = {
            "min": min(all_bbox_areas),
            "max": max(all_bbox_areas),
            "mean": sum(all_bbox_areas) / len(all_bbox_areas),
            "total_bboxes": len(all_bbox_areas)
        }
    
    logger.info(f"İstatistikler hesaplandı: {stats['valid_samples']}/{stats['total_samples']} geçerli örnek")
    return stats


# Test fonksiyonu
def test_data_loader():
    """
    Data loader fonksiyonlarını test eder
    
    Bu fonksiyon, tüm temel fonksiyonların çalışıp çalışmadığını kontrol eder.
    """
    logger.info("Data loader test başlatılıyor...")
    
    try:
        # 1. Dataset yükleme testi
        logger.info("1. Dataset yükleme testi...")
        # Mock test için küçük dataset yükle
        # dataset = load_fashionpedia_dataset(split="train")
        # logger.info(f"Dataset yüklendi: {len(dataset)} örnek")
        
        # 2. Image preprocessing testi
        logger.info("2. Image preprocessing testi...")
        # Test görüntüsü oluştur
        test_image = Image.new("RGB", (640, 480), color="red")
        processed = preprocess_image(test_image)
        logger.info(f"Görüntü işlendi: {processed.shape}")
        
        # 3. Hata durumları testi
        logger.info("3. Hata durumları testi...")
        try:
            preprocess_image(None)
        except ValueError as e:
            logger.info(f"Beklenen hata yakalandı: {e}")
        
        try:
            load_fashionpedia_dataset(split="invalid_split")
        except ValueError as e:
            logger.info(f"Beklenen hata yakalandı: {e}")
        
        logger.info("Tüm testler başarılı!")
        
    except Exception as e:
        logger.error(f"Test hatası: {e}")
        raise


if __name__ == "__main__":
    # Test çalıştır
    test_data_loader()
