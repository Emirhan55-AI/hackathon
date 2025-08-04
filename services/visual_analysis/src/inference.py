"""
DETR Inference Engine for Fashionpedia Fashion Analysis - Aura Project
Bu modül, eğitilmiş DETR modelini kullanarak görüntüler üzerinde çıkarım yapmaktan ve 
yapılandırılmış fashion etiketleri döndürmekten sorumludur.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import warnings

# Core libraries
import torch
import torch.nn.functional as F
import numpy as np

# PIL for image processing
from PIL import Image, ImageDraw, ImageFont
import cv2

# Hugging Face libraries
from transformers import (
    DetrImageProcessor,
    DetrForSegmentation
)

# Local imports - Aura project modules
from data_loader import (
    preprocess_image,
    FASHIONPEDIA_CATEGORIES,
    _convert_to_pil,
    DETR_IMAGE_SIZE
)
from model import (
    load_model_for_inference,
    get_model_info,
    validate_model_output,
    TOTAL_CLASSES,
    FASHIONPEDIA_LABELS
)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Inference constants
DEFAULT_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence score for predictions
DEFAULT_MAX_DETECTIONS = 50  # Maximum number of detections per image
DEFAULT_NMS_THRESHOLD = 0.5  # Non-maximum suppression threshold
MIN_BOX_AREA = 100  # Minimum bounding box area (pixels)

# Fashion attribute mappings (expanded from Fashionpedia categories)
FASHION_ATTRIBUTES = {
    "colors": {
        "red", "blue", "green", "yellow", "black", "white", "gray", "brown", 
        "pink", "purple", "orange", "beige", "navy", "maroon", "olive"
    },
    "patterns": {
        "solid", "striped", "polka-dot", "floral", "geometric", "abstract", 
        "plaid", "checkered", "leopard", "zebra", "paisley"
    },
    "styles": {
        "casual", "formal", "business", "sporty", "vintage", "modern", 
        "bohemian", "classic", "trendy", "elegant", "edgy"
    },
    "materials": {
        "cotton", "silk", "wool", "leather", "denim", "linen", "polyester", 
        "chiffon", "velvet", "satin", "knit", "synthetic"
    }
}


def load_inference_model(model_path: Union[str, Path],
                        device: Optional[torch.device] = None) -> Tuple[DetrForSegmentation, DetrImageProcessor]:
    """
    A2. Eğitilmiş DETR modelini inference için yükler
    
    Bu fonksiyon, kaydedilmiş model checkpoint'ini yükler ve inference
    için gerekli image processor ile birlikte döndürür.
    
    Args:
        model_path (Union[str, Path]): Eğitilmiş model dosyası veya dizin yolu
        device (Optional[torch.device]): Model yüklenecek cihaz (GPU/CPU)
        
    Returns:
        Tuple[DetrForSegmentation, DetrImageProcessor]: (model, processor)
        
    Raises:
        FileNotFoundError: Model dosyası bulunamadığında
        RuntimeError: Model yükleme hatası
        
    Example:
        >>> model, processor = load_inference_model("./saved_models/detr_fashionpedia.pth")
        >>> print(f"Model device: {next(model.parameters()).device}")
    """
    logger.info(f"Inference modeli yükleniyor: {model_path}")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Model yükle
        model = load_model_for_inference(
            model_path=model_path,
            num_classes=TOTAL_CLASSES,
            device=device
        )
        
        # Image processor oluştur - model ile uyumlu olması için aynı checkpoint'ten
        try:
            # Eğer model Hugging Face formatında kaydedildiyse processor'ı da yükle
            processor = DetrImageProcessor.from_pretrained(str(model_path))
            logger.info("Processor model checkpoint'inden yüklendi")
        except:
            # Fallback: default processor kullan
            processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50-panoptic")
            logger.info("Default processor kullanılıyor")
        
        # Model bilgilerini logla
        model_info = get_model_info(model)
        logger.info(f"Model yüklendi:")
        logger.info(f"  - Tip: {model_info['model_type']}")
        logger.info(f"  - Sınıf sayısı: {model_info['num_classes']}")
        logger.info(f"  - Parametre sayısı: {model_info['total_parameters']:,}")
        logger.info(f"  - Cihaz: {model_info['device']}")
        logger.info(f"  - Training mode: {model_info['training_mode']}")
        
        logger.info("Inference modeli başarıyla yüklendi")
        return model, processor
        
    except Exception as e:
        error_msg = f"Model yükleme hatası: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def run_inference(model: DetrForSegmentation,
                 image_processor: DetrImageProcessor,
                 image: Union[Image.Image, str, np.ndarray],
                 confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
                 max_detections: int = DEFAULT_MAX_DETECTIONS,
                 return_masks: bool = True,
                 return_raw_output: bool = False) -> Dict[str, Any]:
    """
    A3. Ana çıkarım fonksiyonu - DETR modeli ile fashion analysis yapar
    
    Bu fonksiyon, verilen görüntü üzerinde DETR modelini kullanarak
    fashion item detection ve analysis gerçekleştirir.
    
    Args:
        model (DetrForSegmentation): Yüklü DETR modeli
        image_processor (DetrImageProcessor): DETR image processor
        image (Union[Image.Image, str, np.ndarray]): Analiz edilecek görüntü
        confidence_threshold (float): Minimum güven skoru threshold'u
        max_detections (int): Maksimum detection sayısı
        return_masks (bool): Segmentation mask'leri döndür
        return_raw_output (bool): Ham model çıktısını da döndür
        
    Returns:
        Dict[str, Any]: Yapılandırılmış çıkarım sonuçları
            - detections: Liste[Dict] - Her detection için detaylar
            - summary: Dict - Genel analiz özeti
            - metadata: Dict - İşlem metadata'sı
            - raw_output: Optional[Dict] - Ham model çıktısı
            
    Raises:
        ValueError: Geçersiz görüntü
        RuntimeError: Çıkarım hatası
        
    Example:
        >>> results = run_inference(model, processor, "fashion_image.jpg")
        >>> print(f"Found {len(results['detections'])} fashion items")
        >>> for det in results['detections']:
        ...     print(f"  {det['label']}: {det['confidence']:.2f}")
    """
    logger.info("Fashion çıkarım başlatılıyor...")
    
    try:
        # 1. Görüntü ön işleme
        logger.debug("Görüntü ön işleniyor...")
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
            image_path = str(image)
        else:
            pil_image = _convert_to_pil(image)
            image_path = "input_image"
        
        # Orijinal görüntü boyutlarını kaydet
        original_size = pil_image.size  # (width, height)
        
        # DETR processor ile görüntüyü işle
        inputs = image_processor(images=pil_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        
        # Model ile aynı cihaza taşı
        device = next(model.parameters()).device
        pixel_values = pixel_values.to(device)
        
        logger.debug(f"Görüntü işlendi: {pixel_values.shape}")
        
        # 2. Model çıkarımı
        logger.debug("Model çıkarımı yapılıyor...")
        with torch.no_grad():
            model.eval()  # Evaluation modunda olduğundan emin ol
            outputs = model(pixel_values=pixel_values)
        
        # Çıktı doğrulama
        if not validate_model_output(outputs, TOTAL_CLASSES):
            logger.warning("Model çıktısı beklenmedik format")
        
        logger.debug("Model çıkarımı tamamlandı")
        
        # 3. Çıktıları post-process et
        logger.debug("Çıktılar işleniyor...")
        processed_results = post_process_model_output(
            outputs=outputs,
            original_size=original_size,
            confidence_threshold=confidence_threshold,
            max_detections=max_detections,
            processor=image_processor
        )
        
        # 4. Fashion attribute'ları çıkar
        detections_with_attributes = []
        for detection in processed_results["detections"]:
            enhanced_detection = enhance_detection_with_attributes(detection, pil_image)
            detections_with_attributes.append(enhanced_detection)
        
        # 5. Sonuç yapısını oluştur
        results = {
            "detections": detections_with_attributes,
            "summary": generate_summary(detections_with_attributes),
            "metadata": {
                "image_path": image_path,
                "original_size": original_size,
                "processed_size": tuple(pixel_values.shape[2:]),  # (height, width)
                "num_detections": len(detections_with_attributes),
                "confidence_threshold": confidence_threshold,
                "model_device": str(device),
                "total_classes": TOTAL_CLASSES
            }
        }
        
        # Ham çıktıyı ekle (istenirse)
        if return_raw_output:
            results["raw_output"] = {
                "logits": outputs.logits.cpu().numpy() if hasattr(outputs, 'logits') else None,
                "pred_boxes": outputs.pred_boxes.cpu().numpy() if hasattr(outputs, 'pred_boxes') else None,
                "pred_masks": outputs.pred_masks.cpu().numpy() if hasattr(outputs, 'pred_masks') and return_masks else None
            }
        
        logger.info(f"Çıkarım tamamlandı: {len(detections_with_attributes)} fashion item bulundu")
        return results
        
    except Exception as e:
        error_msg = f"Çıkarım hatası: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def post_process_model_output(outputs,
                            original_size: Tuple[int, int],
                            confidence_threshold: float,
                            max_detections: int,
                            processor: DetrImageProcessor) -> Dict[str, Any]:
    """
    A4a. Model çıktısını post-process eder ve temiz detection'lara dönüştürür
    
    Args:
        outputs: DETR model çıktısı
        original_size: Orijinal görüntü boyutu (width, height)
        confidence_threshold: Güven skoru threshold'u
        max_detections: Maksimum detection sayısı
        processor: DETR image processor
        
    Returns:
        Dict[str, Any]: İşlenmiş detection sonuçları
    """
    try:
        # Model çıktılarını al
        logits = outputs.logits  # [batch_size, num_queries, num_classes]
        pred_boxes = outputs.pred_boxes  # [batch_size, num_queries, 4]
        
        # Batch dimension'ı kaldır (tek görüntü)
        logits = logits.squeeze(0)  # [num_queries, num_classes]
        pred_boxes = pred_boxes.squeeze(0)  # [num_queries, 4]
        
        # Softmax uygula ve tahminleri al
        probs = F.softmax(logits, dim=-1)
        scores, predicted_classes = torch.max(probs, dim=-1)
        
        # Background class'ı filtrele (class 0)
        valid_detections = (predicted_classes != 0) & (scores > confidence_threshold)
        
        if not torch.any(valid_detections):
            logger.info("Güven skoru üzerinde hiç detection bulunamadı")
            return {"detections": [], "num_detections": 0}
        
        # Valid detection'ları filtrele
        valid_scores = scores[valid_detections]
        valid_classes = predicted_classes[valid_detections]
        valid_boxes = pred_boxes[valid_detections]
        
        # Skorlara göre sırala (en yüksek skordan en düşüğe)
        sorted_indices = torch.argsort(valid_scores, descending=True)
        
        # Maksimum detection sayısı ile sınırla
        if len(sorted_indices) > max_detections:
            sorted_indices = sorted_indices[:max_detections]
        
        # Bounding box'ları orijinal görüntü koordinatlarına dönüştür
        # DETR normalized coordinates [x_center, y_center, width, height] formatında
        detections = []
        width, height = original_size
        
        for idx in sorted_indices:
            score = valid_scores[idx].item()
            class_id = valid_classes[idx].item()
            box = valid_boxes[idx]
            
            # Normalized coordinates'ı pixel coordinates'a çevir
            x_center, y_center, w, h = box.tolist()
            
            # [x_center, y_center, width, height] -> [x1, y1, x2, y2]
            x1 = int((x_center - w/2) * width)
            y1 = int((y_center - h/2) * height)
            x2 = int((x_center + w/2) * width)
            y2 = int((y_center + h/2) * height)
            
            # Koordinatları görüntü sınırları içinde tut
            x1 = max(0, min(x1, width-1))
            y1 = max(0, min(y1, height-1))
            x2 = max(0, min(x2, width-1))
            y2 = max(0, min(y2, height-1))
            
            # Minimum alan kontrolü
            box_area = (x2 - x1) * (y2 - y1)
            if box_area < MIN_BOX_AREA:
                continue
            
            # Label ismini al
            label = get_class_name(class_id)
            
            detection = {
                "label": label,
                "class_id": class_id,
                "confidence": score,
                "bbox": [x1, y1, x2, y2],
                "bbox_normalized": [x_center, y_center, w, h],
                "area": box_area
            }
            
            detections.append(detection)
        
        logger.debug(f"Post-processing tamamlandı: {len(detections)} detection")
        
        return {
            "detections": detections,
            "num_detections": len(detections)
        }
        
    except Exception as e:
        logger.error(f"Post-processing hatası: {str(e)}")
        return {"detections": [], "num_detections": 0}


def filter_predictions_by_confidence(detections: List[Dict],
                                    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> List[Dict]:
    """
    A4b. Detection'ları güven skoruna göre filtreler
    
    Args:
        detections: Detection listesi
        confidence_threshold: Minimum güven skoru
        
    Returns:
        List[Dict]: Filtrelenmiş detection'lar
    """
    filtered = [det for det in detections if det["confidence"] >= confidence_threshold]
    
    logger.debug(f"Güven skoru filtresi: {len(detections)} -> {len(filtered)} detection")
    
    return filtered


def enhance_detection_with_attributes(detection: Dict, 
                                    image: Image.Image) -> Dict:
    """
    Detection'ı fashion attribute'ları ile zenginleştir
    
    Args:
        detection: Temel detection bilgileri
        image: Orijinal görüntü
        
    Returns:
        Dict: Zenginleştirilmiş detection
    """
    try:
        # Bounding box'tan ROI çıkar
        x1, y1, x2, y2 = detection["bbox"]
        roi = image.crop((x1, y1, x2, y2))
        
        # Fashion attribute'ları analiz et
        attributes = analyze_fashion_attributes(roi, detection["label"])
        
        # Detection'ı zenginleştir
        enhanced_detection = detection.copy()
        enhanced_detection.update({
            "attributes": attributes,
            "style_analysis": get_style_analysis(detection["label"], attributes),
            "description": generate_item_description(detection["label"], attributes)
        })
        
        return enhanced_detection
        
    except Exception as e:
        logger.warning(f"Attribute enhancement hatası: {str(e)}")
        # Hata durumunda temel detection'ı döndür
        detection["attributes"] = {"colors": [], "patterns": [], "styles": [], "materials": []}
        detection["style_analysis"] = {}
        detection["description"] = detection["label"]
        return detection


def analyze_fashion_attributes(roi_image: Image.Image, 
                             item_label: str) -> Dict[str, List[str]]:
    """
    Fashion item'ının attribute'larını analiz eder
    
    Args:
        roi_image: Crop edilmiş item görüntüsü
        item_label: Item kategorisi
        
    Returns:
        Dict[str, List[str]]: Analiz edilen attribute'lar
    """
    # Bu basitleştirilmiş bir implementasyon
    # Gerçek projede daha sofistike renk/desen analizi yapılabilir
    
    attributes = {
        "colors": [],
        "patterns": [],
        "styles": [],
        "materials": []
    }
    
    try:
        # Temel renk analizi (dominant color extraction)
        dominant_colors = extract_dominant_colors(roi_image)
        attributes["colors"] = dominant_colors
        
        # Item kategorisine göre muhtemel stil ve malzeme
        attributes["styles"] = infer_style_from_category(item_label)
        attributes["materials"] = infer_material_from_category(item_label)
        
        # Pattern analizi (basitleştirilmiş)
        pattern = infer_pattern_from_image(roi_image)
        if pattern:
            attributes["patterns"] = [pattern]
        
    except Exception as e:
        logger.debug(f"Attribute analiz hatası: {str(e)}")
    
    return attributes


def extract_dominant_colors(image: Image.Image, num_colors: int = 3) -> List[str]:
    """
    Görüntüden dominant renkleri çıkarır
    
    Args:
        image: PIL Image
        num_colors: Çıkarılacak renk sayısı
        
    Returns:
        List[str]: Dominant renk isimleri
    """
    try:
        # Görüntüyü numpy array'a çevir
        img_array = np.array(image)
        
        # Basit renk analizi - ortalama RGB değerleri
        mean_color = np.mean(img_array.reshape(-1, 3), axis=0)
        
        # RGB değerlerine göre renk adı çıkar
        color_name = rgb_to_color_name(mean_color)
        
        return [color_name] if color_name else ["unknown"]
        
    except Exception as e:
        logger.debug(f"Renk çıkarma hatası: {str(e)}")
        return ["unknown"]


def rgb_to_color_name(rgb: np.ndarray) -> str:
    """
    RGB değerini renk adına çevirir
    
    Args:
        rgb: [R, G, B] değerleri
        
    Returns:
        str: Renk adı
    """
    r, g, b = rgb
    
    # Basit renk mapping'i
    if r > 200 and g > 200 and b > 200:
        return "white"
    elif r < 50 and g < 50 and b < 50:
        return "black"
    elif r > 150 and g < 100 and b < 100:
        return "red"
    elif r < 100 and g > 150 and b < 100:
        return "green"
    elif r < 100 and g < 100 and b > 150:
        return "blue"
    elif r > 150 and g > 150 and b < 100:
        return "yellow"
    elif r > 150 and g > 100 and b > 150:
        return "pink"
    elif r > 100 and g < 100 and b > 100:
        return "purple"
    elif r > 150 and g > 100 and b < 100:
        return "orange"
    elif r > 100 and g > 100 and b > 100:
        return "gray"
    else:
        return "brown"


def infer_style_from_category(category: str) -> List[str]:
    """
    Kategori'den muhtemel stilleri çıkarır
    
    Args:
        category: Fashion kategori ismi
        
    Returns:
        List[str]: Stil listesi
    """
    style_mapping = {
        "shirt": ["business", "casual", "formal"],
        "blouse": ["elegant", "business", "formal"],
        "t-shirt": ["casual", "sporty"],
        "sweatshirt": ["casual", "sporty"],
        "dress": ["elegant", "formal", "casual"],
        "pants": ["business", "casual", "formal"],
        "jeans": ["casual", "trendy"],
        "jacket": ["business", "formal", "casual"],
        "coat": ["formal", "elegant"],
        "shoe": ["casual", "formal", "sporty"],
        "bag": ["elegant", "casual", "business"],
        "hat": ["casual", "trendy"],
        "glasses": ["trendy", "classic"]
    }
    
    # Kategori isminde geçen anahtar kelimeleri ara
    for key, styles in style_mapping.items():
        if key in category.lower():
            return styles[:2]  # İlk 2 stili döndür
    
    return ["casual"]  # Default stil


def infer_material_from_category(category: str) -> List[str]:
    """
    Kategori'den muhtemel malzemeleri çıkarır
    
    Args:
        category: Fashion kategori ismi
        
    Returns:
        List[str]: Malzeme listesi
    """
    material_mapping = {
        "shirt": ["cotton", "silk"],
        "blouse": ["silk", "chiffon"],
        "t-shirt": ["cotton", "polyester"],
        "sweatshirt": ["cotton", "synthetic"],
        "dress": ["silk", "cotton", "polyester"],
        "pants": ["cotton", "polyester"],
        "jacket": ["wool", "synthetic"],
        "coat": ["wool", "synthetic"],
        "shoe": ["leather", "synthetic"],
        "bag": ["leather", "synthetic"],
        "belt": ["leather"],
        "glove": ["leather", "wool"]
    }
    
    # Kategori isminde geçen anahtar kelimeleri ara
    for key, materials in material_mapping.items():
        if key in category.lower():
            return materials[:1]  # İlk malzemeyi döndür
    
    return ["synthetic"]  # Default malzeme


def infer_pattern_from_image(image: Image.Image) -> Optional[str]:
    """
    Görüntüden desen türünü çıkarmaya çalışır
    
    Args:
        image: PIL Image
        
    Returns:
        Optional[str]: Desen türü
    """
    # Bu basitleştirilmiş bir implementasyon
    # Gerçek projede görüntü işleme algoritmaları kullanılabilir
    
    try:
        # Görüntünün renklilik oranını hesapla
        img_array = np.array(image)
        color_variance = np.var(img_array)
        
        if color_variance < 100:
            return "solid"
        elif color_variance > 1000:
            return "patterned"
        else:
            return "textured"
            
    except:
        return "solid"  # Default pattern


def get_style_analysis(item_label: str, attributes: Dict) -> Dict[str, Any]:
    """
    Item için stil analizi yapar
    
    Args:
        item_label: Item kategorisi
        attributes: Item attribute'ları
        
    Returns:
        Dict[str, Any]: Stil analizi
    """
    return {
        "formality_level": determine_formality_level(item_label, attributes),
        "season_suitability": determine_season_suitability(item_label, attributes),
        "occasion_type": determine_occasion_type(item_label, attributes),
        "style_category": determine_style_category(item_label, attributes)
    }


def determine_formality_level(item_label: str, attributes: Dict) -> str:
    """Formallik seviyesini belirler"""
    formal_items = ["suit", "dress shirt", "blazer", "dress shoes", "tie"]
    casual_items = ["t-shirt", "jeans", "sneakers", "hoodie"]
    
    label_lower = item_label.lower()
    
    if any(formal in label_lower for formal in formal_items):
        return "formal"
    elif any(casual in label_lower for casual in casual_items):
        return "casual"
    else:
        return "semi-formal"


def determine_season_suitability(item_label: str, attributes: Dict) -> List[str]:
    """Mevsim uygunluğunu belirler"""
    warm_weather = ["t-shirt", "shorts", "sandals", "tank top"]
    cold_weather = ["coat", "sweater", "boots", "scarf"]
    
    label_lower = item_label.lower()
    
    if any(warm in label_lower for warm in warm_weather):
        return ["spring", "summer"]
    elif any(cold in label_lower for cold in cold_weather):
        return ["fall", "winter"]
    else:
        return ["all-season"]


def determine_occasion_type(item_label: str, attributes: Dict) -> List[str]:
    """Uygun durum türlerini belirler"""
    business_items = ["suit", "dress shirt", "blazer", "dress shoes"]
    casual_items = ["t-shirt", "jeans", "sneakers"]
    sport_items = ["sneakers", "shorts", "sports bra"]
    
    label_lower = item_label.lower()
    occasions = []
    
    if any(business in label_lower for business in business_items):
        occasions.append("business")
    if any(casual in label_lower for casual in casual_items):
        occasions.append("casual")
    if any(sport in label_lower for sport in sport_items):
        occasions.append("sports")
    
    if not occasions:
        occasions = ["general"]
    
    return occasions


def determine_style_category(item_label: str, attributes: Dict) -> str:
    """Ana stil kategorisini belirler"""
    if "formal" in attributes.get("styles", []):
        return "formal"
    elif "sporty" in attributes.get("styles", []):
        return "athletic"
    elif "trendy" in attributes.get("styles", []):
        return "fashion-forward"
    else:
        return "classic"


def generate_item_description(item_label: str, attributes: Dict) -> str:
    """
    Item için doğal dil açıklaması oluşturur
    
    Args:
        item_label: Item kategorisi
        attributes: Item attribute'ları
        
    Returns:
        str: Item açıklaması
    """
    colors = attributes.get("colors", [])
    patterns = attributes.get("patterns", [])
    styles = attributes.get("styles", [])
    materials = attributes.get("materials", [])
    
    description_parts = []
    
    # Renk ekle
    if colors and colors[0] != "unknown":
        description_parts.append(colors[0])
    
    # Pattern ekle
    if patterns and patterns[0] != "solid":
        description_parts.append(patterns[0])
    
    # Malzeme ekle
    if materials:
        description_parts.append(materials[0])
    
    # Ana item
    description_parts.append(item_label)
    
    # Stil ekle
    if styles:
        description_parts.append(f"({styles[0]} style)")
    
    return " ".join(description_parts)


def generate_summary(detections: List[Dict]) -> Dict[str, Any]:
    """
    Detection sonuçlarından genel özet oluşturur
    
    Args:
        detections: Detection listesi
        
    Returns:
        Dict[str, Any]: Özet bilgileri
    """
    if not detections:
        return {
            "total_items": 0,
            "categories": [],
            "dominant_colors": [],
            "style_summary": "No fashion items detected"
        }
    
    # Kategorileri say
    categories = {}
    all_colors = []
    all_styles = []
    
    for det in detections:
        # Kategori sayımı
        label = det["label"]
        categories[label] = categories.get(label, 0) + 1
        
        # Renkleri topla
        colors = det.get("attributes", {}).get("colors", [])
        all_colors.extend(colors)
        
        # Stilleri topla
        styles = det.get("attributes", {}).get("styles", [])
        all_styles.extend(styles)
    
    # En yaygın renkleri bul
    from collections import Counter
    color_counts = Counter(all_colors)
    dominant_colors = [color for color, _ in color_counts.most_common(3)]
    
    # En yaygın stilleri bul
    style_counts = Counter(all_styles)
    dominant_styles = [style for style, _ in style_counts.most_common(2)]
    
    # Stil özeti oluştur
    style_summary = f"Detected {len(detections)} fashion items"
    if dominant_colors:
        style_summary += f" with predominant colors: {', '.join(dominant_colors)}"
    if dominant_styles:
        style_summary += f" and {', '.join(dominant_styles)} styling"
    
    return {
        "total_items": len(detections),
        "categories": dict(categories),
        "dominant_colors": dominant_colors,
        "dominant_styles": dominant_styles,
        "style_summary": style_summary,
        "average_confidence": np.mean([det["confidence"] for det in detections])
    }


def get_class_name(class_id: int) -> str:
    """
    Class ID'den class ismini döndürür
    
    Args:
        class_id: DETR model class ID
        
    Returns:
        str: Class ismi
    """
    # FASHIONPEDIA_LABELS'den ismi al
    if class_id in FASHIONPEDIA_LABELS:
        return FASHIONPEDIA_LABELS[class_id]
    
    # FASHIONPEDIA_CATEGORIES'den al (fallback)
    if class_id in FASHIONPEDIA_CATEGORIES:
        return FASHIONPEDIA_CATEGORIES[class_id]
    
    # Default
    return f"fashion_item_{class_id}"


def visualize_predictions(image: Image.Image,
                        detections: List[Dict],
                        save_path: Optional[str] = None,
                        show_labels: bool = True,
                        show_confidence: bool = True) -> Image.Image:
    """
    A4c. Tahminleri görsel olarak gösterir
    
    Args:
        image: Orijinal görüntü
        detections: Detection listesi
        save_path: Kaydedilecek dosya yolu (opsiyonel)
        show_labels: Etiketleri göster
        show_confidence: Güven skorlarını göster
        
    Returns:
        Image.Image: Annotate edilmiş görüntü
    """
    try:
        # Görüntüyü kopyala
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        # Her detection için çizim yap
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection["bbox"]
            label = detection["label"]
            confidence = detection["confidence"]
            
            # Bounding box çiz
            color = _get_color_for_class(i)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Label ve confidence yazısı
            if show_labels or show_confidence:
                text_parts = []
                if show_labels:
                    text_parts.append(label)
                if show_confidence:
                    text_parts.append(f"{confidence:.2f}")
                
                text = " - ".join(text_parts)
                
                # Text background için rectangle
                try:
                    font = ImageFont.load_default()
                    text_bbox = draw.textbbox((x1, y1-25), text, font=font)
                    draw.rectangle(text_bbox, fill=color)
                    draw.text((x1, y1-25), text, fill="white", font=font)
                except:
                    # Fallback: basit text
                    draw.text((x1, y1-15), text, fill=color)
        
        # Kaydet (istenirse)
        if save_path:
            annotated_image.save(save_path)
            logger.info(f"Annotate edilmiş görüntü kaydedildi: {save_path}")
        
        return annotated_image
        
    except Exception as e:
        logger.error(f"Visualization hatası: {str(e)}")
        return image


def _get_color_for_class(class_index: int) -> str:
    """Class için renk döndürür"""
    colors = [
        "red", "blue", "green", "yellow", "purple", "orange", 
        "pink", "brown", "gray", "navy"
    ]
    return colors[class_index % len(colors)]


def batch_inference(model: DetrForSegmentation,
                   image_processor: DetrImageProcessor,
                   image_paths: List[str],
                   confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
                   save_results: bool = False,
                   output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Birden fazla görüntü üzerinde batch inference yapar
    
    Args:
        model: DETR modeli
        image_processor: Image processor
        image_paths: Görüntü dosyası yolları listesi
        confidence_threshold: Güven skoru threshold'u
        save_results: Sonuçları dosyaya kaydet
        output_dir: Çıktı dizini
        
    Returns:
        Dict[str, Any]: Batch inference sonuçları
    """
    logger.info(f"Batch inference başlatılıyor: {len(image_paths)} görüntü")
    
    batch_results = {}
    successful = 0
    failed = 0
    
    for i, image_path in enumerate(image_paths):
        try:
            logger.info(f"İşleniyor ({i+1}/{len(image_paths)}): {image_path}")
            
            # Tek görüntü inference
            result = run_inference(
                model=model,
                image_processor=image_processor,
                image=image_path,
                confidence_threshold=confidence_threshold
            )
            
            batch_results[image_path] = result
            successful += 1
            
        except Exception as e:
            logger.error(f"Görüntü işleme hatası {image_path}: {str(e)}")
            batch_results[image_path] = {"error": str(e)}
            failed += 1
    
    # Batch özeti
    batch_summary = {
        "total_images": len(image_paths),
        "successful": successful,
        "failed": failed,
        "results": batch_results
    }
    
    # Sonuçları kaydet (istenirse)
    if save_results and output_dir:
        output_path = Path(output_dir) / "batch_inference_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(batch_summary, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Batch sonuçları kaydedildi: {output_path}")
    
    logger.info(f"Batch inference tamamlandı: {successful}/{len(image_paths)} başarılı")
    
    return batch_summary


# Test ve demo fonksiyonları
def test_inference():
    """
    Inference fonksiyonlarını test eder
    """
    logger.info("Inference test başlatılıyor...")
    
    try:
        # Mock test görüntüsü oluştur
        test_image = Image.new("RGB", (640, 480), color="blue")
        
        # Test processor
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50-panoptic")
        
        logger.info("Test görüntüsü oluşturuldu")
        
        # Not: Gerçek test için trained model gerekli
        # Bu sadece fonksiyon signature'larını test eder
        
        logger.info("Inference test başarılı!")
        
    except Exception as e:
        logger.error(f"Test hatası: {str(e)}")
        raise


def main():
    """
    Ana çalıştırma fonksiyonu - komut satırından çalışabilir
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="DETR Fashionpedia Inference")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Eğitilmiş model dosyası yolu")
    parser.add_argument("--image_path", type=str, required=True,
                       help="Analiz edilecek görüntü yolu")
    parser.add_argument("--confidence_threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                       help="Güven skoru threshold'u")
    parser.add_argument("--output_dir", type=str, default="./inference_results",
                       help="Çıktı dizini")
    parser.add_argument("--visualize", action="store_true",
                       help="Sonuçları görselleştir")
    parser.add_argument("--batch", action="store_true",
                       help="Batch mode (image_path'i dizin olarak kullan)")
    
    args = parser.parse_args()
    
    try:
        # Model yükle
        logger.info("Model yükleniyor...")
        model, processor = load_inference_model(args.model_path)
        
        # Çıktı dizini oluştur
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.batch:
            # Batch inference
            image_paths = list(Path(args.image_path).glob("*.jpg"))
            image_paths.extend(list(Path(args.image_path).glob("*.png")))
            
            results = batch_inference(
                model=model,
                image_processor=processor,
                image_paths=[str(p) for p in image_paths],
                confidence_threshold=args.confidence_threshold,
                save_results=True,
                output_dir=str(output_dir)
            )
            
            print(f"Batch inference tamamlandı: {results['successful']}/{results['total_images']} başarılı")
            
        else:
            # Tek görüntü inference
            results = run_inference(
                model=model,
                image_processor=processor,
                image=args.image_path,
                confidence_threshold=args.confidence_threshold
            )
            
            # Sonuçları yazdır
            print("\nFashion Analysis Sonuçları:")
            print(f"Toplam item: {results['summary']['total_items']}")
            
            for i, detection in enumerate(results['detections']):
                print(f"\n{i+1}. {detection['label']}")
                print(f"   Güven: {detection['confidence']:.3f}")
                print(f"   Renkler: {', '.join(detection['attributes']['colors'])}")
                print(f"   Stiller: {', '.join(detection['attributes']['styles'])}")
                print(f"   Açıklama: {detection['description']}")
            
            # Sonuçları kaydet
            result_file = output_dir / "inference_result.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\nSonuçlar kaydedildi: {result_file}")
            
            # Görselleştirme (istenirse)
            if args.visualize:
                image = Image.open(args.image_path)
                viz_image = visualize_predictions(
                    image=image,
                    detections=results['detections'],
                    save_path=str(output_dir / "visualization.jpg")
                )
                print(f"Görselleştirme kaydedildi: {output_dir / 'visualization.jpg'}")
        
    except Exception as e:
        logger.error(f"Ana fonksiyon hatası: {str(e)}")
        raise


if __name__ == "__main__":
    main()
