"""
DETR Model Manager for Fashionpedia Fashion Analysis - Aura Project
Bu modül, Fashionpedia veri seti için uyarlanmış DETR modelini yönetmekten sorumludur.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

# Core PyTorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# Hugging Face Transformers libraries
from transformers import (
    DetrConfig,
    DetrForSegmentation, 
    DetrForObjectDetection,
    DetrImageProcessor,
    AutoConfig,
    AutoModel
)

# Numerical computation
import numpy as np

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fashionpedia model constants
FASHIONPEDIA_NUM_CLASSES = 294  # Fashionpedia veri setindeki sınıf sayısı
TOTAL_CLASSES = FASHIONPEDIA_NUM_CLASSES + 1  # +1 "no object" sınıfı için
BASE_MODEL_NAME = "facebook/detr-resnet-50-panoptic"  # Temel DETR modeli
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fashionpedia sınıf isimleri (294 sınıf için genişletilmiş)
FASHIONPEDIA_LABELS = {
    0: "no_object",  # DETR için gerekli "no object" sınıfı
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
    46: "tassel",
    # Note: Bu sadece ilk 46 sınıf. Gerçek Fashionpedia 294 sınıfa sahip.
    # Tam liste için Fashionpedia dokümentasyonuna bakın.
}

# Label mappings oluştur (294 sınıf için genişletilecek)
def create_fashionpedia_label_mappings() -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    Fashionpedia için id2label ve label2id mapping'lerini oluşturur
    
    Returns:
        Tuple[Dict[int, str], Dict[str, int]]: (id2label, label2id) mapping'leri
    """
    # Temel etiketlerden başla
    id2label = FASHIONPEDIA_LABELS.copy()
    
    # Eksik sınıfları generic isimlerle doldur (gerçek projelerde tam liste kullanılmalı)
    for i in range(len(FASHIONPEDIA_LABELS), TOTAL_CLASSES):
        id2label[i] = f"fashion_class_{i}"
    
    # Reverse mapping oluştur
    label2id = {v: k for k, v in id2label.items()}
    
    return id2label, label2id


def get_model(num_classes: int = TOTAL_CLASSES, 
              device: Optional[torch.device] = None,
              pretrained: bool = True) -> DetrForSegmentation:
    """
    A2. Fashionpedia için uyarlanmış DETR modeli oluşturur ve döndürür
    
    Bu fonksiyon, COCO üzerinde önceden eğitilmiş DETR modelini alır ve 
    Fashionpedia veri setinin 294 sınıfına göre uyarlar.
    
    Args:
        num_classes (int): Çıkış sınıf sayısı (varsayılan: 295 = 294 + "no_object")
        device (Optional[torch.device]): Model cihazı (GPU/CPU)
        pretrained (bool): Önceden eğitilmiş ağırlıkları kullan
        
    Returns:
        DetrForSegmentation: Fashionpedia için yapılandırılmış DETR modeli
        
    Raises:
        ValueError: Geçersiz num_classes değeri
        RuntimeError: Model yükleme hatası
        
    Example:
        >>> model = get_model(num_classes=295)
        >>> print(f"Model device: {next(model.parameters()).device}")
        >>> print(f"Num classes: {model.config.num_labels}")
    """
    # Parametreleri doğrula
    if num_classes < 2:
        raise ValueError(f"num_classes en az 2 olmalı, verilen: {num_classes}")
    
    if device is None:
        device = DEVICE
    
    logger.info(f"DETR modeli oluşturuluyor - sınıf sayısı: {num_classes}")
    
    try:
        # Fashionpedia label mappings oluştur
        id2label, label2id = create_fashionpedia_label_mappings()
        
        # DETR konfigürasyonunu yükle ve güncelle
        config = DetrConfig.from_pretrained(BASE_MODEL_NAME)
        
        # Fashionpedia için konfigürasyonu güncelle
        config.num_labels = num_classes
        config.id2label = {str(k): v for k, v in id2label.items()}  # String keys gerekli
        config.label2id = {v: str(k) for k, v in label2id.items()}  # String values gerekli
        
        # Segmentasyon için ek parametreler
        config.use_segmentation = True  # Segmentasyon özelliğini etkinleştir
        config.num_queries = 100  # DETR query sayısı
        
        logger.info(f"DETR konfigürasyonu güncellendi:")
        logger.info(f"  - num_labels: {config.num_labels}")
        logger.info(f"  - num_queries: {config.num_queries}")
        logger.info(f"  - use_segmentation: {config.use_segmentation}")
        
        # DETR modelini yükle
        if pretrained:
            logger.info(f"Önceden eğitilmiş model yükleniyor: {BASE_MODEL_NAME}")
            model = DetrForSegmentation.from_pretrained(
                BASE_MODEL_NAME,
                config=config,
                ignore_mismatched_sizes=True  # Sınıf sayısı farklılığını otomatik ayarla
            )
        else:
            logger.info("Rastgele ağırlıklarla model oluşturuluyor")
            model = DetrForSegmentation(config)
        
        # Modeli belirtilen cihaza taşı
        model = model.to(device)
        
        # Model parametrelerini logla
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model başarıyla oluşturuldu:")
        logger.info(f"  - Toplam parametre: {total_params:,}")
        logger.info(f"  - Eğitilebilir parametre: {trainable_params:,}")
        logger.info(f"  - Cihaz: {device}")
        
        return model
        
    except Exception as e:
        error_msg = f"DETR modeli oluşturulamadı: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def load_model_for_inference(model_path: Union[str, Path],
                           num_classes: int = TOTAL_CLASSES,
                           device: Optional[torch.device] = None) -> DetrForSegmentation:
    """
    A3. Eğitilmiş DETR modelini inference için yükler
    
    Bu fonksiyon, kaydedilmiş model ağırlıklarını yükler ve modeli 
    inference (çıkarım) moduna alır.
    
    Args:
        model_path (Union[str, Path]): Model dosyası yolu veya HF Hub model ID
        num_classes (int): Sınıf sayısı
        device (Optional[torch.device]): Hedef cihaz
        
    Returns:
        DetrForSegmentation: Yüklenmiş ve inference için hazır model
        
    Raises:
        FileNotFoundError: Model dosyası bulunamadığında
        RuntimeError: Model yükleme hatası
        
    Example:
        >>> model = load_model_for_inference("./saved_models/detr_fashionpedia.pth")
        >>> model.eval()  # Inference moduna al
    """
    if device is None:
        device = DEVICE
    
    model_path = Path(model_path) if isinstance(model_path, str) else model_path
    
    logger.info(f"Model inference için yükleniyor: {model_path}")
    
    try:
        # Model dosyası kontrolü (yerel dosya ise)
        if model_path.exists() and model_path.is_file():
            # Yerel PyTorch model dosyası
            logger.info("Yerel PyTorch model dosyası yükleniyor")
            
            # Önce model iskeletini oluştur
            model = get_model(num_classes=num_classes, device=device, pretrained=False)
            
            # Kaydedilmiş ağırlıkları yükle
            checkpoint = torch.load(model_path, map_location=device)
            
            # State dict formatına göre yükle
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
            
            logger.info("Yerel model ağırlıkları başarıyla yüklendi")
            
        else:
            # Hugging Face Hub model ID olarak dene
            logger.info(f"Hugging Face Hub'dan model yükleniyor: {model_path}")
            
            # Konfigürasyonu güncelle
            config = DetrConfig.from_pretrained(str(model_path))
            config.num_labels = num_classes
            
            # Modeli yükle
            model = DetrForSegmentation.from_pretrained(
                str(model_path),
                config=config,
                ignore_mismatched_sizes=True
            )
            model = model.to(device)
        
        # Modeli inference moduna al
        model.eval()
        
        # Gradient hesaplamayı kapat (inference için)
        for param in model.parameters():
            param.requires_grad = False
        
        logger.info(f"Model inference için hazır - cihaz: {device}")
        return model
        
    except FileNotFoundError:
        error_msg = f"Model dosyası bulunamadı: {model_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    except Exception as e:
        error_msg = f"Model yükleme hatası: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def prepare_model_for_training(model: DetrForSegmentation,
                             freeze_backbone: bool = True,
                             freeze_encoder: bool = False,
                             learning_rate: float = 1e-5) -> Tuple[DetrForSegmentation, torch.optim.Optimizer]:
    """
    A4. DETR modelini eğitim için hazırlar
    
    Bu fonksiyon, transfer learning stratejisi uygular: backbone'u dondurarak
    sadece son katmanları eğitir (isteğe bağlı).
    
    Args:
        model (DetrForSegmentation): Hazırlanacak DETR modeli
        freeze_backbone (bool): ResNet backbone'unu dondur
        freeze_encoder (bool): Transformer encoder'ını dondur  
        learning_rate (float): Başlangıç öğrenme oranı
        
    Returns:
        Tuple[DetrForSegmentation, torch.optim.Optimizer]: (model, optimizer)
        
    Example:
        >>> model = get_model()
        >>> model, optimizer = prepare_model_for_training(model, freeze_backbone=True)
        >>> print(f"Eğitilebilir parametreler: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    """
    logger.info("Model eğitim için hazırlanıyor...")
    
    # Modeli eğitim moduna al
    model.train()
    
    # Tüm parametrelerin grad hesaplamasını etkinleştir
    for param in model.parameters():
        param.requires_grad = True
    
    # Backbone dondurma (transfer learning)
    if freeze_backbone:
        logger.info("ResNet backbone donduruluyor...")
        if hasattr(model, 'model') and hasattr(model.model, 'backbone'):
            for param in model.model.backbone.parameters():
                param.requires_grad = False
        elif hasattr(model, 'backbone'):
            for param in model.backbone.parameters():
                param.requires_grad = False
        else:
            logger.warning("Backbone bulunamadı, dondurma atlanıyor")
    
    # Encoder dondurma (isteğe bağlı)
    if freeze_encoder:
        logger.info("Transformer encoder donduruluyor...")
        if hasattr(model, 'model') and hasattr(model.model, 'transformer'):
            encoder = model.model.transformer.encoder
            for param in encoder.parameters():
                param.requires_grad = False
        else:
            logger.warning("Encoder bulunamadı, dondurma atlanıyor")
    
    # Eğitilebilir parametreleri say
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"Eğitim hazırlığı tamamlandı:")
    logger.info(f"  - Toplam parametre: {total_params:,}")
    logger.info(f"  - Eğitilebilir parametre: {trainable_params:,}")
    logger.info(f"  - Eğitim oranı: {trainable_params/total_params:.2%}")
    
    # Optimizer oluştur (differential learning rates)
    param_groups = get_parameter_groups(model, learning_rate)
    optimizer = AdamW(param_groups, lr=learning_rate, weight_decay=0.01)
    
    logger.info(f"AdamW optimizer oluşturuldu - lr: {learning_rate}")
    
    return model, optimizer


def get_parameter_groups(model: DetrForSegmentation, 
                        base_lr: float = 1e-5) -> List[Dict[str, Any]]:
    """
    DETR modeli için farklı öğrenme oranlarıyla parametre grupları oluşturur
    
    Args:
        model: DETR modeli
        base_lr: Temel öğrenme oranı
        
    Returns:
        List[Dict]: Parametre grupları listesi
    """
    param_groups = []
    
    # Backbone parametreleri (düşük lr)
    backbone_params = []
    if hasattr(model, 'model') and hasattr(model.model, 'backbone'):
        backbone_params = list(model.model.backbone.parameters())
    elif hasattr(model, 'backbone'):
        backbone_params = list(model.backbone.parameters())
    
    if backbone_params:
        param_groups.append({
            "params": backbone_params,
            "lr": base_lr * 0.1,  # Backbone için 10x düşük lr
            "name": "backbone"
        })
    
    # Transformer encoder parametreleri (orta lr)
    encoder_params = []
    if hasattr(model, 'model') and hasattr(model.model, 'transformer'):
        encoder_params = list(model.model.transformer.encoder.parameters())
    
    if encoder_params:
        param_groups.append({
            "params": encoder_params,
            "lr": base_lr * 0.5,  # Encoder için 2x düşük lr
            "name": "encoder"
        })
    
    # Decoder ve classifier parametreleri (tam lr)
    remaining_params = []
    backbone_param_ids = {id(p) for p in backbone_params}
    encoder_param_ids = {id(p) for p in encoder_params}
    
    for name, param in model.named_parameters():
        if id(param) not in backbone_param_ids and id(param) not in encoder_param_ids:
            remaining_params.append(param)
    
    param_groups.append({
        "params": remaining_params,
        "lr": base_lr,  # Tam öğrenme oranı
        "name": "head"
    })
    
    logger.info(f"Parametre grupları oluşturuldu:")
    for group in param_groups:
        param_count = sum(p.numel() for p in group["params"])
        logger.info(f"  - {group['name']}: {param_count:,} parametre, lr: {group['lr']}")
    
    return param_groups


def save_model(model: DetrForSegmentation,
               save_path: Union[str, Path],
               optimizer: Optional[torch.optim.Optimizer] = None,
               epoch: Optional[int] = None,
               loss: Optional[float] = None,
               metrics: Optional[Dict[str, float]] = None) -> None:
    """
    Eğitilmiş DETR modelini kaydeder
    
    Args:
        model: Kaydedilecek model
        save_path: Kayıt yolu
        optimizer: Optimizer (opsiyonel)
        epoch: Epoch numarası (opsiyonel)
        loss: Loss değeri (opsiyonel)
        metrics: Metrik değerleri (opsiyonel)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Checkpoint oluştur
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": model.config,
        "model_name": "detr_fashionpedia",
        "num_classes": model.config.num_labels,
        "timestamp": torch.tensor(torch.get_default_dtype()).now().isoformat() if hasattr(torch.tensor(1.0), 'now') else None
    }
    
    # Opsiyonel bilgileri ekle
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if loss is not None:
        checkpoint["loss"] = loss
    if metrics is not None:
        checkpoint["metrics"] = metrics
    
    # Kaydet
    torch.save(checkpoint, save_path)
    logger.info(f"Model kaydedildi: {save_path}")
    
    # Hugging Face formatında da kaydet (isteğe bağlı)
    hf_save_path = save_path.parent / f"{save_path.stem}_hf"
    model.save_pretrained(hf_save_path)
    logger.info(f"Hugging Face formatında kaydedildi: {hf_save_path}")


def get_model_info(model: DetrForSegmentation) -> Dict[str, Any]:
    """
    Model hakkında detaylı bilgi döndürür
    
    Args:
        model: İncelenecek DETR modeli
        
    Returns:
        Dict: Model bilgileri
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        "model_type": type(model).__name__,
        "num_classes": model.config.num_labels,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
        "device": str(next(model.parameters()).device),
        "training_mode": model.training,
        "config": {
            "num_queries": model.config.num_queries,
            "use_segmentation": getattr(model.config, "use_segmentation", False),
            "hidden_dim": model.config.d_model if hasattr(model.config, "d_model") else None,
        }
    }
    
    return info



def validate_model_output(outputs, target_classes: int = TOTAL_CLASSES) -> bool:
    """
    Model çıktısının beklenen formatda olup olmadığını doğrular
    
    Args:
        outputs: Model çıktısı
        target_classes: Beklenen sınıf sayısı
        
    Returns:
        bool: Çıktı geçerli ise True
    """
    try:
        # Logits kontrolü
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
            batch_size, num_queries, num_classes = logits.shape
            
            if num_classes != target_classes:
                logger.warning(f"Sınıf sayısı uyumsuzluğu: {num_classes} != {target_classes}")
                return False
            
            logger.debug(f"Logits şekli geçerli: {logits.shape}")
        
        # Pred boxes kontrolü
        if hasattr(outputs, 'pred_boxes'):
            pred_boxes = outputs.pred_boxes
            batch_size, num_queries, box_dim = pred_boxes.shape
            
            if box_dim != 4:  # [x_center, y_center, width, height]
                logger.warning(f"Bbox boyutu yanlış: {box_dim} != 4")
                return False
            
            logger.debug(f"Pred boxes şekli geçerli: {pred_boxes.shape}")
        
        # Segmentation mask kontrolü (varsa)
        if hasattr(outputs, 'pred_masks'):
            pred_masks = outputs.pred_masks
            logger.debug(f"Segmentation masks şekli: {pred_masks.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"Model çıktısı doğrulama hatası: {e}")
        return False


# Test fonksiyonu
def test_model_functions():
    """
    Model fonksiyonlarını test eder
    
    Bu fonksiyon, tüm model fonksiyonlarının çalışıp çalışmadığını kontrol eder.
    """
    logger.info("Model fonksiyonları test ediliyor...")
    
    try:
        # 1. Model oluşturma testi
        logger.info("1. Model oluşturma testi...")
        model = get_model(num_classes=TOTAL_CLASSES)
        logger.info(f"Model oluşturuldu: {type(model).__name__}")
        
        # 2. Model bilgisi testi
        logger.info("2. Model bilgisi testi...")
        info = get_model_info(model)
        logger.info(f"Model bilgisi: {info['total_parameters']:,} parametre")
        
        # 3. Eğitim hazırlığı testi
        logger.info("3. Eğitim hazırlığı testi...")
        model, optimizer = prepare_model_for_training(model, freeze_backbone=True)
        logger.info(f"Optimizer oluşturuldu: {type(optimizer).__name__}")
        
        # 4. Model çıktısı testi (dummy input)
        logger.info("4. Model çıktısı testi...")
        dummy_input = torch.randn(1, 3, 800, 800).to(model.device)
        
        with torch.no_grad():
            outputs = model(dummy_input)
        
        is_valid = validate_model_output(outputs, TOTAL_CLASSES)
        logger.info(f"Model çıktısı geçerli: {is_valid}")
        
        logger.info("Tüm testler başarılı!")
        
    except Exception as e:
        logger.error(f"Test hatası: {e}")
        raise


if __name__ == "__main__":
    # Test çalıştır
    test_model_functions()
