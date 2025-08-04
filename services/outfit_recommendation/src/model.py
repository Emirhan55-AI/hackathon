"""
OutfitTransformer Model Implementation for Fashion Outfit Recommendation - Aura Project
Bu modül, outfit önerisi için OutfitTransformer modelini tanımlar ve yönetir.
"""

import os
import json
import logging
import math
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

# Core libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.optim as optim

# Vision models
import torchvision.models as models
from torchvision.models import ResNet50_Weights

# Hugging Face models
from transformers import (
    AutoModel, 
    AutoConfig,
    PreTrainedModel,
    PretrainedConfig
)

# Numerical computation
import numpy as np

# Local imports
from data_loader import (
    FASHION_CATEGORY_HIERARCHY,
    COMPATIBILITY_RULES,
    MAX_ITEMS_PER_OUTFIT,
    IMAGE_SIZE
)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model constants
OUTFIT_TRANSFORMER_CONFIG = {
    "hidden_size": 512,
    "num_attention_heads": 8,
    "num_transformer_layers": 6,
    "intermediate_size": 2048,
    "dropout_prob": 0.1,
    "layer_norm_eps": 1e-12,
    "max_items_per_outfit": MAX_ITEMS_PER_OUTFIT,
    "image_feature_dim": 2048,  # ResNet50 feature dimension
    "category_vocab_size": 100,  # Fashion kategori sayısı
    "color_vocab_size": 50,     # Renk sayısı
    "style_vocab_size": 30,     # Stil sayısı
    "compatibility_classes": 2   # Compatible/Incompatible
}

# Kategori ve renk mappings
CATEGORY_TO_ID = {}
COLOR_TO_ID = {}
STYLE_TO_ID = {}

# Initialize mappings
def _initialize_vocabulary():
    """Fashion vocabulary'lerini initialize eder"""
    global CATEGORY_TO_ID, COLOR_TO_ID, STYLE_TO_ID
    
    # Kategori mapping'i oluştur
    all_categories = []
    for category_list in FASHION_CATEGORY_HIERARCHY.values():
        all_categories.extend(category_list)
    
    CATEGORY_TO_ID = {cat: idx for idx, cat in enumerate(set(all_categories))}
    CATEGORY_TO_ID["<UNK>"] = len(CATEGORY_TO_ID)
    
    # Renk mapping'i (basit liste)
    common_colors = [
        "black", "white", "gray", "brown", "blue", "red", "green", "yellow",
        "pink", "purple", "orange", "beige", "navy", "maroon", "olive", "gold",
        "silver", "cream", "tan", "burgundy", "turquoise", "coral"
    ]
    COLOR_TO_ID = {color: idx for idx, color in enumerate(common_colors)}
    COLOR_TO_ID["<UNK>"] = len(COLOR_TO_ID)
    
    # Stil mapping'i
    common_styles = [
        "casual", "formal", "business", "sporty", "elegant", "vintage", 
        "modern", "bohemian", "classic", "trendy", "edgy", "romantic",
        "minimalist", "street", "preppy", "punk", "gothic", "retro"
    ]
    STYLE_TO_ID = {style: idx for idx, style in enumerate(common_styles)}
    STYLE_TO_ID["<UNK>"] = len(STYLE_TO_ID)

# Initialize vocabulary at module import
_initialize_vocabulary()


class OutfitTransformerConfig(PretrainedConfig):
    """
    OutfitTransformer model konfigürasyonu
    
    Bu sınıf, OutfitTransformer modelinin tüm konfigürasyon parametrelerini tutar.
    """
    model_type = "outfit_transformer"
    
    def __init__(
        self,
        hidden_size: int = 512,
        num_attention_heads: int = 8,
        num_transformer_layers: int = 6,
        intermediate_size: int = 2048,
        dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
        max_items_per_outfit: int = MAX_ITEMS_PER_OUTFIT,
        image_feature_dim: int = 2048,
        category_vocab_size: int = 100,
        color_vocab_size: int = 50,
        style_vocab_size: int = 30,
        compatibility_classes: int = 2,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_transformer_layers = num_transformer_layers
        self.intermediate_size = intermediate_size
        self.dropout_prob = dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.max_items_per_outfit = max_items_per_outfit
        self.image_feature_dim = image_feature_dim
        self.category_vocab_size = category_vocab_size
        self.color_vocab_size = color_vocab_size
        self.style_vocab_size = style_vocab_size
        self.compatibility_classes = compatibility_classes


class ImageFeatureExtractor(nn.Module):
    """
    Fashion item görüntülerinden feature çıkaran modül
    
    Bu modül, ResNet50 backbone kullanarak fashion item görüntülerinden
    sabit boyutlu feature vektörleri çıkarır.
    """
    
    def __init__(self, feature_dim: int = 2048, freeze_backbone: bool = True):
        super().__init__()
        
        # ResNet50 backbone (pretrained)
        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Son FC katmanını kaldır
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Backbone'u dondur (eğer istenirse)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(2048, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.feature_dim = feature_dim
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Görüntülerden feature çıkar
        
        Args:
            images: [batch_size, num_items, 3, H, W] veya [batch_size, 3, H, W]
            
        Returns:
            torch.Tensor: [batch_size, num_items, feature_dim] veya [batch_size, feature_dim]
        """
        original_shape = images.shape
        
        # Eğer 5D ise (batch_size, num_items, 3, H, W) -> (batch_size * num_items, 3, H, W)
        if len(original_shape) == 5:
            batch_size, num_items = original_shape[:2]
            images = images.view(-1, *original_shape[2:])  # [B*N, 3, H, W]
        else:
            batch_size, num_items = original_shape[0], 1
        
        # ResNet features
        with torch.no_grad() if hasattr(self, '_freeze_backbone') else torch.enable_grad():
            features = self.backbone(images)  # [B*N, 2048, 1, 1]
            features = features.squeeze(-1).squeeze(-1)  # [B*N, 2048]
        
        # Feature projection
        features = self.feature_projection(features)  # [B*N, feature_dim]
        
        # Reshape back
        if len(original_shape) == 5:
            features = features.view(batch_size, num_items, self.feature_dim)
        
        return features


class FashionAttributeEmbedding(nn.Module):
    """
    Fashion attribute'larını (kategori, renk, stil) embed eden modül
    
    Bu modül, kategorik fashion attribute'larını dense embedding'lere dönüştürür.
    """
    
    def __init__(self, 
                 category_vocab_size: int,
                 color_vocab_size: int,
                 style_vocab_size: int,
                 embedding_dim: int = 128):
        super().__init__()
        
        self.category_embedding = nn.Embedding(category_vocab_size, embedding_dim)
        self.color_embedding = nn.Embedding(color_vocab_size, embedding_dim)
        self.style_embedding = nn.Embedding(style_vocab_size, embedding_dim)
        
        # Attribute fusion
        self.attribute_fusion = nn.Sequential(
            nn.Linear(3 * embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.embedding_dim = embedding_dim
    
    def forward(self, 
                category_ids: torch.Tensor,
                color_ids: torch.Tensor, 
                style_ids: torch.Tensor) -> torch.Tensor:
        """
        Attribute ID'lerinden embedding çıkar
        
        Args:
            category_ids: [batch_size, num_items]
            color_ids: [batch_size, num_items]
            style_ids: [batch_size, num_items]
            
        Returns:
            torch.Tensor: [batch_size, num_items, embedding_dim]
        """
        # Embeddings
        cat_emb = self.category_embedding(category_ids)  # [B, N, D]
        color_emb = self.color_embedding(color_ids)      # [B, N, D]
        style_emb = self.style_embedding(style_ids)      # [B, N, D]
        
        # Concatenate ve fuse
        combined = torch.cat([cat_emb, color_emb, style_emb], dim=-1)  # [B, N, 3*D]
        fused = self.attribute_fusion(combined)  # [B, N, D]
        
        return fused


class PositionalEncoding(nn.Module):
    """
    Transformer için positional encoding
    
    Outfit içindeki item'ların sırasını modelin öğrenmesi için positional encoding ekler.
    """
    
    def __init__(self, d_model: int, max_len: int = MAX_ITEMS_PER_OUTFIT):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Positional encoding ekle
        
        Args:
            x: [batch_size, seq_len, d_model]
            
        Returns:
            torch.Tensor: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class OutfitTransformer(PreTrainedModel):
    """
    Ana OutfitTransformer modeli
    
    Bu model, fashion item'lar arasındaki uyumluluğu öğrenir ve
    outfit önerileri yapar. Transformer mimarisi kullanır.
    """
    config_class = OutfitTransformerConfig
    
    def __init__(self, config: OutfitTransformerConfig):
        super().__init__(config)
        
        self.config = config
        
        # Image feature extractor
        self.image_extractor = ImageFeatureExtractor(
            feature_dim=config.image_feature_dim
        )
        
        # Fashion attribute embeddings
        self.attribute_embedding = FashionAttributeEmbedding(
            category_vocab_size=config.category_vocab_size,
            color_vocab_size=config.color_vocab_size,
            style_vocab_size=config.style_vocab_size,
            embedding_dim=128
        )
        
        # Feature fusion (image + attributes)
        self.feature_fusion = nn.Sequential(
            nn.Linear(config.image_feature_dim + 128, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_prob),
            nn.LayerNorm(config.hidden_size)
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=config.hidden_size,
            max_len=config.max_items_per_outfit
        )
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout_prob,
            activation='relu',
            layer_norm_eps=config.layer_norm_eps,
            batch_first=True
        )
        
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_transformer_layers
        )
        
        # Output heads
        self.compatibility_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_prob),
            nn.Linear(config.hidden_size // 2, config.compatibility_classes)
        )
        
        self.outfit_score_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_prob),
            nn.Linear(config.hidden_size // 2, 1)
        )
        
        # Item recommendation head (for fill-in-the-blank)
        self.item_recommendation_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_prob),
            nn.Linear(config.hidden_size, config.image_feature_dim)
        )
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Model ağırlıklarını initialize et"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self,
                item_images: torch.Tensor,
                category_ids: torch.Tensor,
                color_ids: torch.Tensor,
                style_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            item_images: [batch_size, max_items, 3, H, W]
            category_ids: [batch_size, max_items]
            color_ids: [batch_size, max_items]
            style_ids: [batch_size, max_items]
            attention_mask: [batch_size, max_items]
            labels: [batch_size] - outfit compatibility labels
            
        Returns:
            Dict[str, torch.Tensor]: Model outputs
        """
        batch_size, max_items = item_images.shape[:2]
        
        # 1. Extract image features
        image_features = self.image_extractor(item_images)  # [B, N, image_feature_dim]
        
        # 2. Extract attribute embeddings
        attr_features = self.attribute_embedding(
            category_ids, color_ids, style_ids
        )  # [B, N, 128]
        
        # 3. Fuse image and attribute features
        combined_features = torch.cat([image_features, attr_features], dim=-1)
        fused_features = self.feature_fusion(combined_features)  # [B, N, hidden_size]
        
        # 4. Add positional encoding
        encoded_features = self.positional_encoding(fused_features)
        
        # 5. Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, max_items, device=item_images.device)
        
        # Transformer encoder attention mask (True = masked)
        encoder_attention_mask = (attention_mask == 0)
        
        # 6. Transformer encoding
        transformer_output = self.transformer_encoder(
            encoded_features,
            src_key_padding_mask=encoder_attention_mask
        )  # [B, N, hidden_size]
        
        # 7. Aggregate outfit representation (mean pooling over valid items)
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(transformer_output)
        masked_output = transformer_output * mask_expanded
        
        # Sum pooling ve normalize
        outfit_representation = masked_output.sum(dim=1)  # [B, hidden_size]
        valid_items = attention_mask.sum(dim=1, keepdim=True)  # [B, 1]
        outfit_representation = outfit_representation / (valid_items + 1e-8)
        
        # 8. Output predictions
        compatibility_logits = self.compatibility_head(outfit_representation)  # [B, 2]
        outfit_scores = self.outfit_score_head(outfit_representation).squeeze(-1)  # [B]
        
        # Item recommendation features (for each position)
        item_rec_features = self.item_recommendation_head(transformer_output)  # [B, N, image_feature_dim]
        
        outputs = {
            "compatibility_logits": compatibility_logits,
            "outfit_scores": outfit_scores,
            "item_recommendation_features": item_rec_features,
            "outfit_representation": outfit_representation,
            "transformer_output": transformer_output
        }
        
        # Compute loss if labels provided
        if labels is not None:
            compatibility_loss = F.cross_entropy(compatibility_logits, labels)
            
            # Outfit score loss (higher scores for compatible outfits)
            score_targets = labels.float()  # 0 for incompatible, 1 for compatible
            score_loss = F.mse_loss(torch.sigmoid(outfit_scores), score_targets)
            
            total_loss = compatibility_loss + 0.5 * score_loss
            
            outputs["loss"] = total_loss
            outputs["compatibility_loss"] = compatibility_loss
            outputs["score_loss"] = score_loss
        
        return outputs
    
    def get_item_compatibility_scores(self,
                                    query_item: torch.Tensor,
                                    candidate_items: torch.Tensor,
                                    query_attributes: Dict[str, torch.Tensor],
                                    candidate_attributes: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Query item ile candidate item'lar arasındaki uyumluluk skorlarını hesaplar
        
        Args:
            query_item: [3, H, W] - Query item görüntüsü
            candidate_items: [num_candidates, 3, H, W] - Candidate item görüntüleri
            query_attributes: Query item attributes
            candidate_attributes: Candidate item attributes
            
        Returns:
            torch.Tensor: [num_candidates] - Compatibility scores
        """
        num_candidates = candidate_items.shape[0]
        
        # Query item'ı her candidate ile eşleştir
        query_expanded = query_item.unsqueeze(0).repeat(num_candidates, 1, 1, 1)
        
        # Pair-wise outfits oluştur
        paired_items = torch.stack([query_expanded, candidate_items], dim=1)  # [N, 2, 3, H, W]
        
        # Attributes
        query_cat = query_attributes["category_ids"].repeat(num_candidates)
        query_color = query_attributes["color_ids"].repeat(num_candidates)
        query_style = query_attributes["style_ids"].repeat(num_candidates)
        
        cand_cat = candidate_attributes["category_ids"]
        cand_color = candidate_attributes["color_ids"]
        cand_style = candidate_attributes["style_ids"]
        
        paired_categories = torch.stack([query_cat, cand_cat], dim=1)
        paired_colors = torch.stack([query_color, cand_color], dim=1)
        paired_styles = torch.stack([query_style, cand_style], dim=1)
        
        # Model'den geçir
        with torch.no_grad():
            outputs = self.forward(
                item_images=paired_items,
                category_ids=paired_categories,
                color_ids=paired_colors,
                style_ids=paired_styles
            )
            
            compatibility_probs = F.softmax(outputs["compatibility_logits"], dim=-1)
            compatibility_scores = compatibility_probs[:, 1]  # Probability of being compatible
        
        return compatibility_scores


def create_outfit_transformer(config: Optional[OutfitTransformerConfig] = None) -> OutfitTransformer:
    """
    OutfitTransformer modeli oluşturur
    
    Args:
        config: Model konfigürasyonu
        
    Returns:
        OutfitTransformer: Initialized model
    """
    if config is None:
        config = OutfitTransformerConfig(**OUTFIT_TRANSFORMER_CONFIG)
    
    logger.info("OutfitTransformer modeli oluşturuluyor...")
    logger.info(f"Config: {config}")
    
    model = OutfitTransformer(config)
    
    # Model size hesapla
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model oluşturuldu!")
    logger.info(f"Toplam parametreler: {total_params:,}")
    logger.info(f"Eğitilebilir parametreler: {trainable_params:,}")
    
    return model


def save_model(model: OutfitTransformer, 
               save_path: Union[str, Path],
               optimizer_state: Optional[Dict] = None,
               training_metadata: Optional[Dict] = None):
    """
    OutfitTransformer modelini kaydeder
    
    Args:
        model: OutfitTransformer modeli
        save_path: Kaydedilecek dosya yolu
        optimizer_state: Optimizer state dict
        training_metadata: Eğitim metadata'sı
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Model kaydediliyor: {save_path}")
    
    # Model state
    save_dict = {
        "model_state_dict": model.state_dict(),
        "config": model.config,
        "model_type": "OutfitTransformer",
        "vocabulary_mappings": {
            "category_to_id": CATEGORY_TO_ID,
            "color_to_id": COLOR_TO_ID,
            "style_to_id": STYLE_TO_ID
        }
    }
    
    # Optimizer state ekle
    if optimizer_state is not None:
        save_dict["optimizer_state_dict"] = optimizer_state
    
    # Training metadata ekle
    if training_metadata is not None:
        save_dict["training_metadata"] = training_metadata
    
    # Save
    torch.save(save_dict, save_path)
    
    logger.info(f"Model başarıyla kaydedildi: {save_path}")


def load_model(model_path: Union[str, Path],
               device: Optional[torch.device] = None) -> Tuple[OutfitTransformer, Dict]:
    """
    Kaydedilmiş OutfitTransformer modelini yükler
    
    Args:
        model_path: Model dosyası yolu
        device: Model yüklenecek cihaz
        
    Returns:
        Tuple[OutfitTransformer, Dict]: (model, metadata)
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Model yükleniyor: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Config'i yükle
    config = checkpoint["config"]
    
    # Model'i oluştur
    model = OutfitTransformer(config)
    
    # State dict'i yükle
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Vocabulary mappings'i güncelle
    global CATEGORY_TO_ID, COLOR_TO_ID, STYLE_TO_ID
    if "vocabulary_mappings" in checkpoint:
        vocab_mappings = checkpoint["vocabulary_mappings"]
        CATEGORY_TO_ID = vocab_mappings.get("category_to_id", CATEGORY_TO_ID)
        COLOR_TO_ID = vocab_mappings.get("color_to_id", COLOR_TO_ID)
        STYLE_TO_ID = vocab_mappings.get("style_to_id", STYLE_TO_ID)
    
    # Metadata
    metadata = {
        "model_type": checkpoint.get("model_type", "OutfitTransformer"),
        "training_metadata": checkpoint.get("training_metadata", {}),
        "vocabulary_mappings": checkpoint.get("vocabulary_mappings", {})
    }
    
    logger.info(f"Model başarıyla yüklendi!")
    logger.info(f"Device: {device}")
    
    return model, metadata


def convert_attributes_to_ids(item_features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Item feature'larını model input ID'lerine dönüştürür
    
    Args:
        item_features: Item feature dictionary'leri listesi
        
    Returns:
        Dict[str, torch.Tensor]: ID tensörleri
    """
    category_ids = []
    color_ids = []
    style_ids = []
    
    for item in item_features:
        # Category
        category = item.get("category", "").lower()
        cat_id = CATEGORY_TO_ID.get(category, CATEGORY_TO_ID["<UNK>"])
        category_ids.append(cat_id)
        
        # Color
        color = item.get("color", "").lower()
        color_id = COLOR_TO_ID.get(color, COLOR_TO_ID["<UNK>"])
        color_ids.append(color_id)
        
        # Style
        style = item.get("style", "").lower()
        style_id = STYLE_TO_ID.get(style, STYLE_TO_ID["<UNK>"])
        style_ids.append(style_id)
    
    return {
        "category_ids": torch.tensor(category_ids, dtype=torch.long),
        "color_ids": torch.tensor(color_ids, dtype=torch.long),
        "style_ids": torch.tensor(style_ids, dtype=torch.long)
    }


def prepare_model_for_training(model: OutfitTransformer,
                             learning_rate: float = 1e-4,
                             weight_decay: float = 0.01) -> Tuple[OutfitTransformer, torch.optim.Optimizer]:
    """
    Modeli eğitim için hazırlar
    
    Args:
        model: OutfitTransformer modeli
        learning_rate: Öğrenme oranı
        weight_decay: Weight decay
        
    Returns:
        Tuple[OutfitTransformer, torch.optim.Optimizer]: (model, optimizer)
    """
    logger.info("Model eğitim için hazırlanıyor...")
    
    # Model'i training mode'a al
    model.train()
    
    # Optimizer oluştur
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    logger.info(f"Optimizer oluşturuldu: lr={learning_rate}, weight_decay={weight_decay}")
    
    return model, optimizer


def get_model_summary(model: OutfitTransformer) -> Dict[str, Any]:
    """
    Model özet bilgilerini döndürür
    
    Args:
        model: OutfitTransformer modeli
        
    Returns:
        Dict[str, Any]: Model özet bilgileri
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "model_type": "OutfitTransformer",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "config": model.config.to_dict(),
        "vocabulary_sizes": {
            "categories": len(CATEGORY_TO_ID),
            "colors": len(COLOR_TO_ID),
            "styles": len(STYLE_TO_ID)
        }
    }


# Test fonksiyonu
def test_outfit_transformer():
    """
    OutfitTransformer modelini test eder
    
    Bu fonksiyon, modelin temel işlevlerinin çalışıp çalışmadığını kontrol eder.
    """
    logger.info("OutfitTransformer model test başlatılıyor...")
    
    try:
        # 1. Model oluşturma testi
        logger.info("1. Model oluşturma testi...")
        model = create_outfit_transformer()
        
        # 2. Forward pass testi
        logger.info("2. Forward pass testi...")
        batch_size = 2
        max_items = 3
        
        # Dummy input oluştur
        item_images = torch.randn(batch_size, max_items, 3, IMAGE_SIZE, IMAGE_SIZE)
        category_ids = torch.randint(0, len(CATEGORY_TO_ID), (batch_size, max_items))
        color_ids = torch.randint(0, len(COLOR_TO_ID), (batch_size, max_items))
        style_ids = torch.randint(0, len(STYLE_TO_ID), (batch_size, max_items))
        attention_mask = torch.ones(batch_size, max_items)
        labels = torch.randint(0, 2, (batch_size,))
        
        # Forward pass
        outputs = model(
            item_images=item_images,
            category_ids=category_ids,
            color_ids=color_ids,
            style_ids=style_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        logger.info(f"Forward pass başarılı!")
        logger.info(f"Output keys: {list(outputs.keys())}")
        logger.info(f"Compatibility logits shape: {outputs['compatibility_logits'].shape}")
        logger.info(f"Loss: {outputs['loss'].item():.4f}")
        
        # 3. Model summary testi
        logger.info("3. Model summary testi...")
        summary = get_model_summary(model)
        logger.info(f"Model summary: {summary}")
        
        logger.info("Tüm testler başarılı!")
        
    except Exception as e:
        logger.error(f"Test hatası: {e}")
        raise


if __name__ == "__main__":
    # Test çalıştır
    test_outfit_transformer()
