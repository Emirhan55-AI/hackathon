"""
OutfitTransformer Inference Pipeline - Aura Project
Bu modül, eğitilmiş OutfitTransformer modelini kullanarak outfit önerileri ve uyumluluk analizi yapar.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import time

# Core libraries  
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

# Torchvision
import torchvision.transforms as transforms

# Vector similarity
from sklearn.metrics.pairwise import cosine_similarity
import faiss

# Graph operations
import networkx as nx

# Local imports
from model import (
    OutfitTransformer,
    OutfitTransformerConfig,
    load_model,
    convert_attributes_to_ids,
    CATEGORY_TO_ID,
    COLOR_TO_ID,
    STYLE_TO_ID,
    ID_TO_CATEGORY,
    ID_TO_COLOR,
    ID_TO_STYLE
)
from data_loader import (
    get_polyvore_transforms,
    preprocess_image,
    validate_outfit_data
)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inference constants
DEFAULT_INFERENCE_CONFIG = {
    "max_outfit_size": 5,
    "min_compatibility_score": 0.7,
    "top_k_recommendations": 10,
    "similarity_threshold": 0.8,
    "device": "auto"  # auto, cpu, cuda
}

# Fashion rules (basit heuristic'ler)
FASHION_COMPATIBILITY_RULES = {
    "seasonal_colors": {
        "spring": ["pastel", "light", "bright"],
        "summer": ["bright", "vibrant", "light"],
        "fall": ["warm", "earth", "deep"],
        "winter": ["dark", "bold", "cool"]
    },
    "occasion_styles": {
        "casual": ["relaxed", "comfortable", "informal"],
        "formal": ["elegant", "sophisticated", "classic"],
        "business": ["professional", "conservative", "clean"],
        "party": ["glamorous", "trendy", "bold"]
    },
    "incompatible_categories": [
        ("dress", "pants"),
        ("dress", "skirt"),
        ("shorts", "pants"),
        ("swimwear", "outerwear")
    ]
}


class OutfitRecommendationEngine:
    """
    OutfitTransformer tabanlı outfit öneri motoru
    
    Bu sınıf, moda itemları arasındaki uyumluluğu analiz eder ve
    outfit önerileri oluşturur.
    """
    
    def __init__(self,
                 model_path: str,
                 config: Optional[Dict[str, Any]] = None,
                 device: Optional[str] = None):
        """
        Öneri motorunu initialize eder
        
        Args:
            model_path: Eğitilmiş model path
            config: Inference konfigürasyonu
            device: Kullanılacak device
        """
        self.config = config or DEFAULT_INFERENCE_CONFIG.copy()
        
        # Device setup
        if device:
            self.device = torch.device(device)
        elif self.config["device"] == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config["device"])
        
        logger.info(f"Device: {self.device}")
        
        # Load model
        logger.info(f"Model yükleniyor: {model_path}")
        self.model, self.model_metadata = load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Image transforms
        self.transforms = get_polyvore_transforms(
            image_size=224,
            is_training=False
        )
        
        # Item database (runtime'da doldurulacak)
        self.item_database = {}
        self.item_embeddings = None
        self.faiss_index = None
        
        logger.info("OutfitRecommendationEngine hazır!")
    
    def add_item_to_database(self,
                           item_id: str,
                           image_path: str,
                           category: str,
                           color: Optional[str] = None,
                           style: Optional[str] = None,
                           price: Optional[float] = None,
                           brand: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Item veritabanına yeni item ekler
        
        Args:
            item_id: Unique item ID
            image_path: Item görüntü path
            category: Item kategorisi
            color: Item rengi
            style: Item stili
            price: Item fiyatı
            brand: Item markası
            metadata: Ek metadata
            
        Returns:
            Dict[str, Any]: Item verisi
        """
        try:
            # Image preprocessing
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transforms(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                image_features = self.model.image_feature_extractor(image_tensor)
                image_embedding = image_features.mean(dim=1).cpu().numpy()  # Global avg pooling
            
            # Create item entry
            item_data = {
                "item_id": item_id,
                "image_path": image_path,
                "category": category,
                "color": color or "unknown",
                "style": style or "unknown",
                "price": price,
                "brand": brand,
                "image_embedding": image_embedding.flatten(),
                "metadata": metadata or {}
            }
            
            self.item_database[item_id] = item_data
            
            # Update FAISS index
            self._update_faiss_index()
            
            logger.info(f"Item eklendi: {item_id} ({category})")
            return item_data
            
        except Exception as e:
            logger.error(f"Item eklenirken hata: {item_id} - {e}")
            raise
    
    def load_item_database(self, items_data: List[Dict[str, Any]]) -> int:
        """
        Bulk item veritabanı yükleme
        
        Args:
            items_data: Item verileri listesi
            
        Returns:
            int: Yüklenen item sayısı
        """
        logger.info(f"{len(items_data)} item yükleniyor...")
        
        loaded_count = 0
        
        for item_data in items_data:
            try:
                self.add_item_to_database(**item_data)
                loaded_count += 1
            except Exception as e:
                logger.warning(f"Item yüklenemedi: {item_data.get('item_id', 'unknown')} - {e}")
        
        logger.info(f"{loaded_count}/{len(items_data)} item yüklendi")
        return loaded_count
    
    def _update_faiss_index(self):
        """FAISS index'i günceller"""
        if not self.item_database:
            return
        
        # Embeddings'leri topla
        embeddings = []
        item_ids = []
        
        for item_id, item_data in self.item_database.items():
            embeddings.append(item_data["image_embedding"])
            item_ids.append(item_id)
        
        embeddings = np.array(embeddings)
        
        # FAISS index oluştur
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        
        # Normalize embeddings
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        self.faiss_index = index
        self.item_embeddings = embeddings
        self.item_ids_list = item_ids
        
        logger.info(f"FAISS index güncellendi: {len(item_ids)} item")
    
    def find_similar_items(self,
                          target_item_id: str,
                          top_k: int = 10,
                          category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Benzer itemları bulur
        
        Args:
            target_item_id: Hedef item ID
            top_k: Döndürülecek item sayısı
            category_filter: Kategori filtresi
            
        Returns:
            List[Dict[str, Any]]: Benzer itemlar
        """
        if target_item_id not in self.item_database:
            raise ValueError(f"Item bulunamadı: {target_item_id}")
        
        if self.faiss_index is None:
            logger.warning("FAISS index bulunamadı")
            return []
        
        # Target embedding
        target_embedding = self.item_database[target_item_id]["image_embedding"]
        target_embedding = target_embedding.reshape(1, -1)
        faiss.normalize_L2(target_embedding)
        
        # Search
        similarities, indices = self.faiss_index.search(target_embedding, top_k + 1)
        
        similar_items = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx >= len(self.item_ids_list):
                continue
                
            item_id = self.item_ids_list[idx]
            
            # Skip kendisi
            if item_id == target_item_id:
                continue
            
            item_data = self.item_database[item_id].copy()
            item_data["similarity_score"] = float(similarity)
            
            # Category filter
            if category_filter and item_data["category"] != category_filter:
                continue
            
            similar_items.append(item_data)
        
        return similar_items[:top_k]
    
    def predict_outfit_compatibility(self,
                                   item_ids: List[str],
                                   return_scores: bool = True) -> Dict[str, Any]:
        """
        Outfit uyumluluğunu predict eder
        
        Args:
            item_ids: Item ID listesi
            return_scores: Detaylı skorları döndür
            
        Returns:
            Dict[str, Any]: Compatibility prediction
        """
        try:
            # Validate items
            if len(item_ids) < 2:
                raise ValueError("En az 2 item gerekli")
            
            if len(item_ids) > self.config["max_outfit_size"]:
                raise ValueError(f"Maksimum {self.config['max_outfit_size']} item destekleniyor")
            
            # Check items exist
            missing_items = [item_id for item_id in item_ids if item_id not in self.item_database]
            if missing_items:
                raise ValueError(f"Item bulunamadı: {missing_items}")
            
            # Prepare model input
            batch_data = self._prepare_outfit_batch(item_ids)
            
            # Model prediction
            with torch.no_grad():
                outputs = self.model(
                    item_images=batch_data["item_images"],
                    category_ids=batch_data["category_ids"],
                    color_ids=batch_data["color_ids"],
                    style_ids=batch_data["style_ids"],
                    attention_mask=batch_data["attention_mask"]
                )
            
            # Extract results
            compatibility_logits = outputs["compatibility_logits"]
            compatibility_probs = F.softmax(compatibility_logits, dim=-1)
            
            # Predictions
            predicted_label = torch.argmax(compatibility_probs, dim=-1).item()
            compatibility_score = compatibility_probs[0, 1].item()  # Probability of compatible
            
            # Outfit score
            outfit_scores = outputs["outfit_scores"]
            outfit_score = torch.mean(outfit_scores).item()
            
            result = {
                "outfit_id": f"outfit_{'_'.join(item_ids[:3])}",
                "item_ids": item_ids,
                "is_compatible": predicted_label == 1,
                "compatibility_score": compatibility_score,
                "outfit_score": outfit_score,
                "recommendation": "compatible" if compatibility_score > self.config["min_compatibility_score"] else "incompatible"
            }
            
            if return_scores:
                result.update({
                    "detailed_scores": {
                        "compatibility_probs": compatibility_probs[0].cpu().numpy().tolist(),
                        "outfit_scores": outfit_scores[0].cpu().numpy().tolist(),
                        "predicted_label": predicted_label
                    }
                })
            
            # Fashion rules check
            fashion_rules_result = self._check_fashion_rules(item_ids)
            result["fashion_rules"] = fashion_rules_result
            
            return result
            
        except Exception as e:
            logger.error(f"Compatibility prediction hatası: {e}")
            raise
    
    def generate_outfit_recommendations(self,
                                      seed_item_ids: List[str],
                                      target_categories: Optional[List[str]] = None,
                                      occasion: Optional[str] = None,
                                      season: Optional[str] = None,
                                      max_outfits: int = 5) -> List[Dict[str, Any]]:
        """
        Outfit önerileri oluşturur
        
        Args:
            seed_item_ids: Başlangıç itemları
            target_categories: Hedef kategoriler
            occasion: Durum (casual, formal, etc.)
            season: Mevsim
            max_outfits: Maksimum outfit sayısı
            
        Returns:
            List[Dict[str, Any]]: Outfit önerileri
        """
        logger.info(f"Outfit önerileri oluşturuluyor: {seed_item_ids}")
        
        if not seed_item_ids:
            raise ValueError("En az 1 seed item gerekli")
        
        recommendations = []
        
        # Seed itemları validate et
        for item_id in seed_item_ids:
            if item_id not in self.item_database:
                raise ValueError(f"Seed item bulunamadı: {item_id}")
        
        # Candidate items pool
        candidate_items = self._get_candidate_items(
            seed_item_ids=seed_item_ids,
            target_categories=target_categories,
            occasion=occasion,
            season=season
        )
        
        logger.info(f"{len(candidate_items)} candidate item bulundu")
        
        # Generate outfit combinations
        for i in range(max_outfits * 3):  # Generate more, filter best
            outfit_items = self._generate_single_outfit(
                seed_item_ids=seed_item_ids,
                candidate_items=candidate_items,
                target_size=min(self.config["max_outfit_size"], 4)
            )
            
            if not outfit_items or len(outfit_items) < 2:
                continue
            
            # Predict compatibility
            try:
                compatibility_result = self.predict_outfit_compatibility(outfit_items)
                
                if compatibility_result["compatibility_score"] >= self.config["min_compatibility_score"]:
                    recommendation = {
                        "outfit_id": f"rec_{int(time.time())}_{i}",
                        "items": [self.item_database[item_id] for item_id in outfit_items],
                        "compatibility": compatibility_result,
                        "occasion": occasion,
                        "season": season,
                        "generation_method": "transformer_based"
                    }
                    
                    recommendations.append(recommendation)
            
            except Exception as e:
                logger.warning(f"Outfit compatibility check hatası: {e}")
                continue
        
        # Sort by compatibility score
        recommendations.sort(
            key=lambda x: x["compatibility"]["compatibility_score"],
            reverse=True
        )
        
        # Remove duplicates ve limit
        unique_recommendations = []
        seen_item_sets = set()
        
        for rec in recommendations:
            item_set = tuple(sorted([item["item_id"] for item in rec["items"]]))
            if item_set not in seen_item_sets:
                unique_recommendations.append(rec)
                seen_item_sets.add(item_set)
                
                if len(unique_recommendations) >= max_outfits:
                    break
        
        logger.info(f"{len(unique_recommendations)} outfit önerisi oluşturuldu")
        return unique_recommendations
    
    def get_item_recommendations(self,
                               item_id: str,
                               categories: Optional[List[str]] = None,
                               top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Belirli bir item için diğer item önerileri
        
        Args:
            item_id: Hedef item ID
            categories: Önerilecek kategoriler
            top_k: Öneri sayısı
            
        Returns:
            List[Dict[str, Any]]: Item önerileri
        """
        if item_id not in self.item_database:
            raise ValueError(f"Item bulunamadı: {item_id}")
        
        target_item = self.item_database[item_id]
        recommendations = []
        
        # Her kategori için ayrı öneriler
        if categories:
            target_categories = categories
        else:
            # Common complementary categories
            target_categories = ["tops", "bottoms", "shoes", "accessories"]
        
        for category in target_categories:
            if category == target_item["category"]:
                continue  # Skip same category
            
            similar_items = self.find_similar_items(
                target_item_id=item_id,
                top_k=top_k,
                category_filter=category
            )
            
            for item in similar_items:
                # Compatibility check
                try:
                    compatibility = self.predict_outfit_compatibility([item_id, item["item_id"]])
                    
                    if compatibility["compatibility_score"] >= self.config["min_compatibility_score"]:
                        recommendation = {
                            **item,
                            "compatibility_with_target": compatibility["compatibility_score"],
                            "recommendation_reason": f"Compatible {category} for {target_item['category']}"
                        }
                        recommendations.append(recommendation)
                
                except Exception as e:
                    logger.warning(f"Item recommendation hatası: {e}")
                    continue
        
        # Sort by compatibility
        recommendations.sort(
            key=lambda x: x["compatibility_with_target"],
            reverse=True
        )
        
        return recommendations[:top_k]
    
    def analyze_outfit_graph(self, item_ids: List[str]) -> Dict[str, Any]:
        """
        Outfit'i graph olarak analiz eder
        
        Args:
            item_ids: Item ID listesi
            
        Returns:
            Dict[str, Any]: Graph analizi
        """
        if len(item_ids) < 2:
            raise ValueError("En az 2 item gerekli")
        
        # Create compatibility graph
        G = nx.Graph()
        
        # Add nodes
        for item_id in item_ids:
            if item_id in self.item_database:
                item_data = self.item_database[item_id]
                G.add_node(item_id, **item_data)
        
        # Add edges with compatibility scores
        for i, item1 in enumerate(item_ids):
            for j, item2 in enumerate(item_ids[i+1:], i+1):
                if item1 in self.item_database and item2 in self.item_database:
                    try:
                        compatibility = self.predict_outfit_compatibility([item1, item2])
                        weight = compatibility["compatibility_score"]
                        
                        if weight >= self.config["min_compatibility_score"]:
                            G.add_edge(item1, item2, weight=weight)
                    
                    except Exception as e:
                        logger.warning(f"Graph edge hatası: {item1}-{item2}: {e}")
        
        # Graph analysis
        analysis = {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "density": nx.density(G),
            "is_connected": nx.is_connected(G),
            "average_compatibility": np.mean([data["weight"] for _, _, data in G.edges(data=True)]) if G.edges() else 0.0
        }
        
        # Component analysis
        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            analysis["num_components"] = len(components)
            analysis["largest_component_size"] = max(len(comp) for comp in components) if components else 0
        
        # Central items (high degree)
        if G.nodes():
            degrees = dict(G.degree(weight="weight"))
            central_item = max(degrees, key=degrees.get)
            analysis["most_central_item"] = {
                "item_id": central_item,
                "centrality_score": degrees[central_item]
            }
        
        return analysis
    
    def _prepare_outfit_batch(self, item_ids: List[str]) -> Dict[str, torch.Tensor]:
        """Model için batch hazırlar"""
        batch_size = 1
        max_items = len(item_ids)
        
        # Images
        item_images = torch.zeros(batch_size, max_items, 3, 224, 224, device=self.device)
        
        # Attributes
        category_ids = torch.zeros(batch_size, max_items, dtype=torch.long, device=self.device)
        color_ids = torch.zeros(batch_size, max_items, dtype=torch.long, device=self.device)
        style_ids = torch.zeros(batch_size, max_items, dtype=torch.long, device=self.device)
        
        # Attention mask
        attention_mask = torch.zeros(batch_size, max_items, device=self.device)
        
        for i, item_id in enumerate(item_ids):
            item_data = self.item_database[item_id]
            
            # Load image
            try:
                image = Image.open(item_data["image_path"]).convert("RGB")
                image_tensor = self.transforms(image)
                item_images[0, i] = image_tensor
                
                # Attention mask
                attention_mask[0, i] = 1
                
                # Attributes
                category_id = CATEGORY_TO_ID.get(item_data["category"], 0)
                color_id = COLOR_TO_ID.get(item_data["color"], 0)
                style_id = STYLE_TO_ID.get(item_data["style"], 0)
                
                category_ids[0, i] = category_id
                color_ids[0, i] = color_id
                style_ids[0, i] = style_id
                
            except Exception as e:
                logger.warning(f"Item preprocessing hatası {item_id}: {e}")
        
        return {
            "item_images": item_images,
            "category_ids": category_ids,
            "color_ids": color_ids,
            "style_ids": style_ids,
            "attention_mask": attention_mask
        }
    
    def _get_candidate_items(self,
                            seed_item_ids: List[str],
                            target_categories: Optional[List[str]] = None,
                            occasion: Optional[str] = None,
                            season: Optional[str] = None) -> List[str]:
        """Candidate item pool oluşturur"""
        candidates = []
        
        # Seed item'lar için benzer itemlar
        for seed_id in seed_item_ids:
            similar_items = self.find_similar_items(seed_id, top_k=20)
            candidates.extend([item["item_id"] for item in similar_items])
        
        # Kategori filtresi
        if target_categories:
            category_filtered = []
            for item_id in self.item_database:
                if self.item_database[item_id]["category"] in target_categories:
                    category_filtered.append(item_id)
            candidates.extend(category_filtered)
        
        # Unique candidates
        candidates = list(set(candidates))
        
        # Remove seed items
        candidates = [item_id for item_id in candidates if item_id not in seed_item_ids]
        
        return candidates
    
    def _generate_single_outfit(self,
                               seed_item_ids: List[str],
                               candidate_items: List[str],
                               target_size: int = 4) -> List[str]:
        """Tek outfit kombinasyonu oluşturur"""
        outfit = seed_item_ids.copy()
        
        # Add items until target size
        for _ in range(target_size - len(outfit)):
            if not candidate_items:
                break
            
            best_item = None
            best_score = 0.0
            
            # Her candidate için outfit'e uygunluğunu test et
            for candidate_id in candidate_items[:min(50, len(candidate_items))]:  # Limit search
                if candidate_id in outfit:
                    continue
                
                # Test outfit
                test_outfit = outfit + [candidate_id]
                
                try:
                    compatibility = self.predict_outfit_compatibility(test_outfit, return_scores=False)
                    score = compatibility["compatibility_score"]
                    
                    if score > best_score:
                        best_score = score
                        best_item = candidate_id
                
                except Exception:
                    continue
            
            # Add best item
            if best_item and best_score >= self.config["min_compatibility_score"]:
                outfit.append(best_item)
                candidate_items.remove(best_item)
            else:
                break
        
        return outfit
    
    def _check_fashion_rules(self, item_ids: List[str]) -> Dict[str, Any]:
        """Fashion rules ile compatibility check"""
        items = [self.item_database[item_id] for item_id in item_ids]
        
        rules_result = {
            "passed_rules": [],
            "failed_rules": [],
            "warnings": [],
            "overall_score": 1.0
        }
        
        # Category conflicts check
        categories = [item["category"] for item in items]
        for cat1, cat2 in FASHION_COMPATIBILITY_RULES["incompatible_categories"]:
            if cat1 in categories and cat2 in categories:
                rules_result["failed_rules"].append(f"Incompatible categories: {cat1} + {cat2}")
                rules_result["overall_score"] -= 0.3
        
        # Same category too many times
        from collections import Counter
        category_counts = Counter(categories)
        for category, count in category_counts.items():
            if count > 2:
                rules_result["warnings"].append(f"Too many {category} items: {count}")
                rules_result["overall_score"] -= 0.1
        
        # Minimum score
        rules_result["overall_score"] = max(0.0, rules_result["overall_score"])
        
        return rules_result
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Veritabanı istatistikleri"""
        if not self.item_database:
            return {"total_items": 0}
        
        categories = [item["category"] for item in self.item_database.values()]
        colors = [item["color"] for item in self.item_database.values()]
        styles = [item["style"] for item in self.item_database.values()]
        
        from collections import Counter
        
        return {
            "total_items": len(self.item_database),
            "categories": dict(Counter(categories)),
            "colors": dict(Counter(colors)),
            "styles": dict(Counter(styles)),
            "has_faiss_index": self.faiss_index is not None,
            "embedding_dimension": self.item_embeddings.shape[1] if self.item_embeddings is not None else 0
        }


# Utility functions
def load_outfit_engine(model_path: str, 
                      items_database_path: Optional[str] = None,
                      config: Optional[Dict[str, Any]] = None) -> OutfitRecommendationEngine:
    """
    Outfit recommendation engine yükler
    
    Args:
        model_path: Model checkpoint path
        items_database_path: Items database JSON path
        config: Engine konfigürasyonu
        
    Returns:
        OutfitRecommendationEngine: Hazır engine
    """
    # Engine oluştur
    engine = OutfitRecommendationEngine(
        model_path=model_path,
        config=config
    )
    
    # Items database yükle
    if items_database_path and Path(items_database_path).exists():
        logger.info(f"Items database yükleniyor: {items_database_path}")
        
        with open(items_database_path, 'r') as f:
            items_data = json.load(f)
        
        engine.load_item_database(items_data)
    
    return engine


def create_demo_items_database(image_dir: str, output_path: str) -> str:
    """
    Demo için items database oluşturur
    
    Args:
        image_dir: Görüntü dizini
        output_path: Output JSON path
        
    Returns:
        str: Created database path
    """
    demo_items = []
    
    # Example items (gerçek implementation'da automatic extraction olacak)
    categories = ["tops", "bottoms", "shoes", "accessories", "outerwear"]
    colors = ["black", "white", "blue", "red", "brown", "gray"]
    styles = ["casual", "formal", "trendy", "classic", "sporty"]
    
    item_id = 1
    for category in categories:
        for color in colors[:3]:  # Limit combinations
            for style in styles[:2]:
                demo_item = {
                    "item_id": f"demo_{item_id:04d}",
                    "image_path": f"{image_dir}/demo_{category}_{color}_{style}.jpg",
                    "category": category,
                    "color": color,
                    "style": style,
                    "price": round(50 + np.random.uniform(0, 200), 2),
                    "brand": f"Brand_{item_id % 5 + 1}",
                    "metadata": {
                        "season": "all",
                        "occasion": "casual",
                        "material": "cotton"
                    }
                }
                demo_items.append(demo_item)
                item_id += 1
    
    # Save database
    with open(output_path, 'w') as f:
        json.dump(demo_items, f, indent=2)
    
    logger.info(f"Demo database oluşturuldu: {len(demo_items)} item -> {output_path}")
    return output_path


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="OutfitTransformer Inference")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Trained model checkpoint path")
    parser.add_argument("--items_db", type=str, default=None,
                       help="Items database JSON path")
    parser.add_argument("--demo", action="store_true",
                       help="Run demo mode")
    
    args = parser.parse_args()
    
    if args.demo:
        # Demo mode
        logger.info("Demo mode çalıştırılıyor...")
        
        # Create demo database
        demo_db_path = "demo_items_database.json"
        create_demo_items_database("./demo_images", demo_db_path)
        
        # Load engine
        engine = load_outfit_engine(
            model_path=args.model_path,
            items_database_path=demo_db_path
        )
        
        # Example usage
        print("=== Demo OutfitTransformer Inference ===")
        print(f"Database stats: {engine.get_database_stats()}")
        
        # Example recommendations
        seed_items = ["demo_0001", "demo_0002"]  # Example seed items
        recommendations = engine.generate_outfit_recommendations(
            seed_item_ids=seed_items,
            max_outfits=3
        )
        
        print(f"\nGenerated {len(recommendations)} outfit recommendations")
        for i, rec in enumerate(recommendations):
            print(f"Recommendation {i+1}:")
            print(f"  Compatibility: {rec['compatibility']['compatibility_score']:.3f}")
            print(f"  Items: {[item['item_id'] for item in rec['items']]}")
    
    else:
        # Regular inference mode
        engine = load_outfit_engine(
            model_path=args.model_path,
            items_database_path=args.items_db
        )
        
        print("OutfitTransformer Inference Engine hazır!")
        print(f"Database stats: {engine.get_database_stats()}")
