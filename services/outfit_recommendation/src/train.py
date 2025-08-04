"""
OutfitTransformer Model Training Pipeline - Aura Project
Bu modül, OutfitTransformer modelinin eğitimi için gerekli fonksiyonları içerir.
"""

import os
import json
import logging
import time
import argparse
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import random

# Core libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

# Hugging Face
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator

# Progress tracking
from tqdm import tqdm
import wandb

# Evaluation metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np

# Local imports
from data_loader import (
    load_polyvore_dataset,
    create_outfit_dataloader,
    PolyvoreOutfitDataset,
    get_polyvore_transforms,
    validate_outfit_data
)
from model import (
    OutfitTransformer,
    OutfitTransformerConfig,
    create_outfit_transformer,
    save_model,
    load_model,
    prepare_model_for_training,
    convert_attributes_to_ids,
    CATEGORY_TO_ID,
    COLOR_TO_ID,
    STYLE_TO_ID
)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Training constants
DEFAULT_TRAINING_CONFIG = {
    "batch_size": 16,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "num_epochs": 50,
    "warmup_steps": 1000,
    "gradient_clip_val": 1.0,
    "eval_steps": 500,
    "save_steps": 1000,
    "logging_steps": 100,
    "seed": 42
}

# Evaluation metrics
EVALUATION_METRICS = [
    "accuracy", "precision", "recall", "f1", "auc"
]


def set_seed(seed: int):
    """
    Reproducibility için seed ayarlar
    
    Args:
        seed: Random seed değeri
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic behavior (performans düşebilir)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Seed ayarlandı: {seed}")


def create_negative_outfits(positive_outfits: List[Dict[str, Any]], 
                          items: Dict[str, Dict[str, Any]],
                          negative_ratio: float = 1.0) -> List[Dict[str, Any]]:
    """
    Pozitif outfitlerden negatif (uyumsuz) outfitler oluşturur
    
    Args:
        positive_outfits: Pozitif outfit listesi
        items: Item mapping
        negative_ratio: Pozitif outfit başına negatif outfit oranı
        
    Returns:
        List[Dict[str, Any]]: Negatif outfit listesi
    """
    logger.info("Negatif outfitler oluşturuluyor...")
    
    negative_outfits = []
    item_ids = list(items.keys())
    
    num_negatives = int(len(positive_outfits) * negative_ratio)
    
    for i in range(num_negatives):
        # Random item'lar seç
        num_items = random.randint(2, 5)  # 2-5 item per outfit
        selected_items = random.sample(item_ids, min(num_items, len(item_ids)))
        
        # Negatif outfit oluştur
        negative_outfit = {
            "outfit_id": f"negative_{i}",
            "items": selected_items,
            "season": "unknown",
            "occasion": "unknown",
            "label": 0  # Incompatible
        }
        
        negative_outfits.append(negative_outfit)
    
    logger.info(f"{len(negative_outfits)} negatif outfit oluşturuldu")
    return negative_outfits


def prepare_training_data(data_dir: str, 
                         image_dir: str,
                         negative_ratio: float = 1.0,
                         val_split: float = 0.2) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    Eğitim ve validation veri setlerini hazırlar
    
    Args:
        data_dir: Polyvore veri dizini
        image_dir: Görüntü dizini
        negative_ratio: Negatif örnek oranı
        val_split: Validation split oranı
        
    Returns:
        Tuple[DataLoader, DataLoader, Dict]: (train_loader, val_loader, metadata)
    """
    logger.info("Eğitim verisi hazırlanıyor...")
    
    # Polyvore train data yükle
    train_dataset = load_polyvore_dataset(data_dir, split="train")
    
    # Pozitif outfitler (original)
    positive_outfits = train_dataset["outfits"]
    items = train_dataset["items"]
    
    # Pozitif outfitlere label ekle
    for outfit in positive_outfits:
        outfit["label"] = 1  # Compatible
    
    # Negatif outfitler oluştur
    negative_outfits = create_negative_outfits(
        positive_outfits, items, negative_ratio
    )
    
    # Tüm outfitleri birleştir
    all_outfits = positive_outfits + negative_outfits
    random.shuffle(all_outfits)
    
    # Updated dataset oluştur
    combined_dataset = {
        "outfits": all_outfits,
        "items": items,
        "split": "train_combined",
        "num_outfits": len(all_outfits),
        "num_items": len(items)
    }
    
    # Train/validation split
    total_outfits = len(all_outfits)
    val_size = int(total_outfits * val_split)
    train_size = total_outfits - val_size
    
    train_outfits = all_outfits[:train_size]
    val_outfits = all_outfits[train_size:]
    
    train_data = {
        "outfits": train_outfits,
        "items": items,
        "split": "train",
        "num_outfits": len(train_outfits),
        "num_items": len(items)
    }
    
    val_data = {
        "outfits": val_outfits,
        "items": items,
        "split": "validation",
        "num_outfits": len(val_outfits),
        "num_items": len(items)
    }
    
    # DataLoader'ları oluştur
    train_loader = create_outfit_dataloader(
        dataset=train_data,
        image_dir=image_dir,
        batch_size=DEFAULT_TRAINING_CONFIG["batch_size"],
        shuffle=True,
        num_workers=4
    )
    
    val_loader = create_outfit_dataloader(
        dataset=val_data,
        image_dir=image_dir,
        batch_size=DEFAULT_TRAINING_CONFIG["batch_size"],
        shuffle=False,
        num_workers=4
    )
    
    metadata = {
        "total_outfits": total_outfits,
        "train_outfits": len(train_outfits),
        "val_outfits": len(val_outfits),
        "positive_outfits": len(positive_outfits),
        "negative_outfits": len(negative_outfits),
        "total_items": len(items)
    }
    
    logger.info(f"Veri hazırlandı: {metadata}")
    return train_loader, val_loader, metadata


def compute_metrics(predictions: np.ndarray, 
                   labels: np.ndarray,
                   prediction_scores: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Evaluation metriklerini hesaplar
    
    Args:
        predictions: Predicted class labels [batch_size]
        labels: True labels [batch_size] 
        prediction_scores: Prediction probabilities [batch_size, num_classes]
        
    Returns:
        Dict[str, float]: Computed metrics
    """
    metrics = {}
    
    # Accuracy
    metrics["accuracy"] = accuracy_score(labels, predictions)
    
    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    
    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1"] = f1
    
    # AUC (eğer prediction scores verilmişse)
    if prediction_scores is not None:
        try:
            if prediction_scores.shape[1] == 2:  # Binary classification
                auc = roc_auc_score(labels, prediction_scores[:, 1])
                metrics["auc"] = auc
        except Exception as e:
            logger.warning(f"AUC hesaplanamadı: {e}")
            metrics["auc"] = 0.0
    
    return metrics


def evaluate_model(model: OutfitTransformer,
                  dataloader: DataLoader,
                  device: torch.device,
                  criterion: nn.Module) -> Dict[str, float]:
    """
    Modeli evaluation veri seti üzerinde değerlendirir
    
    Args:
        model: OutfitTransformer modeli
        dataloader: Evaluation DataLoader
        device: Device
        criterion: Loss function
        
    Returns:
        Dict[str, float]: Evaluation metrics
    """
    model.eval()
    
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation"):
            # Batch'i device'a taşı
            item_images = batch["item_images"].to(device)
            num_items = batch["num_items"].to(device)
            
            # Attribute ID'lerini oluştur
            batch_size, max_items = item_images.shape[:2]
            
            # Labels (outfit compatibility)
            labels = []
            for outfit_features in batch["outfit_features"]:
                # Outfit features'dan label çıkar (eğer varsa)
                label = outfit_features.get("label", 1)  # Default compatible
                labels.append(label)
            
            labels = torch.tensor(labels, dtype=torch.long, device=device)
            
            # Dummy attribute IDs (gerçek implementation'da batch'ten alınacak)
            category_ids = torch.randint(0, len(CATEGORY_TO_ID), (batch_size, max_items), device=device)
            color_ids = torch.randint(0, len(COLOR_TO_ID), (batch_size, max_items), device=device)
            style_ids = torch.randint(0, len(STYLE_TO_ID), (batch_size, max_items), device=device)
            
            # Attention mask (valid items)
            attention_mask = torch.zeros(batch_size, max_items, device=device)
            for i, n_items in enumerate(num_items):
                attention_mask[i, :n_items] = 1
            
            # Forward pass
            outputs = model(
                item_images=item_images,
                category_ids=category_ids,
                color_ids=color_ids,
                style_ids=style_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Loss
            loss = outputs["loss"]
            total_loss += loss.item()
            
            # Predictions
            compatibility_logits = outputs["compatibility_logits"]
            predictions = torch.argmax(compatibility_logits, dim=-1)
            scores = torch.softmax(compatibility_logits, dim=-1)
            
            # Collect results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
    
    # Convert to numpy
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_labels, all_scores)
    metrics["eval_loss"] = total_loss / len(dataloader)
    
    model.train()
    return metrics


class OutfitTransformerTrainer:
    """
    OutfitTransformer modeli için trainer sınıfı
    
    Bu sınıf, modelin eğitimini, değerlendirmesini ve kaydetmesini yönetir.
    """
    
    def __init__(self,
                 model: OutfitTransformer,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 config: Dict[str, Any],
                 output_dir: str,
                 use_wandb: bool = False,
                 project_name: str = "outfit-transformer"):
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Accelerator for multi-GPU training
        self.accelerator = Accelerator()
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
        
        # Learning rate scheduler
        total_steps = len(train_dataloader) * config["num_epochs"]
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config["warmup_steps"],
            num_training_steps=total_steps
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Accelerator prepare
        self.model, self.optimizer, self.train_dataloader, self.val_dataloader = \
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader, self.val_dataloader
            )
        
        # Logging
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project=project_name,
                config=config,
                name=f"outfit-transformer-{int(time.time())}"
            )
        
        # Tensorboard
        self.tb_writer = SummaryWriter(log_dir=self.output_dir / "tensorboard")
        
        # Training state
        self.global_step = 0
        self.best_metric = 0.0
        self.training_history = []
        
        logger.info(f"Trainer initialized - Device: {self.device}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Tek epoch eğitimi
        
        Args:
            epoch: Epoch numarası
            
        Returns:
            Dict[str, float]: Epoch metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_compatibility_loss = 0.0
        total_score_loss = 0.0
        
        progress_bar = tqdm(
            self.train_dataloader, 
            desc=f"Epoch {epoch+1}/{self.config['num_epochs']}"
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Batch'i device'a taşı (accelerator otomatik yapıyor)
            item_images = batch["item_images"]
            num_items = batch["num_items"]
            
            batch_size, max_items = item_images.shape[:2]
            
            # Labels oluştur (gerçek implementation'da batch'ten gelecek)
            labels = []
            for outfit_features in batch["outfit_features"]:
                label = outfit_features.get("label", 1)
                labels.append(label)
            
            labels = torch.tensor(labels, dtype=torch.long, device=self.device)
            
            # Dummy attributes (gerçek implementation'da batch'ten gelecek)
            category_ids = torch.randint(0, len(CATEGORY_TO_ID), (batch_size, max_items), device=self.device)
            color_ids = torch.randint(0, len(COLOR_TO_ID), (batch_size, max_items), device=self.device)
            style_ids = torch.randint(0, len(STYLE_TO_ID), (batch_size, max_items), device=self.device)
            
            # Attention mask
            attention_mask = torch.zeros(batch_size, max_items, device=self.device)
            for i, n_items in enumerate(num_items):
                attention_mask[i, :n_items] = 1
            
            # Forward pass
            outputs = self.model(
                item_images=item_images,
                category_ids=category_ids,
                color_ids=color_ids,
                style_ids=style_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs["loss"]
            compatibility_loss = outputs["compatibility_loss"]
            score_loss = outputs["score_loss"]
            
            # Backward pass
            self.accelerator.backward(loss)
            
            # Gradient clipping
            if self.config["gradient_clip_val"] > 0:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config["gradient_clip_val"]
                )
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Update global step
            self.global_step += 1
            
            # Accumulate losses
            total_loss += loss.item()
            total_compatibility_loss += compatibility_loss.item()
            total_score_loss += score_loss.item()
            
            # Progress bar update
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Logging
            if self.global_step % self.config["logging_steps"] == 0:
                self._log_training_step(loss.item(), compatibility_loss.item(), score_loss.item())
            
            # Evaluation
            if self.global_step % self.config["eval_steps"] == 0:
                eval_metrics = self._evaluate()
                self._log_evaluation(eval_metrics)
                
                # Save best model
                if eval_metrics["f1"] > self.best_metric:
                    self.best_metric = eval_metrics["f1"]
                    self._save_checkpoint("best_model.pt", eval_metrics)
            
            # Periodic save
            if self.global_step % self.config["save_steps"] == 0:
                self._save_checkpoint(f"checkpoint_step_{self.global_step}.pt")
        
        # Epoch metrics
        num_batches = len(self.train_dataloader)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": total_loss / num_batches,
            "train_compatibility_loss": total_compatibility_loss / num_batches,
            "train_score_loss": total_score_loss / num_batches,
            "learning_rate": self.scheduler.get_last_lr()[0]
        }
        
        return epoch_metrics
    
    def train(self) -> Dict[str, Any]:
        """
        Ana eğitim loop'u
        
        Returns:
            Dict[str, Any]: Training history
        """
        logger.info("Eğitim başlatılıyor...")
        logger.info(f"Config: {self.config}")
        
        start_time = time.time()
        
        for epoch in range(self.config["num_epochs"]):
            # Train epoch
            epoch_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self._evaluate()
            
            # Combine metrics
            combined_metrics = {**epoch_metrics, **val_metrics}
            self.training_history.append(combined_metrics)
            
            # Log epoch results
            self._log_epoch(combined_metrics)
            
            # Early stopping check (basit implementation)
            if epoch > 10 and val_metrics["f1"] < 0.1:  # Model öğrenmiyor
                logger.warning("Model öğrenmiyor, eğitim durduruluyor")
                break
        
        training_time = time.time() - start_time
        
        # Final save
        final_metrics = self.training_history[-1] if self.training_history else {}
        self._save_checkpoint("final_model.pt", final_metrics)
        
        # Training summary
        summary = {
            "total_training_time": training_time,
            "total_epochs": len(self.training_history),
            "best_f1_score": self.best_metric,
            "final_metrics": final_metrics,
            "training_history": self.training_history
        }
        
        # Save training summary
        with open(self.output_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Eğitim tamamlandı! Süre: {training_time:.2f}s")
        logger.info(f"En iyi F1 skoru: {self.best_metric:.4f}")
        
        return summary
    
    def _evaluate(self) -> Dict[str, float]:
        """Validation evaluation"""
        return evaluate_model(
            self.model, 
            self.val_dataloader, 
            self.device, 
            self.criterion
        )
    
    def _log_training_step(self, loss: float, comp_loss: float, score_loss: float):
        """Training step logging"""
        if self.use_wandb:
            wandb.log({
                "train/loss": loss,
                "train/compatibility_loss": comp_loss,
                "train/score_loss": score_loss,
                "train/learning_rate": self.scheduler.get_last_lr()[0],
                "global_step": self.global_step
            })
        
        self.tb_writer.add_scalar("train/loss", loss, self.global_step)
        self.tb_writer.add_scalar("train/compatibility_loss", comp_loss, self.global_step)
        self.tb_writer.add_scalar("train/score_loss", score_loss, self.global_step)
        self.tb_writer.add_scalar("train/learning_rate", self.scheduler.get_last_lr()[0], self.global_step)
    
    def _log_evaluation(self, metrics: Dict[str, float]):
        """Evaluation logging"""
        if self.use_wandb:
            wandb.log({f"eval/{k}": v for k, v in metrics.items()})
        
        for key, value in metrics.items():
            self.tb_writer.add_scalar(f"eval/{key}", value, self.global_step)
    
    def _log_epoch(self, metrics: Dict[str, Any]):
        """Epoch logging"""
        logger.info(f"Epoch {metrics['epoch']} Summary:")
        logger.info(f"  Train Loss: {metrics['train_loss']:.4f}")
        logger.info(f"  Val Loss: {metrics['eval_loss']:.4f}")
        logger.info(f"  Val Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Val F1: {metrics['f1']:.4f}")
        logger.info(f"  Learning Rate: {metrics['learning_rate']:.2e}")
    
    def _save_checkpoint(self, filename: str, metrics: Optional[Dict[str, float]] = None):
        """Model checkpoint kaydetme"""
        checkpoint_path = self.output_dir / filename
        
        training_metadata = {
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "config": self.config,
            "metrics": metrics or {}
        }
        
        save_model(
            model=self.accelerator.unwrap_model(self.model),
            save_path=checkpoint_path,
            optimizer_state=self.optimizer.state_dict(),
            training_metadata=training_metadata
        )
        
        logger.info(f"Checkpoint kaydedildi: {checkpoint_path}")


def train_outfit_transformer(data_dir: str,
                            image_dir: str,
                            output_dir: str,
                            config: Optional[Dict[str, Any]] = None,
                            resume_from: Optional[str] = None,
                            use_wandb: bool = False) -> Dict[str, Any]:
    """
    OutfitTransformer modelini eğitir
    
    Args:
        data_dir: Polyvore veri dizini
        image_dir: Görüntü dizini
        output_dir: Çıktı dizini
        config: Eğitim konfigürasyonu
        resume_from: Resume checkpoint path
        use_wandb: Wandb kullan
        
    Returns:
        Dict[str, Any]: Training summary
    """
    # Config setup
    if config is None:
        config = DEFAULT_TRAINING_CONFIG.copy()
    
    # Seed set
    set_seed(config["seed"])
    
    # Output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Eğitim başlatılıyor...")
    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Image dir: {image_dir}")
    logger.info(f"Output dir: {output_dir}")
    
    # Prepare data
    train_loader, val_loader, data_metadata = prepare_training_data(
        data_dir=data_dir,
        image_dir=image_dir,
        negative_ratio=1.0,
        val_split=0.2
    )
    
    # Create model
    if resume_from and Path(resume_from).exists():
        logger.info(f"Model resume ediliyor: {resume_from}")
        model, _ = load_model(resume_from)
    else:
        logger.info("Yeni model oluşturuluyor...")
        model = create_outfit_transformer()
    
    # Create trainer
    trainer = OutfitTransformerTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=config,
        output_dir=output_dir,
        use_wandb=use_wandb
    )
    
    # Train
    training_summary = trainer.train()
    
    # Add data metadata
    training_summary["data_metadata"] = data_metadata
    
    return training_summary


# CLI interface
def main():
    """Ana CLI fonksiyonu"""
    parser = argparse.ArgumentParser(description="OutfitTransformer Model Training")
    
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Polyvore dataset directory")
    parser.add_argument("--image_dir", type=str, required=True,
                       help="Image directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for models and logs")
    parser.add_argument("--config_file", type=str, default=None,
                       help="Training config JSON file")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Resume training from checkpoint")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases logging")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=50,
                       help="Number of epochs")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Config setup
    config = DEFAULT_TRAINING_CONFIG.copy()
    
    # Config file'dan yükle
    if args.config_file and Path(args.config_file).exists():
        with open(args.config_file, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    # CLI args ile override
    config.update({
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "seed": args.seed
    })
    
    # Training başlat
    summary = train_outfit_transformer(
        data_dir=args.data_dir,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        config=config,
        resume_from=args.resume_from,
        use_wandb=args.use_wandb
    )
    
    logger.info("Eğitim tamamlandı!")
    logger.info(f"Özet: {summary}")


if __name__ == "__main__":
    main()
