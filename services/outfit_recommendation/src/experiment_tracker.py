"""
Experiment Tracker for Outfit Recommendation Service
Specialized experiment tracking for outfit recommendation model training
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid
import numpy as np


class OutfitExperimentTracker:
    """
    Specialized experiment tracker for outfit recommendation experiments.
    Includes metrics specific to recommendation systems.
    """
    
    def __init__(self, 
                 experiment_name: str = "outfit_recommendation",
                 run_name: Optional[str] = None,
                 base_dir: str = "experiments"):
        """Initialize outfit recommendation experiment tracker."""
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        
        # Create unique run ID
        self.run_id = str(uuid.uuid4())[:8]
        self.run_name = run_name or f"outfit_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.run_id}"
        
        # Setup directories
        self.experiment_dir = self.base_dir / experiment_name
        self.run_dir = self.experiment_dir / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize experiment data
        self.start_time = time.time()
        self.metrics = {}
        self.params = {}
        self.artifacts = {}
        self.logs = []
        self.recommendation_metrics = {}
        
        # Create run metadata
        self.metadata = {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "model_type": "outfit_recommendation"
        }
        
        self._save_metadata()
        self.log("Outfit recommendation experiment started", level="INFO")
    
    def log_model_config(self, config: Dict[str, Any]) -> None:
        """Log model configuration specific to outfit recommendation."""
        outfit_config = {
            "model_architecture": config.get("model_architecture", "OutfitTransformer"),
            "hidden_dim": config.get("hidden_dim", 512),
            "num_layers": config.get("num_layers", 6),
            "num_heads": config.get("num_heads", 8),
            "dropout": config.get("dropout", 0.1),
            "item_embedding_dim": config.get("item_embedding_dim", 256),
            "style_embedding_dim": config.get("style_embedding_dim", 128),
            "color_embedding_dim": config.get("color_embedding_dim", 64),
            "max_outfit_size": config.get("max_outfit_size", 8),
            "compatibility_threshold": config.get("compatibility_threshold", 0.6)
        }
        
        self.params.update(outfit_config)
        self._save_params()
        self.log(f"Logged model configuration: {list(outfit_config.keys())}", level="INFO")
    
    def log_training_config(self, config: Dict[str, Any]) -> None:
        """Log training configuration."""
        training_config = {
            "epochs": config.get("epochs", 30),
            "batch_size": config.get("batch_size", 32),
            "learning_rate": config.get("learning_rate", 2e-4),
            "weight_decay": config.get("weight_decay", 1e-5),
            "optimizer": config.get("optimizer", "AdamW"),
            "lr_scheduler": config.get("lr_scheduler", "reduce_on_plateau"),
            "patience": config.get("patience", 5),
            "factor": config.get("factor", 0.5),
            "gradient_clip_norm": config.get("gradient_clip_norm", 1.0)
        }
        
        self.params.update(training_config)
        self._save_params()
        self.log(f"Logged training configuration", level="INFO")
    
    def log_data_config(self, config: Dict[str, Any]) -> None:
        """Log data configuration."""
        data_config = {
            "dataset_size": config.get("dataset_size", 0),
            "train_size": config.get("train_size", 0),
            "val_size": config.get("val_size", 0),
            "test_size": config.get("test_size", 0),
            "min_outfit_size": config.get("min_outfit_size", 2),
            "max_outfit_size": config.get("max_outfit_size", 8),
            "style_categories": config.get("style_categories", []),
            "augmentation_enabled": config.get("augmentation_enabled", False)
        }
        
        self.params.update(data_config)
        self._save_params()
        self.log(f"Logged data configuration", level="INFO")
    
    def log_recommendation_metrics(self, 
                                 metrics: Dict[str, float], 
                                 k_values: List[int] = [5, 10, 20],
                                 step: Optional[int] = None) -> None:
        """Log recommendation-specific metrics."""
        timestamp = time.time()
        
        # Standard recommendation metrics
        rec_metrics = {}
        
        for k in k_values:
            if f"ndcg@{k}" in metrics:
                rec_metrics[f"ndcg@{k}"] = metrics[f"ndcg@{k}"]
            if f"hit_rate@{k}" in metrics:
                rec_metrics[f"hit_rate@{k}"] = metrics[f"hit_rate@{k}"]
            if f"precision@{k}" in metrics:
                rec_metrics[f"precision@{k}"] = metrics[f"precision@{k}"]
            if f"recall@{k}" in metrics:
                rec_metrics[f"recall@{k}"] = metrics[f"recall@{k}"]
        
        # Additional metrics
        additional_metrics = {
            "diversity": metrics.get("diversity", 0.0),
            "coverage": metrics.get("coverage", 0.0),
            "novelty": metrics.get("novelty", 0.0),
            "serendipity": metrics.get("serendipity", 0.0),
            "style_consistency": metrics.get("style_consistency", 0.0),
            "color_harmony": metrics.get("color_harmony", 0.0),
            "occasion_appropriateness": metrics.get("occasion_appropriateness", 0.0)
        }
        
        all_metrics = {**rec_metrics, **additional_metrics}
        
        # Log metrics
        for key, value in all_metrics.items():
            if key not in self.recommendation_metrics:
                self.recommendation_metrics[key] = []
            
            metric_entry = {
                "value": value,
                "timestamp": timestamp,
                "step": step
            }
            self.recommendation_metrics[key].append(metric_entry)
        
        self._save_recommendation_metrics()
        self.log(f"Logged recommendation metrics: {list(all_metrics.keys())}", level="INFO")
    
    def log_compatibility_matrix_stats(self, compatibility_matrix: np.ndarray) -> None:
        """Log statistics about the compatibility matrix."""
        stats = {
            "compatibility_matrix_shape": compatibility_matrix.shape,
            "mean_compatibility": float(np.mean(compatibility_matrix)),
            "std_compatibility": float(np.std(compatibility_matrix)),
            "min_compatibility": float(np.min(compatibility_matrix)),
            "max_compatibility": float(np.max(compatibility_matrix)),
            "sparsity": float(np.sum(compatibility_matrix == 0) / compatibility_matrix.size)
        }
        
        self.params.update(stats)
        self._save_params()
        self.log("Logged compatibility matrix statistics", level="INFO")
    
    def log_outfit_analysis(self, outfit_analysis: Dict[str, Any]) -> None:
        """Log analysis of generated outfits."""
        analysis = {
            "avg_outfit_size": outfit_analysis.get("avg_outfit_size", 0.0),
            "style_distribution": outfit_analysis.get("style_distribution", {}),
            "color_distribution": outfit_analysis.get("color_distribution", {}),
            "category_distribution": outfit_analysis.get("category_distribution", {}),
            "seasonal_distribution": outfit_analysis.get("seasonal_distribution", {}),
            "occasion_distribution": outfit_analysis.get("occasion_distribution", {}),
            "avg_style_score": outfit_analysis.get("avg_style_score", 0.0),
            "avg_color_harmony": outfit_analysis.get("avg_color_harmony", 0.0)
        }
        
        self.params.update(analysis)
        self._save_params()
        self.log("Logged outfit analysis", level="INFO")
    
    def log_user_feedback(self, feedback_data: Dict[str, Any]) -> None:
        """Log user feedback on recommendations."""
        feedback = {
            "total_feedback_samples": feedback_data.get("total_samples", 0),
            "avg_user_rating": feedback_data.get("avg_rating", 0.0),
            "positive_feedback_ratio": feedback_data.get("positive_ratio", 0.0),
            "style_preference_accuracy": feedback_data.get("style_accuracy", 0.0),
            "color_preference_accuracy": feedback_data.get("color_accuracy", 0.0),
            "occasion_match_accuracy": feedback_data.get("occasion_accuracy", 0.0)
        }
        
        self.params.update(feedback)
        self._save_params()
        self.log("Logged user feedback data", level="INFO")
    
    def log_model_size_info(self, model_info: Dict[str, Any]) -> None:
        """Log information about model size and complexity."""
        size_info = {
            "total_parameters": model_info.get("total_parameters", 0),
            "trainable_parameters": model_info.get("trainable_parameters", 0),
            "model_size_mb": model_info.get("model_size_mb", 0.0),
            "inference_time_ms": model_info.get("inference_time_ms", 0.0),
            "memory_usage_mb": model_info.get("memory_usage_mb", 0.0),
            "flops": model_info.get("flops", 0)
        }
        
        self.params.update(size_info)
        self._save_params()
        self.log("Logged model size information", level="INFO")
    
    def save_recommendation_examples(self, examples: List[Dict[str, Any]]) -> None:
        """Save example recommendations for analysis."""
        examples_file = self.run_dir / "recommendation_examples.json"
        with open(examples_file, "w") as f:
            json.dump(examples, f, indent=2)
        
        self.artifacts["recommendation_examples"] = {
            "path": str(examples_file),
            "type": "examples",
            "count": len(examples),
            "timestamp": datetime.now().isoformat()
        }
        
        self._save_artifacts()
        self.log(f"Saved {len(examples)} recommendation examples", level="INFO")
    
    def log_training_epoch(self, 
                          epoch: int,
                          train_loss: float,
                          val_loss: float,
                          learning_rate: float,
                          additional_metrics: Optional[Dict[str, float]] = None) -> None:
        """Log training metrics for a specific epoch."""
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": learning_rate
        }
        
        if additional_metrics:
            epoch_metrics.update(additional_metrics)
        
        # Log metrics with epoch as step
        for key, value in epoch_metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            
            metric_entry = {
                "value": value,
                "timestamp": time.time(),
                "step": epoch
            }
            self.metrics[key].append(metric_entry)
        
        self._save_metrics()
        self.log(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}", level="INFO")
    
    def finish(self, status: str = "completed") -> None:
        """Finish the experiment."""
        self.metadata["end_time"] = datetime.now().isoformat()
        self.metadata["duration"] = time.time() - self.start_time
        self.metadata["status"] = status
        
        self._save_metadata()
        self._save_summary()
        
        self.log(f"Outfit recommendation experiment finished with status: {status}", level="INFO")
    
    def get_best_metrics(self) -> Dict[str, float]:
        """Get the best values for each metric."""
        best_metrics = {}
        
        # For recommendation metrics, higher is usually better
        higher_is_better = [
            "ndcg", "hit_rate", "precision", "recall", "diversity", 
            "coverage", "novelty", "serendipity", "style_consistency",
            "color_harmony", "occasion_appropriateness"
        ]
        
        # For loss metrics, lower is better
        lower_is_better = ["loss", "val_loss", "train_loss"]
        
        for metric_name, history in self.metrics.items():
            if not history:
                continue
                
            values = [entry["value"] for entry in history]
            
            # Determine if higher or lower is better
            if any(keyword in metric_name.lower() for keyword in higher_is_better):
                best_metrics[f"best_{metric_name}"] = max(values)
            elif any(keyword in metric_name.lower() for keyword in lower_is_better):
                best_metrics[f"best_{metric_name}"] = min(values)
            else:
                # Default to latest value
                best_metrics[f"latest_{metric_name}"] = values[-1]
        
        # Same for recommendation metrics
        for metric_name, history in self.recommendation_metrics.items():
            if not history:
                continue
                
            values = [entry["value"] for entry in history]
            best_metrics[f"best_{metric_name}"] = max(values)  # Recommendation metrics are generally higher-is-better
        
        return best_metrics
    
    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message
        }
        self.logs.append(log_entry)
        
        # Also write to log file
        log_file = self.run_dir / "experiment.log"
        with open(log_file, "a") as f:
            f.write(f"[{log_entry['timestamp']}] {level}: {message}\n")
        
        # Print to console
        print(f"[{level}] {message}")
    
    def _save_metadata(self) -> None:
        """Save experiment metadata."""
        with open(self.run_dir / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)
    
    def _save_params(self) -> None:
        """Save parameters."""
        with open(self.run_dir / "params.json", "w") as f:
            json.dump(self.params, f, indent=2)
    
    def _save_metrics(self) -> None:
        """Save metrics."""
        with open(self.run_dir / "metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)
    
    def _save_recommendation_metrics(self) -> None:
        """Save recommendation-specific metrics."""
        with open(self.run_dir / "recommendation_metrics.json", "w") as f:
            json.dump(self.recommendation_metrics, f, indent=2)
    
    def _save_artifacts(self) -> None:
        """Save artifacts metadata."""
        with open(self.run_dir / "artifacts.json", "w") as f:
            json.dump(self.artifacts, f, indent=2)
    
    def _save_summary(self) -> None:
        """Save experiment summary."""
        summary = {
            "metadata": self.metadata,
            "params": self.params,
            "best_metrics": self.get_best_metrics(),
            "artifacts": list(self.artifacts.keys()),
            "total_logs": len(self.logs)
        }
        
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Also save to experiment-level summary
        experiment_summary_file = self.experiment_dir / "experiments_summary.jsonl"
        with open(experiment_summary_file, "a") as f:
            f.write(json.dumps(summary) + "\n")


# Convenience function
def create_outfit_tracker(run_name: Optional[str] = None) -> OutfitExperimentTracker:
    """Create experiment tracker for outfit recommendation."""
    return OutfitExperimentTracker(run_name=run_name)


# Example usage
if __name__ == "__main__":
    # Example outfit recommendation experiment
    tracker = create_outfit_tracker("outfit_transformer_v1")
    
    # Log configurations
    tracker.log_model_config({
        "model_architecture": "OutfitTransformer",
        "hidden_dim": 512,
        "num_layers": 6,
        "num_heads": 8,
        "dropout": 0.1,
        "item_embedding_dim": 256,
        "style_embedding_dim": 128,
        "max_outfit_size": 8
    })
    
    tracker.log_training_config({
        "epochs": 30,
        "batch_size": 32,
        "learning_rate": 2e-4,
        "optimizer": "AdamW"
    })
    
    tracker.log_data_config({
        "dataset_size": 50000,
        "train_size": 40000,
        "val_size": 5000,
        "test_size": 5000,
        "style_categories": ["casual", "formal", "business", "party"]
    })
    
    # Simulate training
    for epoch in range(3):
        tracker.log_training_epoch(
            epoch=epoch,
            train_loss=0.5 - epoch * 0.1,
            val_loss=0.6 - epoch * 0.08,
            learning_rate=2e-4 * (0.5 ** epoch)
        )
        
        # Log recommendation metrics
        tracker.log_recommendation_metrics({
            "ndcg@10": 0.3 + epoch * 0.05,
            "hit_rate@10": 0.25 + epoch * 0.04,
            "diversity": 0.7 + epoch * 0.02,
            "coverage": 0.6 + epoch * 0.03,
            "style_consistency": 0.8 + epoch * 0.01
        }, step=epoch)
    
    # Log model info
    tracker.log_model_size_info({
        "total_parameters": 15000000,
        "trainable_parameters": 15000000,
        "model_size_mb": 60.0,
        "inference_time_ms": 50.0
    })
    
    # Save example recommendations
    examples = [
        {
            "input_items": ["jeans", "t-shirt"],
            "recommended_items": ["sneakers", "jacket", "watch"],
            "style_score": 0.85,
            "user_rating": 4
        }
    ]
    tracker.save_recommendation_examples(examples)
    
    # Finish experiment
    tracker.finish("completed")
    
    print(f"Experiment completed: {tracker.run_name}")
    print(f"Best metrics: {tracker.get_best_metrics()}")
