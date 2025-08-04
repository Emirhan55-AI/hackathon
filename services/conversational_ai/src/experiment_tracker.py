"""
Experiment Tracker for Conversational AI Service
Specialized experiment tracking for QLoRA fine-tuning and RAG experiments
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid


class ConversationalExperimentTracker:
    """
    Specialized experiment tracker for conversational AI experiments.
    Includes metrics specific to language model fine-tuning and RAG systems.
    """
    
    def __init__(self, 
                 experiment_name: str = "conversational_ai",
                 run_name: Optional[str] = None,
                 base_dir: str = "experiments"):
        """Initialize conversational AI experiment tracker."""
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        
        # Create unique run ID
        self.run_id = str(uuid.uuid4())[:8]
        self.run_name = run_name or f"conv_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.run_id}"
        
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
        self.generation_metrics = {}
        self.rag_metrics = {}
        
        # Create run metadata
        self.metadata = {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "model_type": "conversational_ai",
            "training_type": None  # Will be set to "qlora", "rag", or "hybrid"
        }
        
        self._save_metadata()
        self.log("Conversational AI experiment started", level="INFO")
    
    def log_qlora_config(self, config: Dict[str, Any]) -> None:
        """Log QLoRA fine-tuning configuration."""
        qlora_config = {
            "base_model_name": config.get("base_model_name", "meta-llama/Llama-2-7b-chat-hf"),
            "model_max_length": config.get("model_max_length", 4096),
            "lora_r": config.get("lora_r", 64),
            "lora_alpha": config.get("lora_alpha", 16),
            "lora_dropout": config.get("lora_dropout", 0.1),
            "lora_target_modules": config.get("lora_target_modules", []),
            "load_in_4bit": config.get("load_in_4bit", True),
            "bnb_4bit_use_double_quant": config.get("bnb_4bit_use_double_quant", True),
            "bnb_4bit_quant_type": config.get("bnb_4bit_quant_type", "nf4"),
            "bnb_4bit_compute_dtype": config.get("bnb_4bit_compute_dtype", "float16")
        }
        
        self.params.update(qlora_config)
        self.metadata["training_type"] = "qlora"
        self._save_params()
        self._save_metadata()
        self.log(f"Logged QLoRA configuration", level="INFO")
    
    def log_rag_config(self, config: Dict[str, Any]) -> None:
        """Log RAG system configuration."""
        rag_config = {
            "embedding_model": config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            "vector_store_type": config.get("vector_store_type", "faiss"),
            "chunk_size": config.get("chunk_size", 1000),
            "chunk_overlap": config.get("chunk_overlap", 200),
            "vector_dim": config.get("vector_dim", 384),
            "similarity_metric": config.get("similarity_metric", "cosine"),
            "retrieval_k": config.get("retrieval_k", 5),
            "retrieval_threshold": config.get("retrieval_threshold", 0.7),
            "rerank_enabled": config.get("rerank_enabled", False),
            "rerank_model": config.get("rerank_model", "")
        }
        
        self.params.update(rag_config)
        if self.metadata["training_type"] == "qlora":
            self.metadata["training_type"] = "hybrid"
        else:
            self.metadata["training_type"] = "rag"
        self._save_params()
        self._save_metadata()
        self.log(f"Logged RAG configuration", level="INFO")
    
    def log_training_config(self, config: Dict[str, Any]) -> None:
        """Log training configuration."""
        training_config = {
            "epochs": config.get("epochs", 3),
            "batch_size": config.get("batch_size", 1),
            "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 16),
            "learning_rate": config.get("learning_rate", 2e-4),
            "max_grad_norm": config.get("max_grad_norm", 0.3),
            "warmup_ratio": config.get("warmup_ratio", 0.03),
            "lr_scheduler_type": config.get("lr_scheduler_type", "cosine"),
            "optimizer": config.get("optimizer", "paged_adamw_32bit"),
            "weight_decay": config.get("weight_decay", 0.001),
            "save_steps": config.get("save_steps", 100),
            "eval_steps": config.get("eval_steps", 50),
            "logging_steps": config.get("logging_steps", 10),
            "max_steps": config.get("max_steps", -1)
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
            "max_length": config.get("max_length", 2048),
            "instruction_format": config.get("instruction_format", "alpaca"),
            "context_window": config.get("context_window", 4096),
            "truncation_strategy": config.get("truncation_strategy", "right"),
            "padding_strategy": config.get("padding_strategy", "right"),
            "special_tokens": config.get("special_tokens", {}),
            "knowledge_base_size": config.get("knowledge_base_size", 0),
            "document_count": config.get("document_count", 0)
        }
        
        self.params.update(data_config)
        self._save_params()
        self.log(f"Logged data configuration", level="INFO")
    
    def log_generation_metrics(self, 
                             metrics: Dict[str, float], 
                             step: Optional[int] = None) -> None:
        """Log text generation metrics."""
        timestamp = time.time()
        
        generation_metrics = {
            "perplexity": metrics.get("perplexity", 0.0),
            "bleu_score": metrics.get("bleu_score", 0.0),
            "rouge_1": metrics.get("rouge_1", 0.0),
            "rouge_2": metrics.get("rouge_2", 0.0),
            "rouge_l": metrics.get("rouge_l", 0.0),
            "meteor": metrics.get("meteor", 0.0),
            "bert_score": metrics.get("bert_score", 0.0),
            "semantic_similarity": metrics.get("semantic_similarity", 0.0),
            "response_length": metrics.get("response_length", 0.0),
            "response_time_ms": metrics.get("response_time_ms", 0.0),
            "tokens_per_second": metrics.get("tokens_per_second", 0.0)
        }
        
        # Log metrics
        for key, value in generation_metrics.items():
            if value > 0:  # Only log non-zero metrics
                if key not in self.generation_metrics:
                    self.generation_metrics[key] = []
                
                metric_entry = {
                    "value": value,
                    "timestamp": timestamp,
                    "step": step
                }
                self.generation_metrics[key].append(metric_entry)
        
        self._save_generation_metrics()
        self.log(f"Logged generation metrics: {list(generation_metrics.keys())}", level="INFO")
    
    def log_rag_metrics(self, 
                       metrics: Dict[str, float], 
                       step: Optional[int] = None) -> None:
        """Log RAG-specific metrics."""
        timestamp = time.time()
        
        rag_metrics = {
            "retrieval_precision": metrics.get("retrieval_precision", 0.0),
            "retrieval_recall": metrics.get("retrieval_recall", 0.0),
            "retrieval_f1": metrics.get("retrieval_f1", 0.0),
            "context_relevance": metrics.get("context_relevance", 0.0),
            "answer_relevance": metrics.get("answer_relevance", 0.0),
            "context_utilization": metrics.get("context_utilization", 0.0),
            "hallucination_rate": metrics.get("hallucination_rate", 0.0),
            "knowledge_coverage": metrics.get("knowledge_coverage", 0.0),
            "retrieval_time_ms": metrics.get("retrieval_time_ms", 0.0),
            "embedding_time_ms": metrics.get("embedding_time_ms", 0.0),
            "rerank_time_ms": metrics.get("rerank_time_ms", 0.0),
            "total_rag_time_ms": metrics.get("total_rag_time_ms", 0.0)
        }
        
        # Log metrics
        for key, value in rag_metrics.items():
            if value > 0:  # Only log non-zero metrics
                if key not in self.rag_metrics:
                    self.rag_metrics[key] = []
                
                metric_entry = {
                    "value": value,
                    "timestamp": timestamp,
                    "step": step
                }
                self.rag_metrics[key].append(metric_entry)
        
        self._save_rag_metrics()
        self.log(f"Logged RAG metrics: {list(rag_metrics.keys())}", level="INFO")
    
    def log_fashion_specific_metrics(self, 
                                   metrics: Dict[str, float], 
                                   step: Optional[int] = None) -> None:
        """Log fashion domain-specific metrics."""
        timestamp = time.time()
        
        fashion_metrics = {
            "style_accuracy": metrics.get("style_accuracy", 0.0),
            "color_accuracy": metrics.get("color_accuracy", 0.0),
            "trend_awareness": metrics.get("trend_awareness", 0.0),
            "seasonal_appropriateness": metrics.get("seasonal_appropriateness", 0.0),
            "occasion_matching": metrics.get("occasion_matching", 0.0),
            "brand_knowledge": metrics.get("brand_knowledge", 0.0),
            "size_recommendation_accuracy": metrics.get("size_recommendation_accuracy", 0.0),
            "budget_awareness": metrics.get("budget_awareness", 0.0),
            "sustainability_consideration": metrics.get("sustainability_consideration", 0.0),
            "personalization_score": metrics.get("personalization_score", 0.0)
        }
        
        # Log as regular metrics
        for key, value in fashion_metrics.items():
            if value > 0:
                if key not in self.metrics:
                    self.metrics[key] = []
                
                metric_entry = {
                    "value": value,
                    "timestamp": timestamp,
                    "step": step
                }
                self.metrics[key].append(metric_entry)
        
        self._save_metrics()
        self.log(f"Logged fashion-specific metrics: {list(fashion_metrics.keys())}", level="INFO")
    
    def log_training_step(self, 
                         step: int,
                         epoch: float,
                         loss: float,
                         learning_rate: float,
                         grad_norm: Optional[float] = None,
                         additional_metrics: Optional[Dict[str, float]] = None) -> None:
        """Log training metrics for a specific step."""
        step_metrics = {
            "step": step,
            "epoch": epoch,
            "train_loss": loss,
            "learning_rate": learning_rate
        }
        
        if grad_norm is not None:
            step_metrics["grad_norm"] = grad_norm
        
        if additional_metrics:
            step_metrics.update(additional_metrics)
        
        # Log metrics with step
        for key, value in step_metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            
            metric_entry = {
                "value": value,
                "timestamp": time.time(),
                "step": step
            }
            self.metrics[key].append(metric_entry)
        
        self._save_metrics()
        
        if step % 100 == 0:  # Log every 100 steps
            self.log(f"Step {step}: loss={loss:.4f}, lr={learning_rate:.6f}", level="INFO")
    
    def log_evaluation_results(self, eval_results: Dict[str, Any]) -> None:
        """Log comprehensive evaluation results."""
        eval_summary = {
            "eval_timestamp": datetime.now().isoformat(),
            "eval_samples": eval_results.get("eval_samples", 0),
            "eval_loss": eval_results.get("eval_loss", 0.0),
            "eval_perplexity": eval_results.get("eval_perplexity", 0.0),
            "eval_runtime": eval_results.get("eval_runtime", 0.0),
            "eval_samples_per_second": eval_results.get("eval_samples_per_second", 0.0)
        }
        
        self.params.update(eval_summary)
        self._save_params()
        
        # Log individual metrics
        if "metrics" in eval_results:
            self.log_generation_metrics(eval_results["metrics"])
        
        self.log(f"Logged evaluation results: eval_loss={eval_summary['eval_loss']:.4f}", level="INFO")
    
    def save_conversation_examples(self, examples: List[Dict[str, Any]]) -> None:
        """Save example conversations for analysis."""
        examples_file = self.run_dir / "conversation_examples.json"
        with open(examples_file, "w") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        
        self.artifacts["conversation_examples"] = {
            "path": str(examples_file),
            "type": "examples",
            "count": len(examples),
            "timestamp": datetime.now().isoformat()
        }
        
        self._save_artifacts()
        self.log(f"Saved {len(examples)} conversation examples", level="INFO")
    
    def save_retrieval_examples(self, examples: List[Dict[str, Any]]) -> None:
        """Save example retrieval results for RAG analysis."""
        examples_file = self.run_dir / "retrieval_examples.json"
        with open(examples_file, "w") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        
        self.artifacts["retrieval_examples"] = {
            "path": str(examples_file),
            "type": "retrieval_examples",
            "count": len(examples),
            "timestamp": datetime.now().isoformat()
        }
        
        self._save_artifacts()
        self.log(f"Saved {len(examples)} retrieval examples", level="INFO")
    
    def log_model_size_info(self, model_info: Dict[str, Any]) -> None:
        """Log information about model size and memory usage."""
        size_info = {
            "base_model_parameters": model_info.get("base_model_parameters", 0),
            "trainable_parameters": model_info.get("trainable_parameters", 0),
            "lora_parameters": model_info.get("lora_parameters", 0),
            "model_size_gb": model_info.get("model_size_gb", 0.0),
            "gpu_memory_usage_gb": model_info.get("gpu_memory_usage_gb", 0.0),
            "inference_memory_gb": model_info.get("inference_memory_gb", 0.0),
            "quantization_enabled": model_info.get("quantization_enabled", False),
            "quantization_bits": model_info.get("quantization_bits", 16)
        }
        
        self.params.update(size_info)
        self._save_params()
        self.log("Logged model size information", level="INFO")
    
    def log_vector_store_info(self, vector_store_info: Dict[str, Any]) -> None:
        """Log information about the vector store."""
        vs_info = {
            "total_documents": vector_store_info.get("total_documents", 0),
            "total_chunks": vector_store_info.get("total_chunks", 0),
            "vector_store_size_mb": vector_store_info.get("vector_store_size_mb", 0.0),
            "embedding_dimension": vector_store_info.get("embedding_dimension", 0),
            "index_type": vector_store_info.get("index_type", ""),
            "build_time_seconds": vector_store_info.get("build_time_seconds", 0.0),
            "avg_chunk_length": vector_store_info.get("avg_chunk_length", 0.0)
        }
        
        self.params.update(vs_info)
        self._save_params()
        self.log("Logged vector store information", level="INFO")
    
    def finish(self, status: str = "completed") -> None:
        """Finish the experiment."""
        self.metadata["end_time"] = datetime.now().isoformat()
        self.metadata["duration"] = time.time() - self.start_time
        self.metadata["status"] = status
        
        self._save_metadata()
        self._save_summary()
        
        self.log(f"Conversational AI experiment finished with status: {status}", level="INFO")
    
    def get_best_metrics(self) -> Dict[str, float]:
        """Get the best values for each metric."""
        best_metrics = {}
        
        # For generation metrics, lower is better for loss and perplexity, higher for others
        lower_is_better = ["loss", "perplexity", "train_loss", "eval_loss", "hallucination_rate"]
        
        all_metrics = {**self.metrics, **self.generation_metrics, **self.rag_metrics}
        
        for metric_name, history in all_metrics.items():
            if not history:
                continue
                
            values = [entry["value"] for entry in history]
            
            if any(keyword in metric_name.lower() for keyword in lower_is_better):
                best_metrics[f"best_{metric_name}"] = min(values)
            else:
                best_metrics[f"best_{metric_name}"] = max(values)
        
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
    
    def _save_generation_metrics(self) -> None:
        """Save generation-specific metrics."""
        with open(self.run_dir / "generation_metrics.json", "w") as f:
            json.dump(self.generation_metrics, f, indent=2)
    
    def _save_rag_metrics(self) -> None:
        """Save RAG-specific metrics."""
        with open(self.run_dir / "rag_metrics.json", "w") as f:
            json.dump(self.rag_metrics, f, indent=2)
    
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
def create_conversational_tracker(run_name: Optional[str] = None) -> ConversationalExperimentTracker:
    """Create experiment tracker for conversational AI."""
    return ConversationalExperimentTracker(run_name=run_name)


# Example usage
if __name__ == "__main__":
    # Example conversational AI experiment
    tracker = create_conversational_tracker("qlora_llama2_fashion")
    
    # Log QLoRA configuration
    tracker.log_qlora_config({
        "base_model_name": "meta-llama/Llama-2-7b-chat-hf",
        "lora_r": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "load_in_4bit": True
    })
    
    # Log RAG configuration
    tracker.log_rag_config({
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "vector_store_type": "faiss",
        "chunk_size": 1000,
        "retrieval_k": 5
    })
    
    # Log training configuration
    tracker.log_training_config({
        "epochs": 3,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 2e-4
    })
    
    # Simulate training
    for step in range(0, 300, 50):
        tracker.log_training_step(
            step=step,
            epoch=step / 100,
            loss=2.0 - step * 0.005,
            learning_rate=2e-4 * (0.95 ** (step // 50)),
            grad_norm=1.0 - step * 0.002
        )
    
    # Log generation metrics
    tracker.log_generation_metrics({
        "perplexity": 15.2,
        "bleu_score": 0.45,
        "rouge_l": 0.38,
        "response_time_ms": 250
    })
    
    # Log RAG metrics
    tracker.log_rag_metrics({
        "retrieval_precision": 0.75,
        "context_relevance": 0.82,
        "answer_relevance": 0.78,
        "retrieval_time_ms": 45
    })
    
    # Log fashion-specific metrics
    tracker.log_fashion_specific_metrics({
        "style_accuracy": 0.85,
        "color_accuracy": 0.78,
        "trend_awareness": 0.72,
        "personalization_score": 0.68
    })
    
    # Save examples
    conversation_examples = [
        {
            "user": "What should I wear to a summer wedding?",
            "assistant": "For a summer wedding, I recommend a light, breathable dress in a floral or pastel pattern...",
            "retrieval_context": "Summer wedding attire guidelines...",
            "rating": 4.5
        }
    ]
    tracker.save_conversation_examples(conversation_examples)
    
    # Finish experiment
    tracker.finish("completed")
    
    print(f"Experiment completed: {tracker.run_name}")
    print(f"Best metrics: {tracker.get_best_metrics()}")
