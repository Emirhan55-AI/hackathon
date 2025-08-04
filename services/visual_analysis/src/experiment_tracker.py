"""
Experiment Tracker for Aura AI Platform
Simple experiment tracking system for ML model training and evaluation
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid


class ExperimentTracker:
    """
    Simple experiment tracking system for logging training metrics,
    hyperparameters, and model artifacts.
    """
    
    def __init__(self, 
                 experiment_name: str,
                 project_name: str = "aura-ai",
                 base_dir: str = "experiments"):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the experiment (e.g., 'visual_analysis', 'outfit_recommendation')
            project_name: Name of the project
            base_dir: Base directory for storing experiment data
        """
        self.experiment_name = experiment_name
        self.project_name = project_name
        self.base_dir = Path(base_dir)
        
        # Create unique run ID
        self.run_id = str(uuid.uuid4())[:8]
        self.run_name = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.run_id}"
        
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
        
        # Create run metadata
        self.metadata = {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "experiment_name": experiment_name,
            "project_name": project_name,
            "start_time": datetime.now().isoformat(),
            "status": "running"
        }
        
        self._save_metadata()
        self.log("Experiment started", level="INFO")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters and configuration."""
        self.params.update(params)
        self._save_params()
        self.log(f"Logged parameters: {list(params.keys())}", level="INFO")
    
    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter."""
        self.params[key] = value
        self._save_params()
        self.log(f"Logged parameter: {key} = {value}", level="INFO")
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
        """Log multiple metrics."""
        timestamp = time.time()
        
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            
            metric_entry = {
                "value": value,
                "timestamp": timestamp,
                "step": step
            }
            self.metrics[key].append(metric_entry)
        
        self._save_metrics()
        self.log(f"Logged metrics: {list(metrics.keys())}", level="INFO")
    
    def log_metric(self, key: str, value: Union[float, int], step: Optional[int] = None) -> None:
        """Log a single metric."""
        self.log_metrics({key: value}, step=step)
    
    def log_artifact(self, artifact_path: str, artifact_type: str = "file") -> None:
        """Log an artifact (model file, plot, etc.)."""
        artifact_name = os.path.basename(artifact_path)
        
        # Copy artifact to run directory
        import shutil
        dest_path = self.run_dir / "artifacts" / artifact_name
        dest_path.parent.mkdir(exist_ok=True)
        
        if os.path.exists(artifact_path):
            shutil.copy2(artifact_path, dest_path)
            
            self.artifacts[artifact_name] = {
                "path": str(dest_path),
                "original_path": artifact_path,
                "type": artifact_type,
                "timestamp": datetime.now().isoformat()
            }
            
            self._save_artifacts()
            self.log(f"Logged artifact: {artifact_name}", level="INFO")
        else:
            self.log(f"Artifact not found: {artifact_path}", level="WARNING")
    
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
    
    def finish(self, status: str = "completed") -> None:
        """Finish the experiment."""
        self.metadata["end_time"] = datetime.now().isoformat()
        self.metadata["duration"] = time.time() - self.start_time
        self.metadata["status"] = status
        
        self._save_metadata()
        self._save_summary()
        
        self.log(f"Experiment finished with status: {status}", level="INFO")
    
    def get_metric_history(self, metric_name: str) -> List[Dict]:
        """Get the history of a specific metric."""
        return self.metrics.get(metric_name, [])
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """Get the latest value for each metric."""
        latest_metrics = {}
        for metric_name, history in self.metrics.items():
            if history:
                latest_metrics[metric_name] = history[-1]["value"]
        return latest_metrics
    
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
    
    def _save_artifacts(self) -> None:
        """Save artifacts metadata."""
        with open(self.run_dir / "artifacts.json", "w") as f:
            json.dump(self.artifacts, f, indent=2)
    
    def _save_summary(self) -> None:
        """Save experiment summary."""
        summary = {
            "metadata": self.metadata,
            "params": self.params,
            "latest_metrics": self.get_latest_metrics(),
            "artifacts": list(self.artifacts.keys()),
            "total_logs": len(self.logs)
        }
        
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Also save to experiment-level summary
        experiment_summary_file = self.experiment_dir / "experiments_summary.jsonl"
        with open(experiment_summary_file, "a") as f:
            f.write(json.dumps(summary) + "\n")


class MetricsLogger:
    """Helper class for logging training metrics during model training."""
    
    def __init__(self, tracker: ExperimentTracker):
        self.tracker = tracker
        self.epoch_metrics = {}
        self.batch_metrics = {}
        
    def log_epoch_start(self, epoch: int) -> None:
        """Log the start of an epoch."""
        self.current_epoch = epoch
        self.epoch_metrics = {}
        self.tracker.log(f"Starting epoch {epoch}", level="INFO")
    
    def log_batch_metrics(self, metrics: Dict[str, float], batch_idx: int) -> None:
        """Log metrics for a single batch."""
        for key, value in metrics.items():
            if key not in self.batch_metrics:
                self.batch_metrics[key] = []
            self.batch_metrics[key].append(value)
        
        # Log to tracker with step as global step
        global_step = self.current_epoch * 1000 + batch_idx  # Approximate global step
        self.tracker.log_metrics(metrics, step=global_step)
    
    def log_epoch_end(self, metrics: Dict[str, float]) -> None:
        """Log metrics at the end of an epoch."""
        self.epoch_metrics.update(metrics)
        
        # Calculate epoch averages for batch metrics
        epoch_averages = {}
        for key, values in self.batch_metrics.items():
            if values:
                epoch_averages[f"epoch_avg_{key}"] = sum(values) / len(values)
        
        # Log epoch metrics
        all_epoch_metrics = {**metrics, **epoch_averages}
        self.tracker.log_metrics(all_epoch_metrics, step=self.current_epoch)
        
        # Clear batch metrics for next epoch
        self.batch_metrics = {}
        
        self.tracker.log(f"Completed epoch {self.current_epoch} - Metrics: {metrics}", level="INFO")


class ExperimentComparison:
    """Class for comparing multiple experiments."""
    
    def __init__(self, experiment_name: str, base_dir: str = "experiments"):
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.experiment_dir = self.base_dir / experiment_name
    
    def list_experiments(self) -> List[Dict]:
        """List all experiments in the experiment directory."""
        experiments = []
        
        if not self.experiment_dir.exists():
            return experiments
        
        for run_dir in self.experiment_dir.iterdir():
            if run_dir.is_dir():
                summary_file = run_dir / "summary.json"
                if summary_file.exists():
                    with open(summary_file) as f:
                        summary = json.load(f)
                        experiments.append(summary)
        
        return sorted(experiments, key=lambda x: x["metadata"]["start_time"], reverse=True)
    
    def get_best_experiment(self, metric_name: str, higher_is_better: bool = True) -> Optional[Dict]:
        """Get the best experiment based on a specific metric."""
        experiments = self.list_experiments()
        
        if not experiments:
            return None
        
        # Filter experiments that have the metric
        valid_experiments = [
            exp for exp in experiments 
            if metric_name in exp.get("latest_metrics", {})
        ]
        
        if not valid_experiments:
            return None
        
        # Sort by metric
        best_experiment = max(
            valid_experiments,
            key=lambda x: x["latest_metrics"][metric_name] if higher_is_better 
            else -x["latest_metrics"][metric_name]
        )
        
        return best_experiment
    
    def compare_experiments(self, metric_names: List[str]) -> List[Dict]:
        """Compare experiments across multiple metrics."""
        experiments = self.list_experiments()
        
        comparison = []
        for exp in experiments:
            exp_comparison = {
                "run_name": exp["metadata"]["run_name"],
                "run_id": exp["metadata"]["run_id"],
                "start_time": exp["metadata"]["start_time"],
                "status": exp["metadata"]["status"],
                "metrics": {}
            }
            
            for metric_name in metric_names:
                exp_comparison["metrics"][metric_name] = (
                    exp.get("latest_metrics", {}).get(metric_name, None)
                )
            
            comparison.append(exp_comparison)
        
        return comparison


# Convenience functions for different services
def create_visual_analysis_tracker(run_name: Optional[str] = None) -> ExperimentTracker:
    """Create experiment tracker for visual analysis service."""
    name = run_name or "visual_analysis"
    return ExperimentTracker(name, project_name="aura-ai-visual-analysis")


def create_outfit_tracker(run_name: Optional[str] = None) -> ExperimentTracker:
    """Create experiment tracker for outfit recommendation service."""
    name = run_name or "outfit_recommendation"
    return ExperimentTracker(name, project_name="aura-ai-outfit-recommendation")


def create_conversational_tracker(run_name: Optional[str] = None) -> ExperimentTracker:
    """Create experiment tracker for conversational AI service."""
    name = run_name or "conversational_ai"
    return ExperimentTracker(name, project_name="aura-ai-conversational-ai")


# Example usage
if __name__ == "__main__":
    # Example: Visual Analysis experiment
    tracker = create_visual_analysis_tracker("detr_fashion_detector_v1")
    
    # Log hyperparameters
    tracker.log_params({
        "model": "DETR",
        "backbone": "resnet50",
        "learning_rate": 1e-4,
        "batch_size": 8,
        "epochs": 50
    })
    
    # Simulate training loop
    metrics_logger = MetricsLogger(tracker)
    
    for epoch in range(3):  # Simulate 3 epochs
        metrics_logger.log_epoch_start(epoch)
        
        # Simulate batch training
        for batch in range(10):
            batch_metrics = {
                "loss": 0.5 - (epoch * 0.1) + (batch * 0.01),
                "accuracy": 0.7 + (epoch * 0.05) - (batch * 0.001)
            }
            metrics_logger.log_batch_metrics(batch_metrics, batch)
        
        # Log epoch metrics
        epoch_metrics = {
            "val_loss": 0.4 - (epoch * 0.08),
            "val_accuracy": 0.75 + (epoch * 0.04),
            "mAP": 0.6 + (epoch * 0.03)
        }
        metrics_logger.log_epoch_end(epoch_metrics)
    
    # Log artifacts
    tracker.log_artifact("model.pth", "model")
    tracker.log_artifact("training_plot.png", "plot")
    
    # Finish experiment
    tracker.finish("completed")
    
    print(f"Experiment completed: {tracker.run_name}")
    print(f"Latest metrics: {tracker.get_latest_metrics()}")
