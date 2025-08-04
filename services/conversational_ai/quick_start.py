#!/usr/bin/env python3
"""
Aura Fashion Assistant - Quick Start Script
==========================================

Bu script, QLoRA fine-tuning işlemini hızlı bir şekilde başlatmak için hazırlanmıştır.

Usage:
    python quick_start.py [--config CONFIG_NAME] [--test]
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from loguru import logger

def check_requirements():
    """Check if all requirements are met"""
    logger.info("🔍 Checking requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8+ required")
        return False
    
    # Check if running in appropriate environment
    try:
        import torch
        import transformers
        import peft
        import trl
        logger.success("✅ Required packages found")
    except ImportError as e:
        logger.error(f"❌ Missing package: {e}")
        logger.info("Please run: pip install -r requirements.txt")
        return False
    
    # Check CUDA
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.success(f"✅ CUDA available - GPU Memory: {gpu_memory:.1f}GB")
        
        if gpu_memory < 15:
            logger.warning("⚠️ Less than 16GB GPU memory detected")
            logger.info("Consider using memory-optimized config")
    else:
        logger.warning("⚠️ CUDA not available - CPU-only mode")
    
    return True


def check_dataset():
    """Check if dataset exists"""
    logger.info("📊 Checking dataset...")
    
    data_dir = Path("data")
    dataset_path = data_dir / "womens_clothing_reviews.csv"
    
    if dataset_path.exists():
        logger.success(f"✅ Dataset found: {dataset_path}")
        return str(dataset_path)
    else:
        logger.warning("⚠️ Dataset not found")
        logger.info("Please download Women's Clothing Reviews dataset from Kaggle:")
        logger.info("https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews")
        logger.info(f"And place it at: {dataset_path}")
        
        # Create sample dataset for testing
        logger.info("Creating sample dataset for testing...")
        create_sample_dataset(dataset_path)
        return str(dataset_path)


def create_sample_dataset(output_path: Path):
    """Create a small sample dataset for testing"""
    import pandas as pd
    
    sample_data = {
        "Review Text": [
            "This dress is absolutely beautiful! The quality is amazing and it fits perfectly. The fabric is soft and comfortable, and the color is exactly as shown in the picture.",
            "Great shoes, very comfortable and stylish. I love the color! Perfect for both casual and formal occasions. The heel height is just right.",
            "The jacket is okay but the material feels cheap. Not worth the price. I expected better quality for this brand. The fit is also a bit strange.",
            "Perfect fit and excellent quality. Highly recommend this brand! The customer service was also outstanding. Will definitely buy again.",
            "Love this sweater! So cozy and warm. The design is elegant and timeless. Perfect for winter weather. The size runs true to fit.",
            "Beautiful blouse with intricate details. The embroidery work is exquisite. However, it requires delicate care which can be inconvenient.",
            "These jeans are amazing! Perfect fit, great quality denim, and the wash is beautiful. They're comfortable for all-day wear.",
            "The handbag is stylish but smaller than expected. The leather quality is good but the color was slightly different from the website photos.",
            "Excellent workout clothes! The fabric is breathable and moisture-wicking. Perfect for yoga and running. The fit is flattering too.",
            "This skirt is gorgeous and versatile. Can be dressed up or down. The length is perfect and the fabric doesn't wrinkle easily."
        ],
        "Title": [
            "Beautiful Dress",
            "Comfortable Stylish Shoes",
            "Mediocre Jacket",
            "Excellent Quality Brand",
            "Cozy Winter Sweater",
            "Elegant Detailed Blouse",
            "Perfect Fit Jeans",
            "Stylish Small Handbag",
            "Great Workout Clothes",
            "Versatile Gorgeous Skirt"
        ]
    }
    
    df = pd.DataFrame(sample_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.success(f"✅ Sample dataset created: {output_path} ({len(df)} samples)")


def get_config_args(config_name: str):
    """Get command line arguments for different configurations"""
    
    configs = {
        "basic": [
            "--num_train_epochs", "3",
            "--per_device_train_batch_size", "4",
            "--gradient_accumulation_steps", "4",
            "--learning_rate", "2e-4",
            "--lora_r", "16",
            "--lora_alpha", "32",
            "--bf16"
        ],
        
        "memory_optimized": [
            "--num_train_epochs", "2",
            "--per_device_train_batch_size", "2",
            "--gradient_accumulation_steps", "8",
            "--learning_rate", "1e-4",
            "--max_seq_length", "512",
            "--lora_r", "8",
            "--lora_alpha", "16",
            "--bf16"
        ],
        
        "fast": [
            "--num_train_epochs", "1",
            "--per_device_train_batch_size", "4",
            "--gradient_accumulation_steps", "2",
            "--learning_rate", "5e-4",
            "--max_steps", "100",
            "--max_seq_length", "512",
            "--lora_r", "8",
            "--lora_alpha", "16",
            "--bf16"
        ],
        
        "production": [
            "--num_train_epochs", "5",
            "--per_device_train_batch_size", "8",
            "--gradient_accumulation_steps", "2",
            "--learning_rate", "1e-4",
            "--max_seq_length", "1024",
            "--lora_r", "32",
            "--lora_alpha", "64",
            "--warmup_steps", "100",
            "--weight_decay", "0.01",
            "--bf16"
        ]
    }
    
    return configs.get(config_name, configs["basic"])


def run_training(config_name: str, dataset_path: str, test_mode: bool = False):
    """Run the fine-tuning process"""
    logger.info(f"🚀 Starting fine-tuning with {config_name} configuration...")
    
    # Build command
    cmd = [sys.executable, "src/finetune.py"]
    
    # Add dataset path
    cmd.extend(["--dataset_path", dataset_path])
    
    # Add configuration arguments
    config_args = get_config_args(config_name)
    cmd.extend(config_args)
    
    # Test mode modifications
    if test_mode:
        logger.info("🧪 Running in test mode - reducing training time")
        # Override some settings for faster testing
        cmd.extend([
            "--max_steps", "20",
            "--logging_steps", "5",
            "--save_steps", "10",
            "--eval_steps", "10"
        ])
    
    # Show command
    logger.info("📝 Training command:")
    logger.info(" ".join(cmd))
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        logger.success("🎉 Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Training failed with exit code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        logger.warning("⚠️ Training interrupted by user")
        return False


def main():
    parser = argparse.ArgumentParser(description="Quick Start for Aura QLoRA Fine-Tuning")
    parser.add_argument(
        "--config", 
        choices=["basic", "memory_optimized", "fast", "production"],
        default="fast",
        help="Training configuration to use"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (very short training)"
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip requirement checks"
    )
    
    args = parser.parse_args()
    
    logger.info("🎯 Aura Fashion Assistant - Quick Start")
    logger.info("=" * 50)
    
    # Check requirements
    if not args.skip_checks:
        if not check_requirements():
            logger.error("❌ Requirements not met. Please install dependencies.")
            sys.exit(1)
    
    # Check dataset
    dataset_path = check_dataset()
    
    # Configuration info
    logger.info(f"📋 Using configuration: {args.config}")
    if args.test:
        logger.info("🧪 Test mode: Enabled (fast training)")
    
    config_descriptions = {
        "basic": "Balanced performance and quality",
        "memory_optimized": "For GPUs with <16GB memory",
        "fast": "Quick testing and prototyping",
        "production": "Best quality for production use"
    }
    
    logger.info(f"📖 Config description: {config_descriptions[args.config]}")
    
    # Confirm before starting
    if not args.test:
        response = input("\nProceed with training? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            logger.info("Training cancelled.")
            sys.exit(0)
    
    # Run training
    success = run_training(args.config, dataset_path, args.test)
    
    if success:
        logger.success("\n🎉 Fine-tuning completed successfully!")
        logger.info("📁 Check saved_models/ directory for the trained model")
        logger.info("📖 See src/README.md for usage instructions")
    else:
        logger.error("\n❌ Fine-tuning failed. Check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
