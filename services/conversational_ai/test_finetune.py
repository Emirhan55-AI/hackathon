"""
Test Script for Aura QLoRA Fine-Tuning Module
==============================================

Bu script, fine-tuning modülünün doğru çalışıp çalışmadığını test eder.

Usage:
    python test_finetune.py [--quick] [--verbose]
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from loguru import logger

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

def test_imports():
    """Test all required imports"""
    logger.info("🧪 Testing imports...")
    
    try:
        import transformers
        import peft
        import trl
        import bitsandbytes
        import datasets
        logger.success("✅ All required packages imported successfully")
        
        # Version check
        logger.info(f"📦 Package versions:")
        logger.info(f"  - transformers: {transformers.__version__}")
        logger.info(f"  - peft: {peft.__version__}")
        logger.info(f"  - trl: {trl.__version__}")
        logger.info(f"  - torch: {torch.__version__}")
        
        return True
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False


def test_cuda():
    """Test CUDA availability"""
    logger.info("🧪 Testing CUDA...")
    
    if torch.cuda.is_available():
        logger.success("✅ CUDA is available")
        logger.info(f"  - Device count: {torch.cuda.device_count()}")
        logger.info(f"  - Current device: {torch.cuda.current_device()}")
        logger.info(f"  - Device name: {torch.cuda.get_device_name(0)}")
        
        # Memory info
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        logger.info(f"  - Memory allocated: {memory_allocated:.2f} GB")
        logger.info(f"  - Memory reserved: {memory_reserved:.2f} GB") 
        logger.info(f"  - Total memory: {total_memory:.2f} GB")
        
        return True
    else:
        logger.warning("⚠️ CUDA is not available - CPU-only mode")
        return False


def test_model_loading():
    """Test model loading with quantization"""
    logger.info("🧪 Testing model loading...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        
        # Test tokenizer loading
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/DialoGPT-small",  # Smaller model for testing
            trust_remote_code=True
        )
        logger.success("✅ Tokenizer loaded successfully")
        
        # Test quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        logger.success("✅ Quantization config created")
        
        # Test small model loading (skip if no CUDA)
        if torch.cuda.is_available():
            logger.info("Loading small test model with quantization...")
            model = AutoModelForCausalLM.from_pretrained(
                "microsoft/DialoGPT-small",
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            logger.success("✅ Quantized model loaded successfully")
            
            # Clean up
            del model
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading error: {e}")
        return False


def test_peft():
    """Test PEFT functionality"""
    logger.info("🧪 Testing PEFT...")
    
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import AutoModelForCausalLM
        
        # Create LoRA config
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn"],  # For DialoGPT
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        logger.success("✅ LoRA config created")
        
        # Test with small model (skip if no CUDA)
        if torch.cuda.is_available():
            logger.info("Testing PEFT model creation...")
            
            # Load small model
            model = AutoModelForCausalLM.from_pretrained(
                "microsoft/DialoGPT-small",
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Apply PEFT
            peft_model = get_peft_model(model, lora_config)
            logger.success("✅ PEFT model created successfully")
            
            # Show trainable parameters
            peft_model.print_trainable_parameters()
            
            # Clean up
            del peft_model, model
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ PEFT test error: {e}")
        return False


def test_dataset():
    """Test dataset creation"""
    logger.info("🧪 Testing dataset creation...")
    
    try:
        from datasets import Dataset
        import pandas as pd
        
        # Create sample data
        sample_data = {
            "Review Text": [
                "This dress is absolutely beautiful! The quality is amazing and it fits perfectly.",
                "Great shoes, very comfortable and stylish. I love the color!",
                "The jacket is okay but the material feels cheap. Not worth the price.",
                "Perfect fit and excellent quality. Highly recommend this brand!"
            ],
            "Title": [
                "Beautiful Dress",
                "Comfortable Shoes", 
                "Mediocre Jacket",
                "Excellent Quality"
            ]
        }
        
        # Create DataFrame
        df = pd.DataFrame(sample_data)
        logger.info(f"Sample DataFrame created: {len(df)} rows")
        
        # Convert to HF Dataset
        dataset = Dataset.from_pandas(df)
        logger.success(f"✅ HuggingFace Dataset created: {len(dataset)} samples")
        
        # Test train/test split
        dataset_dict = dataset.train_test_split(test_size=0.2, random_state=42)
        logger.success(f"✅ Train/test split: {len(dataset_dict['train'])}/{len(dataset_dict['test'])}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset test error: {e}")
        return False


def test_finetune_script():
    """Test finetune script import and basic functionality"""
    logger.info("🧪 Testing finetune script...")
    
    try:
        # Test import
        from finetune import FineTuningConfig, parse_arguments, setup_logging
        logger.success("✅ Finetune script imported successfully")
        
        # Test config creation
        config = FineTuningConfig()
        logger.success("✅ Default config created")
        logger.info(f"  - Model: {config.model_id}")
        logger.info(f"  - Output: {config.new_model_name}")
        logger.info(f"  - LoRA r: {config.lora_r}")
        
        # Test logging setup
        setup_logging(debug=True)
        logger.success("✅ Logging setup successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Finetune script test error: {e}")
        return False


def run_tests(quick: bool = False, verbose: bool = False):
    """Run all tests"""
    logger.info("🚀 Starting Aura QLoRA Fine-Tuning Tests")
    logger.info("=" * 50)
    
    if verbose:
        logger.info("Running in verbose mode")
    
    results = {}
    
    # Basic tests
    results["imports"] = test_imports()
    results["cuda"] = test_cuda()
    results["dataset"] = test_dataset()
    results["finetune_script"] = test_finetune_script()
    
    if not quick:
        # More comprehensive tests
        results["model_loading"] = test_model_loading()
        results["peft"] = test_peft()
    else:
        logger.info("⚡ Quick mode - skipping heavy tests")
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 TEST RESULTS:")
    logger.info("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:.<20} {status}")
        if result:
            passed += 1
    
    logger.info("=" * 50)
    logger.info(f"📈 SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        logger.success("🎉 All tests passed! Fine-tuning environment is ready.")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the requirements.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Aura QLoRA Fine-Tuning Module")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    success = run_tests(quick=args.quick, verbose=args.verbose)
    
    if success:
        logger.info("\n🚀 Ready to start fine-tuning!")
        logger.info("Next steps:")
        logger.info("1. Download women's clothing reviews dataset")
        logger.info("2. Place CSV file in data/ folder")
        logger.info("3. Run: python src/finetune.py --help")
    else:
        logger.error("\n❌ Environment not ready. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
