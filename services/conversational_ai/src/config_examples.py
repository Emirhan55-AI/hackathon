# Aura Fashion Assistant Fine-Tuning Configuration Examples
# =========================================================

try:
    from src.rag_service import RAGConfig
except ImportError:
    try:
        from rag_service import RAGConfig
    except ImportError:
        # Graceful degradation
        RAGConfig = None

# Basic Training Configuration
basic_config = {
    "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
    "new_model_name": "llama3-8b-aura-fashion-qlora",
    "dataset_path": "./data/womens_clothing_reviews.csv",
    "output_dir": "./saved_models",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "max_seq_length": 1024,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bf16": True
}

# Memory Optimized Configuration (for smaller GPUs)
memory_optimized_config = {
    "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
    "new_model_name": "llama3-8b-aura-fashion-qlora-memory-opt",
    "dataset_path": "./data/womens_clothing_reviews.csv",
    "output_dir": "./saved_models",
    "num_train_epochs": 2,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 1e-4,
    "max_seq_length": 512,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "bf16": True
}

# Fast Prototyping Configuration
fast_config = {
    "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
    "new_model_name": "llama3-8b-aura-fashion-qlora-fast",
    "dataset_path": "./data/womens_clothing_reviews.csv",
    "output_dir": "./saved_models",
    "num_train_epochs": 1,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "learning_rate": 5e-4,
    "max_steps": 200,
    "max_seq_length": 512,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "bf16": True
}

# High Quality Configuration (for production)
production_config = {
    "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
    "new_model_name": "llama3-8b-aura-fashion-qlora-production",
    "dataset_path": "./data/womens_clothing_reviews.csv",
    "output_dir": "./saved_models",
    "num_train_epochs": 5,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "learning_rate": 1e-4,
    "max_seq_length": 1024,
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.03,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "bf16": True,
    "report_to": "wandb",
    "early_stopping_patience": 5
}

"""
Usage Examples:

1. Basic Training:
python src/finetune.py \
    --model_id meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset_path ./data/womens_clothing_reviews.csv \
    --output_dir ./saved_models \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --bf16

2. Memory Optimized (for smaller GPUs):
python src/finetune.py \
    --model_id meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset_path ./data/womens_clothing_reviews.csv \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_seq_length 512 \
    --lora_r 8 \
    --lora_alpha 16 \
    --bf16

3. Fast Prototyping:
python src/finetune.py \
    --model_id meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset_path ./data/womens_clothing_reviews.csv \
    --max_steps 200 \
    --per_device_train_batch_size 4 \
    --max_seq_length 512 \
    --lora_r 8 \
    --learning_rate 5e-4 \
    --bf16

4. Production Quality:
python src/finetune.py \
    --model_id meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset_path ./data/womens_clothing_reviews.csv \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --lora_r 32 \
    --lora_alpha 64 \
    --learning_rate 1e-4 \
    --warmup_steps 100 \
    --report_to wandb \
    --bf16

Hardware Requirements:
- GPU with at least 16GB VRAM (RTX 4090, A100, etc.)
- For smaller GPUs (8-12GB): Use memory_optimized_config
- CPU RAM: At least 32GB recommended
- Storage: At least 50GB free space

Dataset Requirements:
- CSV file with 'Review Text' and 'Title' columns
- Minimum 1000 samples recommended
- Maximum 50000 samples for best results
- Clean, fashion-related text data
"""
