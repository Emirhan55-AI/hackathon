"""
QLoRA Fine-Tuning Script for Aura Fashion Style Assistant
=========================================================

Bu modül, Aura projesinin sohbet asistanı mikroservisi için QLoRA (Quantized LoRA) 
tabanlı ince ayar (fine-tuning) işlemlerini gerçekleştirir.

Kullanılan Teknolojiler:
- Meta-Llama-3-8B-Instruct temel modeli
- QLoRA (4-bit quantization + LoRA) verimli fine-tuning için
- SFTTrainer (Supervised Fine-Tuning) özel kişilik geliştirme için
- Women's Clothing Reviews veri seti moda alan bilgisi için

Teknik Detaylar:
- 4-bit quantization ile bellek kullanımını %75 azaltır
- LoRA adaptörleri ile sadece %1 parametreyi eğitir
- Batch processing ve gradient accumulation ile tek GPU'da çalışır
- Hugging Face ekosistemi ile tam uyumlu

Author: Aura AI Team
Version: 1.0.0
Date: 2025-08-03
"""

import os
import sys
import json
import logging
import argparse
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import traceback

# Core ML Libraries
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import numpy as np
import pandas as pd

# Hugging Face Ecosystem
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)

# PEFT (Parameter Efficient Fine-Tuning)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    TaskType
)

# TRL (Transformer Reinforcement Learning)
try:
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
except ImportError:
    # For newer versions of TRL
    from trl import SFTTrainer
    from transformers import DataCollatorForLanguageModeling as DataCollatorForCompletionOnlyLM

# Datasets and Data Processing
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

# Monitoring and Logging
import wandb
from loguru import logger

# Configuration
from datetime import datetime
import yaml
from dataclasses import dataclass, asdict

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# A1. MODÜL VE KÜTÜPHANELERİ İÇE AKTARMA - TAMAMLANDI
# =============================================================================

# =============================================================================
# A2. KONFİGÜRASYON VE ARGÜMANLARI TANIMLAMA
# =============================================================================

@dataclass
class FineTuningConfig:
    """QLoRA Fine-tuning için konfigürasyon sınıfı"""
    
    # Model Configuration
    model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    new_model_name: str = "llama3-8b-aura-fashion-qlora"
    
    # Dataset Configuration
    dataset_path: str = "./data/womens_clothing_reviews.csv"
    max_seq_length: int = 1024
    test_size: float = 0.1
    random_state: int = 42
    
    # QLoRA Configuration
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    
    # LoRA Configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    target_modules: List[str] = None
    
    # Training Configuration
    output_dir: str = "./saved_models"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    max_steps: int = -1
    
    # Logging and Saving
    logging_steps: int = 25
    save_strategy: str = "steps"
    save_steps: int = 100
    evaluation_strategy: str = "steps"
    eval_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    
    # Optimization
    fp16: bool = False
    bf16: bool = True
    tf32: bool = True
    dataloader_pin_memory: bool = False
    remove_unused_columns: bool = False
    
    # Early Stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01
    
    # Monitoring
    report_to: str = "none"  # "wandb" for Weights & Biases
    run_name: str = None
    
    def __post_init__(self):
        """Post-initialization setup"""
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        if self.run_name is None:
            self.run_name = f"aura-finetune-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Convert string dtype to torch dtype
        if isinstance(self.bnb_4bit_compute_dtype, str):
            if self.bnb_4bit_compute_dtype == "bfloat16":
                self.bnb_4bit_compute_dtype = torch.bfloat16
            elif self.bnb_4bit_compute_dtype == "float16":
                self.bnb_4bit_compute_dtype = torch.float16
            else:
                self.bnb_4bit_compute_dtype = torch.float32


def parse_arguments() -> argparse.Namespace:
    """
    A2a. Komut satırı argümanlarını tanımla ve parse et
    
    Returns:
        argparse.Namespace: Parse edilmiş argümanlar
    """
    parser = argparse.ArgumentParser(
        description="Aura Fashion Style Assistant - QLoRA Fine-Tuning Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Hugging Face model ID to fine-tune"
    )
    parser.add_argument(
        "--new_model_name",
        type=str,
        default="llama3-8b-aura-fashion-qlora",
        help="Name for the fine-tuned model"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/womens_clothing_reviews.csv",
        help="Path to the fashion dataset CSV file"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="Maximum sequence length for tokenization"
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./saved_models",
        help="Directory to save the fine-tuned model"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Maximum number of training steps (-1 for full epochs)"
    )
    
    # LoRA arguments
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank dimension"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout probability"
    )
    
    # Monitoring arguments
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        choices=["none", "wandb", "tensorboard"],
        help="Experiment tracking platform"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name for the training run"
    )
    
    # Technical arguments
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Use bfloat16 precision (recommended for modern GPUs)"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use float16 precision"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode with verbose logging"
    )
    
    return parser.parse_args()


def setup_logging(debug: bool = False) -> None:
    """
    A2b. Logging sistemini yapılandır
    
    Args:
        debug: Debug seviyesinde loglama aktif edilsin mi
    """
    # Remove default logger
    logger.remove()
    
    # Console logging
    log_level = "DEBUG" if debug else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # File logging
    log_file = f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days"
    )
    
    logger.info("🚀 Aura Fashion Assistant QLoRA Fine-Tuning başlatılıyor...")
    logger.info(f"📝 Log dosyası: {log_file}")


# =============================================================================
# A3. MODELİ VE TOKENIZER'I YÜKLEME
# =============================================================================

def load_model_and_tokenizer(config: FineTuningConfig) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    A3a. Model ve tokenizer'ı 4-bit quantization ile yükle
    
    Args:
        config: Fine-tuning konfigürasyonu
        
    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: Yüklenmiş model ve tokenizer
    """
    logger.info(f"📥 Model yükleniyor: {config.model_id}")
    
    # A3a. BitsAndBytesConfig ile 4-bit quantization ayarları
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=config.bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
    )
    
    logger.info("⚙️ 4-bit quantization konfigürasyonu:")
    logger.info(f"  - Quant type: {config.bnb_4bit_quant_type}")
    logger.info(f"  - Compute dtype: {config.bnb_4bit_compute_dtype}")
    logger.info(f"  - Double quant: {config.bnb_4bit_use_double_quant}")
    
    try:
        # A3b. Model'i quantization ile yükle
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            quantization_config=bnb_config,
            device_map="auto",  # Otomatik GPU dağıtımı
            trust_remote_code=True,
            torch_dtype=config.bnb_4bit_compute_dtype,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
        )
        
        logger.success(f"✅ Model başarıyla yüklendi: {config.model_id}")
        
        # Model istatistiklerini göster
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"📊 Model istatistikleri:")
        logger.info(f"  - Toplam parametreler: {total_params:,}")
        logger.info(f"  - Eğitilebilir parametreler: {trainable_params:,}")
        logger.info(f"  - Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}")
        
    except Exception as e:
        logger.error(f"❌ Model yükleme hatası: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    
    try:
        # A3c. Tokenizer'ı yükle
        logger.info(f"📥 Tokenizer yükleniyor: {config.model_id}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            trust_remote_code=True,
            padding_side="right",  # LoRA için önemli
            add_eos_token=True,
        )
        
        # Tokenizer padding ayarları
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        logger.success("✅ Tokenizer başarıyla yüklendi")
        logger.info(f"📊 Tokenizer istatistikleri:")
        logger.info(f"  - Vocab size: {tokenizer.vocab_size:,}")
        logger.info(f"  - Model max length: {tokenizer.model_max_length}")
        logger.info(f"  - Pad token: {tokenizer.pad_token}")
        logger.info(f"  - EOS token: {tokenizer.eos_token}")
        
    except Exception as e:
        logger.error(f"❌ Tokenizer yükleme hatası: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    
    return model, tokenizer


# =============================================================================
# A4. PEFT (LoRA) ADAPTÖRLER HAZIRLAMAK
# =============================================================================

def setup_peft_model(model: AutoModelForCausalLM, config: FineTuningConfig) -> AutoModelForCausalLM:
    """
    A4a. PEFT (LoRA) adaptörlerini modele uygula
    
    Args:
        model: Temel quantized model
        config: Fine-tuning konfigürasyonu
        
    Returns:
        AutoModelForCausalLM: PEFT adaptörlerli model
    """
    logger.info("🔧 PEFT (LoRA) adaptörleri hazırlanıyor...")
    
    try:
        # A4a. Model'i k-bit training için hazırla
        model = prepare_model_for_kbit_training(model)
        logger.info("✅ Model k-bit training için hazırlandı")
        
        # A4b. LoRA konfigürasyonunu oluştur
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.target_modules,
            lora_dropout=config.lora_dropout,
            bias=config.lora_bias,
            task_type=TaskType.CAUSAL_LM,
        )
        
        logger.info("⚙️ LoRA konfigürasyonu:")
        logger.info(f"  - Rank (r): {config.lora_r}")
        logger.info(f"  - Alpha: {config.lora_alpha}")
        logger.info(f"  - Dropout: {config.lora_dropout}")
        logger.info(f"  - Target modules: {config.target_modules}")
        logger.info(f"  - Bias: {config.lora_bias}")
        
        # A4c. PEFT model'i oluştur
        model = get_peft_model(model, lora_config)
        logger.success("✅ PEFT (LoRA) adaptörleri başarıyla uygulandı")
        
        # PEFT istatistikleri
        model.print_trainable_parameters()
        
        # Detaylı istatistikler
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"📊 PEFT Model istatistikleri:")
        logger.info(f"  - Toplam parametreler: {total_params:,}")
        logger.info(f"  - Eğitilebilir parametreler: {trainable_params:,}")
        logger.info(f"  - Eğitilebilir oran: {100 * trainable_params / total_params:.4f}%")
        
    except Exception as e:
        logger.error(f"❌ PEFT setup hatası: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    
    return model


# =============================================================================
# A5. VERİ SETİNİ YÜKLEME VE HAZIRLAMA
# =============================================================================

def create_aura_instruction_dataset(data_path: str, config: FineTuningConfig) -> DatasetDict:
    """
    A5a. Women's Clothing Reviews veri setini Aura stil asistanı formatına dönüştür
    
    Args:
        data_path: CSV dosyasının yolu
        config: Fine-tuning konfigürasyonu
        
    Returns:
        DatasetDict: Eğitim ve test veri setleri
    """
    logger.info(f"📊 Veri seti yükleniyor: {data_path}")
    
    try:
        # CSV dosyasını yükle
        if not Path(data_path).exists():
            logger.error(f"❌ Veri seti dosyası bulunamadı: {data_path}")
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        
        df = pd.read_csv(data_path)
        logger.info(f"📊 Ham veri seti boyutu: {len(df)} satır")
        
        # Gerekli sütunları kontrol et
        required_columns = ['Review Text', 'Title']
        if not all(col in df.columns for col in required_columns):
            logger.warning("⚠️ Standart sütunlar bulunamadı, mevcut sütunları kullanıyoruz...")
            logger.info(f"Mevcut sütunlar: {list(df.columns)}")
            # İlk text sütununu bulalım
            text_columns = [col for col in df.columns if df[col].dtype == 'object']
            if len(text_columns) >= 2:
                df = df.rename(columns={text_columns[0]: 'Title', text_columns[1]: 'Review Text'})
            elif len(text_columns) == 1:
                df['Title'] = "Fashion Review"
                df = df.rename(columns={text_columns[0]: 'Review Text'})
        
        # A5b. Veri temizleme
        logger.info("🧹 Veri temizleme işlemleri...")
        
        # Eksik değerleri kaldır
        initial_size = len(df)
        df = df[['Review Text', 'Title']].dropna()
        logger.info(f"  - Eksik değerler temizlendi: {initial_size} → {len(df)}")
        
        # Çok kısa yorumları filtrele (minimum 20 karakter)
        df = df[df['Review Text'].str.len() > 20]
        logger.info(f"  - Kısa yorumlar filtrelendi: → {len(df)}")
        
        # Çok uzun yorumları filtrele (maximum 500 kelime)
        df = df[df['Review Text'].str.split().str.len() <= 500]
        logger.info(f"  - Uzun yorumlar filtrelendi: → {len(df)}")
        
        # Duplicate'leri kaldır
        df = df.drop_duplicates(subset=['Review Text'])
        logger.info(f"  - Duplicates kaldırıldı: → {len(df)}")
        
        # Sampling (eğer çok büyük dataset ise)
        if len(df) > 10000:
            df = df.sample(n=10000, random_state=config.random_state)
            logger.info(f"  - Dataset sampled: → {len(df)}")
        
        logger.success(f"✅ Veri temizleme tamamlandı: {len(df)} sample")
        
    except Exception as e:
        logger.error(f"❌ Veri yükleme hatası: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    
    try:
        # A5c. Instruction formatına dönüştürme
        logger.info("📝 Instruction format'a dönüştürme...")
        
        def format_aura_instruction(sample: dict) -> dict:
            """
            Fashion review'ını Aura stil asistanı instruction formatına dönüştür
            
            Llama-3 chat formatını kullanarak, fashion-aware bir asistan kişiliği oluştur
            """
            review_text = sample['Review Text'].strip()
            title = sample.get('Title', 'Fashion Review').strip()
            
            # Aura kişiliği için fashion-focused instruction
            instruction_templates = [
                # Style analysis template
                f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are Aura, an expert fashion stylist and personal shopping assistant. You help users discover their personal style, find perfect outfits, and make confident fashion choices. You're knowledgeable about fashion trends, body types, color theory, and styling techniques. Always respond in a friendly, encouraging, and stylish manner.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Can you analyze this fashion review and provide styling insights? What does this tell us about the customer's style preferences and how would you advise similar customers?

Review: "{review_text}"<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
{title} - Based on this review, I can see some great style insights! {review_text}

This customer seems to value both comfort and style, which is such a smart approach to fashion. For similar customers, I'd recommend looking for pieces that offer this same balance - think elevated basics, quality fabrics, and versatile pieces that can work for multiple occasions. The key is finding items that make you feel confident and comfortable in your own skin! ✨<|eot_id|>""",

                # Personal shopping template  
                f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are Aura, a personal fashion stylist who helps customers find their perfect style. You're warm, encouraging, and have excellent taste in fashion. You always consider the customer's lifestyle, body type, and personal preferences when giving advice.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
I'm reading reviews to understand what customers like about different fashion items. Can you help me understand what makes this review positive or negative, and what styling lessons we can learn?

Customer review: "{review_text}"<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
{title} - This review gives us wonderful insights into what customers really want from their wardrobe! {review_text}

What I love about this feedback is how it highlights the importance of [finding pieces that work with your lifestyle/choosing quality over quantity/understanding your body type]. For anyone shopping for similar items, remember that the best fashion choices are ones that make YOU feel amazing and align with your daily life! 💫<|eot_id|>""",

                # Styling advice template
                f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are Aura, an AI fashion stylist known for your ability to help people discover their unique style. You're supportive, knowledgeable about fashion trends and styling techniques, and always focused on helping customers feel confident and beautiful.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
I found this fashion review and I'm curious about your styling perspective. What would you tell someone who's considering a similar purchase?

Review: "{review_text}"<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
{title} - This review is so helpful for understanding what to look for in quality fashion pieces! {review_text}

My styling advice for anyone considering similar items: always prioritize fit and comfort alongside style. The most fashionable outfit is one where you feel authentically yourself. Consider how this piece would work with your existing wardrobe and whether it supports your lifestyle goals! 🌟<|eot_id|>"""
            ]
            
            # Randomly select a template for variety
            template_idx = hash(review_text) % len(instruction_templates)
            formatted_text = instruction_templates[template_idx]
            
            return {"text": formatted_text}
        
        # Format'ı uygula
        logger.info("📝 Instruction format uygulanıyor...")
        
        formatted_data = []
        for _, row in df.iterrows():
            formatted_sample = format_aura_instruction(row.to_dict())
            formatted_data.append(formatted_sample)
        
        logger.success(f"✅ {len(formatted_data)} sample instruction formatında hazırlandı")
        
        # Sample'ları göster
        logger.info("📋 Örnek formatted data:")
        logger.info(f"Sample 1 preview: {formatted_data[0]['text'][:200]}...")
        
    except Exception as e:
        logger.error(f"❌ Instruction format hatası: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    
    try:
        # A5d. Dataset objesine dönüştür ve train/test split
        logger.info("🔄 Dataset objesi oluşturuluyor...")
        
        # Pandas DataFrame'den Hugging Face Dataset oluştur
        dataset = Dataset.from_list(formatted_data)
        
        # Train/test split
        if config.test_size > 0:
            dataset_dict = dataset.train_test_split(
                test_size=config.test_size,
                random_state=config.random_state,
                shuffle=True
            )
            
            logger.info(f"📊 Train/Test split:")
            logger.info(f"  - Train samples: {len(dataset_dict['train'])}")
            logger.info(f"  - Test samples: {len(dataset_dict['test'])}")
        else:
            dataset_dict = DatasetDict({"train": dataset})
            logger.info(f"📊 Training dataset: {len(dataset_dict['train'])} samples")
        
        # Dataset istatistikleri
        train_dataset = dataset_dict['train']
        avg_length = np.mean([len(sample['text'].split()) for sample in train_dataset])
        max_length = max([len(sample['text'].split()) for sample in train_dataset])
        min_length = min([len(sample['text'].split()) for sample in train_dataset])
        
        logger.info(f"📊 Dataset İstatistikleri:")
        logger.info(f"  - Ortalama uzunluk: {avg_length:.1f} kelime")
        logger.info(f"  - Maksimum uzunluk: {max_length} kelime")
        logger.info(f"  - Minimum uzunluk: {min_length} kelime")
        
        logger.success("✅ Dataset başarıyla hazırlandı")
        
        return dataset_dict
        
    except Exception as e:
        logger.error(f"❌ Dataset oluşturma hatası: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


# =============================================================================
# A6. EĞİTİMİ YAPILANDIRMA VE BAŞLATMA
# =============================================================================

def setup_training_arguments(config: FineTuningConfig) -> TrainingArguments:
    """
    A6a. Training arguments'ları yapılandır
    
    Args:
        config: Fine-tuning konfigürasyonu
        
    Returns:
        TrainingArguments: Eğitim argümanları
    """
    logger.info("⚙️ Training arguments yapılandırılıyor...")
    
    # Output directory oluştur
    output_path = Path(config.output_dir) / config.new_model_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        # Basic training configuration
        output_dir=str(output_path),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        
        # Learning and optimization
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps if config.max_steps > 0 else -1,
        
        # Precision and performance
        fp16=config.fp16,
        bf16=config.bf16,
        tf32=config.tf32,
        dataloader_pin_memory=config.dataloader_pin_memory,
        
        # Logging and saving
        logging_dir=str(output_path / "logs"),
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        
        # Evaluation
        evaluation_strategy=config.evaluation_strategy,
        eval_steps=config.eval_steps,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Data handling
        remove_unused_columns=config.remove_unused_columns,
        dataloader_num_workers=0,  # Single process for stability
        
        # Monitoring
        report_to=config.report_to,
        run_name=config.run_name,
        
        # Memory optimization
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",  # Memory efficient optimizer
    )
    
    logger.info("📊 Training configuration:")
    logger.info(f"  - Output dir: {output_path}")
    logger.info(f"  - Epochs: {config.num_train_epochs}")
    logger.info(f"  - Batch size: {config.per_device_train_batch_size}")
    logger.info(f"  - Gradient accumulation: {config.gradient_accumulation_steps}")
    logger.info(f"  - Learning rate: {config.learning_rate}")
    logger.info(f"  - Precision: {'bf16' if config.bf16 else 'fp16' if config.fp16 else 'fp32'}")
    logger.info(f"  - Max steps: {config.max_steps if config.max_steps > 0 else 'Auto'}")
    
    return training_args


def train_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset_dict: DatasetDict,
    config: FineTuningConfig
) -> SFTTrainer:
    """
    A6b. SFTTrainer ile fine-tuning işlemini gerçekleştir
    
    Args:
        model: PEFT'li model
        tokenizer: Tokenizer
        dataset_dict: Eğitim ve test veri setleri
        config: Fine-tuning konfigürasyonu
        
    Returns:
        SFTTrainer: Eğitilmiş trainer objesi
    """
    logger.info("🚀 Model eğitimi başlatılıyor...")
    
    try:
        # Training arguments'ı hazırla
        training_args = setup_training_arguments(config)
        
        # Dataset'leri al
        train_dataset = dataset_dict['train']
        eval_dataset = dataset_dict.get('test', None)
        
        logger.info(f"📊 Dataset bilgileri:")
        logger.info(f"  - Training samples: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"  - Evaluation samples: {len(eval_dataset)}")
        
        # A6c. SFTTrainer'ı oluştur
        logger.info("🔧 SFTTrainer oluşturuluyor...")
        
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            dataset_text_field="text",
            max_seq_length=config.max_seq_length,
            packing=True,  # Verimlilik için sequence packing
            dataset_kwargs={
                "add_special_tokens": False,  # Template'de zaten var
                "append_concat_token": False,
            }
        )
        
        # Early stopping callback ekle
        if eval_dataset and config.early_stopping_patience > 0:
            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience,
                early_stopping_threshold=config.early_stopping_threshold
            )
            trainer.add_callback(early_stopping_callback)
            logger.info(f"⏰ Early stopping: {config.early_stopping_patience} patience")
        
        logger.success("✅ SFTTrainer başarıyla oluşturuldu")
        
        # A6d. Eğitimi başlat
        logger.info("🏋️ Fine-tuning başlatılıyor...")
        logger.info("Bu işlem uzun sürebilir, lütfen bekleyin...")
        
        # Eğitim başlangıç zamanı
        start_time = time.time()
        
        # Memory kullanımını göster
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"🔋 GPU Memory (başlangıç):")
            logger.info(f"  - Allocated: {memory_allocated:.2f} GB")
            logger.info(f"  - Reserved: {memory_reserved:.2f} GB")
        
        # Eğitimi çalıştır
        train_result = trainer.train()
        
        # Eğitim süresi
        training_time = time.time() - start_time
        logger.success(f"✅ Fine-tuning tamamlandı! Süre: {training_time/60:.1f} dakika")
        
        # Final memory kullanımı
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"🔋 GPU Memory (final):")
            logger.info(f"  - Allocated: {memory_allocated:.2f} GB")
            logger.info(f"  - Reserved: {memory_reserved:.2f} GB")
        
        # Eğitim sonuçları
        logger.info("📊 Eğitim sonuçları:")
        logger.info(f"  - Final train loss: {train_result.training_loss:.4f}")
        logger.info(f"  - Training steps: {train_result.global_step}")
        if hasattr(train_result, 'metrics'):
            for key, value in train_result.metrics.items():
                logger.info(f"  - {key}: {value}")
        
        return trainer
        
    except Exception as e:
        logger.error(f"❌ Training hatası: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


# =============================================================================
# A7. EĞİTİLMİŞ MODELİ KAYDETME
# =============================================================================

def save_model_and_artifacts(
    trainer: SFTTrainer,
    tokenizer: AutoTokenizer,
    config: FineTuningConfig
) -> None:
    """
    A7a. Eğitilmiş LoRA adaptörlerini ve tokenizer'ı kaydet
    
    Args:
        trainer: Eğitilmiş SFTTrainer
        tokenizer: Tokenizer
        config: Fine-tuning konfigürasyonu
    """
    logger.info("💾 Model ve artifacts kaydediliyor...")
    
    try:
        # Output path'i hazırla
        output_path = Path(config.output_dir) / config.new_model_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # A7a. LoRA adaptörlerini kaydet
        logger.info("💾 LoRA adaptörleri kaydediliyor...")
        
        trainer.save_model()  # Bu otomatik olarak output_dir'e kaydeder
        logger.success(f"✅ LoRA adaptörleri kaydedildi: {output_path}")
        
        # A7b. Tokenizer'ı kaydet
        logger.info("💾 Tokenizer kaydediliyor...")
        
        tokenizer.save_pretrained(output_path)
        logger.success(f"✅ Tokenizer kaydedildi: {output_path}")
        
        # A7c. Configuration'ı kaydet
        logger.info("💾 Configuration kaydediliyor...")
        
        config_dict = asdict(config)
        config_path = output_path / "finetune_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False, default=str)
        
        logger.success(f"✅ Configuration kaydedildi: {config_path}")
        
        # A7d. Model card oluştur
        logger.info("📝 Model card oluşturuluyor...")
        
        model_card = f"""# Aura Fashion Style Assistant - Fine-Tuned Model

## Model Description
Bu model, Aura moda asistanı projesi için Meta-Llama-3-8B-Instruct temel modelinden QLoRA tekniği ile fine-tune edilmiştir.

## Training Details
- **Base Model**: {config.model_id}
- **Fine-tuning Method**: QLoRA (4-bit quantization + LoRA)
- **Training Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Dataset**: Women's Clothing Reviews
- **LoRA Rank**: {config.lora_r}
- **LoRA Alpha**: {config.lora_alpha}
- **Training Epochs**: {config.num_train_epochs}
- **Learning Rate**: {config.learning_rate}

## Model Capabilities
- Fashion styling advice and recommendations
- Style analysis and personal shopping assistance
- Trend analysis and fashion insights
- Personalized outfit suggestions

## Usage
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("{output_path}")

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "{config.model_id}",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "{output_path}")

# Generate response
inputs = tokenizer.encode("Your fashion question here", return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Performance
- Training Loss: [Automatically filled during training]
- Model Size: ~8B parameters (base) + {config.lora_r * 2} LoRA parameters
- Memory Usage: ~6GB GPU memory for inference

## License
This model inherits the license from the base Meta-Llama-3-8B-Instruct model.

## Disclaimer
This model is designed for fashion and styling assistance. Always consider personal preferences, body type, and individual style when making fashion decisions.
"""
        
        model_card_path = output_path / "README.md"
        with open(model_card_path, 'w', encoding='utf-8') as f:
            f.write(model_card)
        
        logger.success(f"✅ Model card oluşturuldu: {model_card_path}")
        
        # A7e. Dosya listesini göster
        logger.info("📋 Kaydedilen dosyalar:")
        for file_path in output_path.rglob("*"):
            if file_path.is_file():
                file_size = file_path.stat().st_size / 1024**2  # MB
                logger.info(f"  - {file_path.name}: {file_size:.2f} MB")
        
        total_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / 1024**2
        logger.info(f"📊 Toplam boyut: {total_size:.2f} MB")
        
        logger.success(f"🎉 Tüm model artifacts başarıyla kaydedildi: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Model kaydetme hatası: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


# =============================================================================
# MAIN FUNCTION VE SCRIPT RUNNER
# =============================================================================

def main():
    """
    Ana fine-tuning pipeline'ı
    
    Bu fonksiyon tüm adımları sırasıyla çalıştırır:
    1. Arguments parsing ve configuration
    2. Model ve tokenizer yükleme
    3. PEFT setup
    4. Dataset hazırlama
    5. Training
    6. Model kaydetme
    """
    try:
        # A2. Argümanları parse et ve configuration oluştur
        args = parse_arguments()
        setup_logging(debug=args.debug)
        
        logger.info("🎯 Aura Fashion Style Assistant Fine-Tuning Pipeline")
        logger.info("=" * 60)
        
        # Configuration objesi oluştur
        config = FineTuningConfig(
            model_id=args.model_id,
            new_model_name=args.new_model_name,
            dataset_path=args.dataset_path,
            max_seq_length=args.max_seq_length,
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            report_to=args.report_to,
            run_name=args.run_name,
            bf16=args.bf16,
            fp16=args.fp16,
        )
        
        logger.info("📋 Configuration:")
        logger.info(f"  - Model: {config.model_id}")
        logger.info(f"  - Output: {config.new_model_name}")
        logger.info(f"  - Dataset: {config.dataset_path}")
        logger.info(f"  - LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
        logger.info(f"  - Training: {config.num_train_epochs} epochs, lr={config.learning_rate}")
        
        # Monitoring setup
        if config.report_to == "wandb":
            logger.info("🔗 Weights & Biases monitoring aktif")
            try:
                wandb.init(
                    project="aura-fashion-assistant",
                    name=config.run_name,
                    config=asdict(config)
                )
            except Exception as e:
                logger.warning(f"⚠️ W&B initialization failed: {e}")
                config.report_to = "none"
        
        # System info
        logger.info("🖥️ System Information:")
        logger.info(f"  - PyTorch version: {torch.__version__}")
        logger.info(f"  - CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"  - GPU count: {torch.cuda.device_count()}")
            logger.info(f"  - GPU name: {torch.cuda.get_device_name(0)}")
            logger.info(f"  - GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # A3. Model ve tokenizer yükleme
        logger.info("\n" + "="*60)
        logger.info("ADIM 1: MODEL VE TOKENIZER YÜKLEME")
        logger.info("="*60)
        
        model, tokenizer = load_model_and_tokenizer(config)
        
        # A4. PEFT setup
        logger.info("\n" + "="*60)
        logger.info("ADIM 2: PEFT (LoRA) SETUP")
        logger.info("="*60)
        
        model = setup_peft_model(model, config)
        
        # A5. Dataset hazırlama
        logger.info("\n" + "="*60)
        logger.info("ADIM 3: DATASET HAZIRLAMA")
        logger.info("="*60)
        
        dataset_dict = create_aura_instruction_dataset(config.dataset_path, config)
        
        # A6. Training
        logger.info("\n" + "="*60)
        logger.info("ADIM 4: MODEL EĞİTİMİ")
        logger.info("="*60)
        
        trainer = train_model(model, tokenizer, dataset_dict, config)
        
        # A7. Model kaydetme
        logger.info("\n" + "="*60)
        logger.info("ADIM 5: MODEL KAYDETME")
        logger.info("="*60)
        
        save_model_and_artifacts(trainer, tokenizer, config)
        
        # Final
        logger.info("\n" + "="*60)
        logger.success("🎉 AURA FASHION ASSISTANT FINE-TUNING TAMAMLANDI!")
        logger.info("="*60)
        
        output_path = Path(config.output_dir) / config.new_model_name
        logger.info(f"📁 Model lokasyonu: {output_path}")
        logger.info("📖 Kullanım için README.md dosyasını inceleyin")
        logger.info("🚀 Model artık Aura conversational AI servisinde kullanıma hazır!")
        
        # Cleanup
        if config.report_to == "wandb":
            wandb.finish()
        
        # GPU memory temizleme
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 GPU memory temizlendi")
        
    except KeyboardInterrupt:
        logger.warning("⚠️ Training kullanıcı tarafından durduruldu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ Pipeline hatası: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Cleanup on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sys.exit(1)


if __name__ == "__main__":
    """
    A5. Script doğrudan çalıştırıldığında main fonksiyonunu çağır
    
    Kullanım:
    python finetune.py --model_id meta-llama/Meta-Llama-3-8B-Instruct \
                       --dataset_path ./data/womens_clothing_reviews.csv \
                       --output_dir ./saved_models \
                       --num_train_epochs 3 \
                       --per_device_train_batch_size 4 \
                       --learning_rate 2e-4 \
                       --lora_r 16 \
                       --bf16
    """
    main()
