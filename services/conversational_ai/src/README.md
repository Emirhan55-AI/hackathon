# Aura Fashion Assistant - QLoRA Fine-Tuning Module

Bu modül, Aura projesinin sohbet asistanı için Meta-Llama-3-8B-Instruct modelini QLoRA (Quantized LoRA) tekniği ile moda alanına özel olarak ince ayar yapmak için geliştirilmiştir.

## 📋 İçindekiler

- [Özellikler](#özellikler)
- [Kurulum](#kurulum)
- [Veri Seti Hazırlığı](#veri-seti-hazırlığı)
- [Kullanım](#kullanım)
- [Konfigürasyon](#konfigürasyon)
- [Sistem Gereksinimleri](#sistem-gereksinimleri)
- [Teknik Detaylar](#teknik-detaylar)
- [Troubleshooting](#troubleshooting)

## ✨ Özellikler

- **QLoRA Fine-Tuning**: 4-bit quantization ile bellek verimli eğitim
- **Fashion-Specific Personality**: Moda alanına özel kişilik geliştirme
- **Memory Optimized**: Tek GPU'da (16GB+) çalışacak şekilde optimize edildi
- **Production Ready**: FastAPI servisinde kullanıma hazır
- **Comprehensive Logging**: Detaylı loglama ve monitoring desteği
- **Flexible Configuration**: Farklı donanım için optimize edilmiş konfigürasyonlar

## 🚀 Kurulum

### 1. Gereksinimler

```bash
# Python 3.8+ gerekli
python --version

# CUDA destekli PyTorch kurulumu
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Proje bağımlılıklarını yükle
cd services/conversational_ai
pip install -r requirements.txt
```

### 2. Model İndirme İzinleri

Meta-Llama modelleri için Hugging Face Hub'da giriş yapmanız gerekiyor:

```bash
# Hugging Face CLI ile giriş
huggingface-cli login

# Meta-Llama-3-8B-Instruct için izin isteği:
# https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
```

## 📊 Veri Seti Hazırlığı

### 1. Women's Clothing Reviews Dataset

Kaggle'dan [Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) veri setini indirin:

```bash
# Veri seti klasörüne yerleştirin
mkdir -p data
# womens_clothing_reviews.csv dosyasını data/ klasörüne kopyalayın
```

### 2. Veri Seti Formatı

CSV dosyasında şu sütunlar olmalı:
- `Review Text`: Müşteri yorumları
- `Title`: Yorum başlıkları (opsiyonel)

### 3. Özel Veri Seti

Kendi veri setinizi kullanmak için aynı format ile CSV oluşturun:

```csv
Review Text,Title
"Bu elbise harika! Çok rahat ve şık görünüyor.","Muhteşem Elbise"
"Ayakkabı tam beden geldi, kalitesi çok iyi.","Kaliteli Ayakkabı"
```

## 🎯 Kullanım

### 1. Temel Kullanım

```bash
# Basit fine-tuning
python src/finetune.py \
    --dataset_path ./data/womens_clothing_reviews.csv \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-4 \
    --bf16
```

### 2. Bellek Optimizasyonlu (Küçük GPU'lar için)

```bash
# 8-12GB GPU için
python src/finetune.py \
    --dataset_path ./data/womens_clothing_reviews.csv \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_seq_length 512 \
    --lora_r 8 \
    --lora_alpha 16 \
    --bf16
```

### 3. Hızlı Prototipleme

```bash
# Hızlı test için
python src/finetune.py \
    --dataset_path ./data/womens_clothing_reviews.csv \
    --max_steps 200 \
    --per_device_train_batch_size 4 \
    --max_seq_length 512 \
    --learning_rate 5e-4
```

### 4. Production Kalitesi

```bash
# En iyi kalite için
python src/finetune.py \
    --dataset_path ./data/womens_clothing_reviews.csv \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --lora_r 32 \
    --lora_alpha 64 \
    --learning_rate 1e-4 \
    --warmup_steps 100 \
    --report_to wandb \
    --bf16
```

## ⚙️ Konfigürasyon

### Komut Satırı Argümanları

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| `--model_id` | `meta-llama/Meta-Llama-3-8B-Instruct` | Temel model |
| `--dataset_path` | `./data/womens_clothing_reviews.csv` | Veri seti yolu |
| `--output_dir` | `./saved_models` | Çıktı dizini |
| `--num_train_epochs` | `3` | Eğitim epoch sayısı |
| `--per_device_train_batch_size` | `4` | Batch size |
| `--gradient_accumulation_steps` | `4` | Gradient accumulation |
| `--learning_rate` | `2e-4` | Öğrenme oranı |
| `--max_seq_length` | `1024` | Maksimum sequence uzunluğu |
| `--lora_r` | `16` | LoRA rank boyutu |
| `--lora_alpha` | `32` | LoRA alpha parametresi |
| `--lora_dropout` | `0.05` | LoRA dropout oranı |
| `--bf16` | `True` | BFloat16 precision |
| `--report_to` | `none` | Monitoring (`wandb`, `tensorboard`) |

### Örnek Konfigürasyonlar

`src/config_examples.py` dosyasında farklı kullanım senaryoları için hazır konfigürasyonlar bulabilirsiniz.

## 🖥️ Sistem Gereksinimleri

### Minimum Gereksinimler

- **GPU**: 16GB+ VRAM (RTX 4090, A100, vb.)
- **RAM**: 32GB sistem belleği
- **Storage**: 50GB+ boş alan
- **CUDA**: 11.8 veya üzeri

### Önerilen Gereksinimler

- **GPU**: 24GB+ VRAM (RTX 4090, A6000, A100)
- **RAM**: 64GB sistem belleği
- **Storage**: 100GB+ SSD
- **CUDA**: 12.0+

### Küçük GPU'lar için

8-12GB GPU'lar için memory optimized konfigürasyon kullanın:
- Batch size: 1-2
- Gradient accumulation: 8-16
- Max sequence length: 512
- LoRA rank: 8

## 🔧 Teknik Detaylar

### QLoRA Optimizasyonları

- **4-bit Quantization**: NF4 formatı ile %75 bellek tasarrufu
- **Double Quantization**: Ek bellek optimizasyonu
- **Paged AdamW**: Bellek verimli optimizer
- **Gradient Checkpointing**: Backward pass bellek optimizasyonu

### LoRA Parametreleri

- **Rank (r)**: 8-32 arası (kalite/hız dengesi)
- **Alpha**: Genellikle rank'ın 2 katı
- **Target Modules**: Attention ve MLP katmanları
- **Dropout**: Overfitting'i önlemek için 0.05-0.1

### Dataset İşleme

- **Instruction Formatting**: Llama-3 chat formatı
- **Fashion-Specific Prompts**: Moda odaklı talimat şablonları
- **Sequence Packing**: Verimlilik için batch optimizasyonu
- **Train/Test Split**: %90/%10 oranında

## 📊 Monitoring ve Logging

### Weights & Biases

```bash
# W&B ile monitoring
pip install wandb
wandb login

python src/finetune.py \
    --report_to wandb \
    --run_name "aura-fashion-v1"
```

### Log Dosyaları

- Training sırasında otomatik log dosyası oluşturulur
- Console'da rengli progress gösterimi
- GPU memory kullanımı tracking

## 🚨 Troubleshooting

### Out of Memory Hatası

```bash
# Batch size'ı azaltın
--per_device_train_batch_size 1
--gradient_accumulation_steps 16

# Sequence length'i azaltın
--max_seq_length 512

# LoRA rank'ı azaltın
--lora_r 8 --lora_alpha 16
```

### Model Yükleme Hatası

```bash
# Hugging Face login kontrol
huggingface-cli whoami

# Cache temizle
rm -rf ~/.cache/huggingface/

# Model izin sayfasını ziyaret edin
# https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
```

### CUDA Hatası

```bash
# CUDA sürümünü kontrol edin
nvidia-smi

# PyTorch CUDA uyumluluğu
python -c "import torch; print(torch.cuda.is_available())"

# Uyumlu PyTorch sürümü yükleyin
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 📁 Çıktı Dosyaları

Eğitim tamamlandıktan sonra `saved_models/` dizininde:

```
saved_models/
└── llama3-8b-aura-fashion-qlora/
    ├── adapter_config.json       # LoRA konfigürasyonu
    ├── adapter_model.safetensors # LoRA ağırlıkları
    ├── tokenizer.json           # Tokenizer dosyaları
    ├── tokenizer_config.json    
    ├── special_tokens_map.json  
    ├── finetune_config.json     # Eğitim konfigürasyonu
    ├── README.md                # Model kartı
    └── logs/                    # Training logları
```

## 🔗 Entegrasyon

Eğitilmiş model Aura conversational AI servisinde kullanım için hazırdır:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Model yükleme
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

model = PeftModel.from_pretrained(
    base_model, 
    "./saved_models/llama3-8b-aura-fashion-qlora"
)

tokenizer = AutoTokenizer.from_pretrained(
    "./saved_models/llama3-8b-aura-fashion-qlora"
)
```

## 📞 Destek

Herhangi bir sorun yaşadığınızda:

1. Bu README'nin troubleshooting bölümünü kontrol edin
2. Log dosyalarını inceleyin
3. System requirements'ları doğrulayın
4. GitHub issues'da benzer problemleri arayın

---

**Aura AI Team** | 2025 | Version 1.0.0
