# MLOps Infrastructure Documentation

Bu dokümantasyon, Aura AI Platform için geliştirilmiş kapsamlı MLOps altyapısını açıklamaktadır. Bu altyapı GitHub Actions ile CI/CD pipeline'ları, DVC ile model sürüm yönetimi ve özelleştirilmiş experiment tracking sistemlerini içermektedir.

## 🏗️ Altyapı Genel Bakış

### Mimari Bileşenler

1. **CI/CD Pipeline** - GitHub Actions tabanlı sürekli entegrasyon ve dağıtım
2. **Model Versioning** - DVC (Data Version Control) ile veri ve model sürüm yönetimi
3. **Experiment Tracking** - Her servis için özelleştirilmiş deneyim takip sistemi
4. **Container Orchestration** - Docker ve Kubernetes ile konteynerleştirme
5. **Cloud Integration** - Google Cloud Platform (GKE, GCR) entegrasyonu

### Desteklenen Servisler

- **Visual Analysis Service** - Görsel analiz ve nesne tespiti
- **Outfit Recommendation Service** - Kıyafet önerisi ve kombinasyon
- **Conversational AI Service** - QLoRA fine-tuning ve RAG sistemi

---

## 🔄 CI/CD Pipeline Kullanımı

### GitHub Actions Workflows

#### 1. Code Quality & Testing (`code-quality-test.yml`)

**Tetikleme Koşulları:**
- Pull request oluşturulduğunda
- Main branch'e push yapıldığında
- Manuel tetikleme (`workflow_dispatch`)

**Özellikler:**
```yaml
# Paralel job execution
jobs:
  quality-check:      # Code quality checks
  visual-analysis:    # Visual analysis service tests
  outfit-rec:         # Outfit recommendation tests
  conversational-ai:  # Conversational AI tests
  integration:        # Integration tests
  security:          # Security scanning
```

**Kullanım:**
```bash
# Manuel tetikleme
gh workflow run code-quality-test.yml

# Specific service test
gh workflow run code-quality-test.yml -f test_service=visual_analysis
```

#### 2. Build & Push (`build-and-push.yml`)

**Tetikleme Koşulları:**
- Main branch'e merge edildiğinde
- Release tag'i oluşturulduğunda

**Özellikler:**
- Değişiklik tespiti (sadece değişen servisler build edilir)
- Multi-stage Docker build
- Güvenlik taraması
- Otomatik tag oluşturma

**Kullanım:**
```bash
# Tag oluşturarak release tetikleme
git tag v1.0.0
git push origin v1.0.0

# Manuel tetikleme
gh workflow run build-and-push.yml -f force_build_all=true
```

#### 3. Deployment (`deploy-to-gke.yml`)

**Tetikleme Koşulları:**
- Build başarılı olduğunda
- Manuel deployment tetikleme

**Özellikler:**
- Staging ve production environment'ları
- Rolling update stratejisi
- Health check ve rollback mekanizması
- Post-deployment verification

**Kullanım:**
```bash
# Staging deployment
gh workflow run deploy-to-gke.yml -f environment=staging

# Production deployment
gh workflow run deploy-to-gke.yml -f environment=production
```

### Secrets Konfigürasyonu

Repository Settings > Secrets > Actions'da aşağıdaki secret'ları tanımlayın:

```yaml
GCP_PROJECT_ID: "your-gcp-project"
GCP_SA_KEY: "base64-encoded-service-account-key"
GKE_CLUSTER_NAME: "aura-cluster"
GKE_ZONE: "us-central1-a"
DOCKER_REGISTRY: "gcr.io/your-project"
```

---

## 📊 DVC Model Versioning

### Pipeline Yapısı

DVC pipeline'ı, her servis için ayrı eğitim aşamalarını tanımlar:

```yaml
stages:
  # Visual Analysis Pipeline
  visual_data_prep:
    cmd: python services/visual_analysis/scripts/prepare_data.py
    deps: [data/raw/visual, services/visual_analysis/scripts/prepare_data.py]
    outs: [data/processed/visual]
    
  visual_train:
    cmd: python services/visual_analysis/scripts/train.py
    deps: [data/processed/visual, services/visual_analysis/src]
    outs: [models/visual_analysis]
    metrics: [metrics/visual_analysis.json]
```

### Kullanım Örnekleri

#### Pipeline Çalıştırma

```bash
# Tüm pipeline'ı çalıştır
dvc repro

# Belirli bir stage'i çalıştır
dvc repro visual_train

# Belirli bir servisi çalıştır
dvc repro visual_analysis_pipeline
```

#### Model Versioning

```bash
# Model değişikliklerini commit et
dvc add models/
git add models.dvc params.yaml dvc.lock
git commit -m "Update models v1.2.0"
git tag v1.2.0

# Model'i uzak storage'a push et
dvc push

# Belirli bir versiyonu checkout et
git checkout v1.1.0
dvc checkout
```

#### Deneyim Karşılaştırma

```bash
# Metric'leri karşılaştır
dvc metrics diff

# Belirli commit'ler arası karşılaştır
dvc metrics diff HEAD~1

# Parametre değişikliklerini göster
dvc params diff
```

### Remote Storage Konfigürasyonu

```bash
# Google Cloud Storage remote ekle
dvc remote add -d gcs gs://aura-dvc-storage

# AWS S3 remote ekle
dvc remote add -d s3 s3://aura-dvc-bucket/data

# Remote credentials set et
dvc remote modify gcs credentialpath service-account.json
```

---

## 🧪 Experiment Tracking Sistemi

### Visual Analysis Experiment Tracker

DETR tabanlı görsel analiz modelleri için özelleştirilmiş tracking:

```python
from services.visual_analysis.src.experiment_tracker import VisualExperimentTracker

# Tracker oluştur
tracker = VisualExperimentTracker("detr_object_detection")

# Model konfigürasyonu log et
tracker.log_model_config({
    "model_name": "facebook/detr-resnet-50",
    "num_classes": 92,
    "hidden_dim": 256,
    "nheads": 8,
    "num_encoder_layers": 6
})

# Training loop
for epoch in range(epochs):
    # Training step
    loss = train_step()
    tracker.log_training_step(epoch, loss, learning_rate)
    
    # Validation
    if epoch % eval_frequency == 0:
        metrics = validate_model()
        tracker.log_detection_metrics(metrics)

# Finish experiment
tracker.finish("completed")
```

**Özel Metric'ler:**
- Object detection: mAP@0.5, mAP@0.75, precision, recall
- Fashion-specific: style_accuracy, color_detection, trend_score
- Performance: inference_time, memory_usage, gpu_utilization

### Outfit Recommendation Tracker

Kombinasyon önerisi için özelleştirilmiş tracking:

```python
from services.outfit_recommendation.src.experiment_tracker import OutfitExperimentTracker

tracker = OutfitExperimentTracker("transformer_recommendation")

# Recommendation metrics
tracker.log_recommendation_metrics({
    "ndcg_at_5": 0.78,
    "hit_rate_at_10": 0.85,
    "diversity_score": 0.72,
    "style_consistency": 0.89,
    "color_harmony": 0.81
})

# Compatibility matrix analysis
tracker.log_compatibility_analysis({
    "category_coverage": 0.95,
    "seasonal_appropriateness": 0.88,
    "occasion_matching": 0.82
})
```

### Conversational AI Tracker

QLoRA fine-tuning ve RAG sistemi için özelleştirilmiş tracking:

```python
from services.conversational_ai.src.experiment_tracker import ConversationalExperimentTracker

tracker = ConversationalExperimentTracker("qlora_llama2_fashion")

# QLoRA configuration
tracker.log_qlora_config({
    "base_model_name": "meta-llama/Llama-2-7b-chat-hf",
    "lora_r": 64,
    "lora_alpha": 16,
    "load_in_4bit": True
})

# RAG configuration
tracker.log_rag_config({
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "retrieval_k": 5,
    "chunk_size": 1000
})

# Generation metrics
tracker.log_generation_metrics({
    "perplexity": 15.2,
    "bleu_score": 0.45,
    "rouge_l": 0.38,
    "response_time_ms": 250
})

# RAG metrics
tracker.log_rag_metrics({
    "retrieval_precision": 0.75,
    "context_relevance": 0.82,
    "answer_relevance": 0.78
})
```

---

## 🐳 Container & Kubernetes

### Docker Build Stratejisi

Her servis için multi-stage build:

```dockerfile
# Base stage
FROM python:3.9-slim as base
COPY requirements.txt .
RUN pip install -r requirements.txt

# Development stage
FROM base as development
COPY . .
CMD ["python", "app.py"]

# Production stage
FROM base as production
COPY --from=build /app/dist /app
CMD ["gunicorn", "app:app"]
```

### Kubernetes Deployment

```yaml
# Service deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: visual-analysis-service
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 1
  template:
    spec:
      containers:
      - name: visual-analysis
        image: gcr.io/aura-project/visual-analysis:latest
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

---

## 📈 Monitoring & Alerting

### Metrics Collection

Her experiment tracker otomatik olarak şu metric'leri toplar:

1. **Training Metrics**: Loss, accuracy, learning rate
2. **Performance Metrics**: Inference time, memory usage
3. **Business Metrics**: User satisfaction, conversion rates
4. **System Metrics**: GPU utilization, throughput

### Dashboard Konfigürasyonu

```python
# Experiment comparison
def compare_experiments(experiment_ids):
    results = {}
    for exp_id in experiment_ids:
        tracker = load_experiment(exp_id)
        results[exp_id] = tracker.get_best_metrics()
    
    return create_comparison_dashboard(results)
```

### Alert Konfigürasyonu

```yaml
# GitHub Actions alert example
- name: Alert on deployment failure
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    text: "Deployment failed for ${{ github.repository }}"
```

---

## 🔧 Troubleshooting

### Yaygın Sorunlar ve Çözümler

#### 1. DVC Pipeline Hatası

```bash
# Cache temizle
dvc cache dir
rm -rf .dvc/cache

# Pipeline'ı force ile çalıştır
dvc repro --force
```

#### 2. Docker Build Hatası

```bash
# Cache'siz build
docker build --no-cache -t service:latest .

# Multi-platform build
docker buildx build --platform linux/amd64,linux/arm64 .
```

#### 3. Kubernetes Deployment Hatası

```bash
# Pod loglarını kontrol et
kubectl logs -f deployment/visual-analysis-service

# Resource kullanımını kontrol et
kubectl top pods

# Service health check
kubectl get pods -l app=visual-analysis
```

#### 4. Experiment Tracker Hatası

```bash
# Experiment dizinini temizle
rm -rf experiments/corrupted_experiment

# Tracker'ı debug mode'da çalıştır
export EXPERIMENT_DEBUG=true
python train.py
```

---

## 📝 En İyi Uygulamalar

### 1. Kod Kalitesi

- Her commit'te otomatik format checking (black, flake8)
- Type hints kullanımı (mypy checking)
- Comprehensive test coverage (>80%)
- Security scanning (bandit, safety)

### 2. Model Versioning

- Her model değişikliğinde DVC ile versioning
- Meaningful commit messages ve tags
- Model performance regression testing
- Artifact storage optimization

### 3. Experiment Management

- Descriptive experiment names
- Comprehensive parameter logging
- Regular metric comparison
- Artifact cleanup policies

### 4. Deployment

- Rolling update stratejisi
- Health check implementation
- Resource limit tanımlama
- Monitoring ve alerting

### 5. Security

- Secret management (GitHub Secrets)
- Container vulnerability scanning
- Network policy implementation
- Access control (RBAC)

---

## 🚀 Gelecek Geliştirmeler

### Roadmap

1. **Q1 2024**
   - A/B testing framework entegrasyonu
   - Advanced monitoring dashboard
   - Auto-scaling implementation

2. **Q2 2024**
   - Multi-cloud deployment support
   - ML feature store integration
   - Real-time model updating

3. **Q3 2024**
   - Federated learning support
   - Edge deployment capabilities
   - Advanced anomaly detection

### Planlanan Özellikler

- **AutoML Integration**: Otomatik hyperparameter tuning
- **Model Explainability**: SHAP ve LIME entegrasyonu
- **Data Drift Detection**: Continuous monitoring
- **Cost Optimization**: Resource usage optimization

---

## 📞 Destek ve İletişim

### Teknik Destek

- **Issues**: GitHub Issues üzerinden
- **Documentation**: Wiki sayfalarında
- **Community**: Discussions sekmesinde

### Katkıda Bulunma

1. Fork the repository
2. Create feature branch
3. Implement changes with tests
4. Submit pull request
5. Code review process

---

Bu MLOps altyapısı, Aura AI Platform'un sürdürülebilir ve ölçeklenebilir gelişimi için tasarlanmıştır. Herhangi bir sorun ya da öneriniz için GitHub Issues'ı kullanabilirsiniz.
