# Aura Visual Analysis API

🔍 **DETR tabanlı Fashion Analysis Mikroservisi**

Bu API, Aura yapay zeka platformunun görsel analiz mikroservisini FastAPI framework'ü ile HTTP API olarak sunar. DETR (Detection Transformer) mimarisi kullanılarak Fashionpedia veri seti üzerinde eğitilmiş model ile fashion item detection ve analysis gerçekleştirir.

## 🚀 Özellikler

- **Fashion Item Detection**: 294 Fashionpedia kategorisinde giyim eşyası ve aksesuar tespiti
- **Attribute Analysis**: Renk, desen, stil ve malzeme analizi
- **Segmentation**: Piksel seviyesinde segmentasyon maskeleri
- **Batch Processing**: Çoklu görüntü analizi desteği
- **REST API**: RESTful HTTP endpoints
- **Interactive Docs**: Swagger/OpenAPI dokümantasyonu
- **Error Handling**: Kapsamlı hata işleme ve logging
- **Performance**: Async FastAPI ile yüksek performans

## 📋 Gereksinimler

- Python 3.8+
- PyTorch 1.9+
- Hugging Face Transformers 4.35+
- FastAPI 0.104+
- GPU önerilir (CPU'da da çalışır)

Tam gereksinimler için: `../requirements.txt`

## 🛠️ Kurulum

### 1. Bağımlılıkları Yükle

```bash
# Ana dizinde
pip install -r requirements.txt
```

### 2. Model Hazırlığı

API, iki şekilde çalışabilir:

**Option A: Eğitilmiş Model**
```bash
# Eğitilmiş modeli saved_models/ dizinine koy
export MODEL_PATH="./saved_models/detr_fashionpedia.pth"
```

**Option B: Pre-trained Model**
```bash
# Otomatik olarak Hugging Face'den indirilir
# MODEL_PATH ayarlanmazsa varsayılan olarak bu kullanılır
```

### 3. Environment Variables (Opsiyonel)

```bash
export HOST="0.0.0.0"           # API host (varsayılan: 0.0.0.0)
export PORT="8000"              # API port (varsayılan: 8000)
export DEBUG="false"            # Debug mode (varsayılan: false)
export MODEL_PATH="..."         # Model dosyası yolu
export WORKERS="1"              # Production worker sayısı
```

## 🏃‍♂️ Çalıştırma

### Development Mode

```bash
# Basit çalıştırma
python main.py

# Veya startup script ile
python startup.py dev
```

### Production Mode

```bash
# Tek worker
uvicorn main:app --host 0.0.0.0 --port 8000

# Çoklu worker
python startup.py prod --workers 4

# Veya direkt uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker (Gelecek)

```bash
# TODO: Dockerfile oluşturulacak
docker build -t aura-visual-analysis .
docker run -p 8000:8000 aura-visual-analysis
```

## 📡 API Endpoints

### Core Endpoints

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| `GET` | `/` | Ana sayfa - API bilgileri |
| `GET` | `/health` | Sağlık kontrolü |
| `GET` | `/docs` | Interactive API dokümantasyonu |

### Analysis Endpoints

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| `POST` | `/analyze` | Tek görüntü analizi |
| `POST` | `/analyze/batch` | Çoklu görüntü analizi |

### Information Endpoints

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| `GET` | `/model/info` | Model detayları |
| `GET` | `/categories` | Fashion kategorileri |
| `GET` | `/stats` | API istatistikleri |

## 🔍 Kullanım Örnekleri

### 1. Tek Görüntü Analizi

```python
import requests

# Görüntü dosyası
files = {"file": open("fashion_image.jpg", "rb")}

# Analiz parametreleri
data = {
    "confidence_threshold": 0.7,
    "max_detections": 50,
    "return_masks": True,
    "include_attributes": True
}

# API çağrısı
response = requests.post(
    "http://localhost:8000/analyze",
    files=files,
    data=data
)

result = response.json()
print(f"Bulunan item sayısı: {len(result['detections'])}")
```

### 2. JavaScript/Fetch ile

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('confidence_threshold', '0.7');

fetch('http://localhost:8000/analyze', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Analysis results:', data);
});
```

### 3. cURL ile

```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@fashion_image.jpg" \
     -F "confidence_threshold=0.7" \
     -F "max_detections=50"
```

## 📊 Response Format

### Başarılı Analiz Yanıtı

```json
{
  "success": true,
  "detections": [
    {
      "label": "shirt, blouse",
      "confidence": 0.89,
      "bbox": [100.5, 150.2, 200.8, 300.1],
      "area": 60240.8,
      "category_id": 1,
      "attributes": {
        "colors": ["blue", "white"],
        "patterns": ["striped"],
        "styles": ["casual"],
        "materials": ["cotton"],
        "dominant_color": "blue"
      },
      "mask": [[...], [...]]
    }
  ],
  "summary": {
    "total_detections": 1,
    "unique_categories": 1,
    "average_confidence": 0.89,
    "image_dimensions": [800, 600],
    "categories_found": ["shirt, blouse"]
  },
  "metadata": {
    "processing_time": 2.34,
    "image_filename": "fashion_image.jpg",
    "parameters_used": {...},
    "timestamp": "2025-08-03 10:30:45"
  },
  "model_info": {...}
}
```

### Hata Yanıtı

```json
{
  "error": "Unsupported file format",
  "details": "Only JPG, PNG, BMP, TIFF, WEBP formats are supported",
  "timestamp": "2025-08-03 10:30:45"
}
```

## 🎯 Desteklenen Fashion Kategorileri

API, **294 Fashionpedia kategorisini** destekler:

### Ana Kategoriler
- **Üst Giyim**: shirt, blouse, top, t-shirt, sweater, cardigan, jacket, vest
- **Alt Giyim**: pants, shorts, skirt, dress, jumpsuit
- **Dış Giyim**: coat, cape, hood
- **Aksesuarlar**: glasses, hat, tie, glove, watch, belt, bag, scarf, umbrella
- **Ayakkabı ve Çorap**: shoe, sock, tights, stockings, leg warmer
- **Detaylar**: collar, lapel, sleeve, pocket, neckline, buckle, zipper
- **Süslemeler**: applique, bead, bow, flower, fringe, ribbon, rivet, ruffle, sequin, tassel

### Fashion Attributes
- **Renkler**: red, blue, green, yellow, black, white, gray, brown, pink, purple, orange, beige, navy, maroon, olive
- **Desenler**: solid, striped, polka-dot, floral, geometric, abstract, plaid, checkered, leopard, zebra, paisley
- **Stiller**: casual, formal, business, sporty, vintage, modern, bohemian, classic, trendy, elegant, edgy
- **Malzemeler**: cotton, silk, wool, leather, denim, linen, polyester, chiffon, velvet, satin, knit, synthetic

## 🧪 Test Etme

### Test Script ile

```bash
# Otomatik test suite
python test_api.py

# Veya startup script ile
python startup.py test
```

### Manuel Test

```bash
# Sağlık kontrolü
curl http://localhost:8000/health

# Model bilgileri
curl http://localhost:8000/model/info

# Kategoriler
curl http://localhost:8000/categories
```

### Interactive Docs

Tarayıcıda: `http://localhost:8000/docs`

## ⚙️ Konfigürasyon

### Analiz Parametreleri

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| `confidence_threshold` | 0.7 | Minimum güven skoru (0.1-1.0) |
| `max_detections` | 50 | Maksimum detection sayısı (1-200) |
| `return_masks` | true | Segmentation maskelerini döndür |
| `include_attributes` | true | Fashion özniteliklerini dahil et |

### Dosya Limitleri

| Limit | Değer |
|-------|-------|
| Maksimum dosya boyutu | 10 MB |
| Desteklenen formatlar | JPG, JPEG, PNG, BMP, TIFF, WEBP |
| Maksimum batch boyutu | 20 dosya |

## 📝 Logging

API, comprehensive logging sağlar:

```
2025-08-03 10:30:45 - INFO - Request: POST /analyze
2025-08-03 10:30:47 - INFO - Response: 200 | Time: 2.34s | Size: 15432
```

Log dosyası: `visual_analysis_api.log`

## 🔧 Troubleshooting

### Yaygın Sorunlar

**1. Model yüklenmiyor**
```bash
# Model path'ini kontrol et
export MODEL_PATH="/path/to/model.pth"

# Veya pre-trained model kullan (MODEL_PATH ayarlama)
```

**2. GPU memory hatası**
```bash
# CPU'da çalıştır
export CUDA_VISIBLE_DEVICES=""
```

**3. Port zaten kullanımda**
```bash
# Farklı port kullan
export PORT="8001"
```

**4. Bağımlılık hatası**
```bash
# Gereksinimleri tekrar yükle
pip install -r requirements.txt --force-reinstall
```

### Debug Mode

```bash
export DEBUG="true"
python main.py
```

## 🚦 Performans

### Benchmark Sonuçları

- **Tek görüntü analizi**: ~2-4 saniye (GPU), ~8-15 saniye (CPU)
- **Batch analiz (10 görüntü)**: ~15-25 saniye (GPU)
- **Memory kullanımı**: ~2-4 GB (model + inference)
- **Throughput**: ~15-30 görüntü/dakika (GPU)

### Optimizasyon İpuçları

1. **GPU kullanın**: CUDA destekli GPU ile 3-5x hızlanma
2. **Batch processing**: Çoklu görüntü için batch endpoint'i kullanın
3. **Image size**: Büyük görüntüleri resize edin (max 1024px)
4. **Workers**: Production'da multi-worker kullanın
5. **Caching**: Model caching ile startup süresini azaltın

## 📚 API Dokümantasyonu

Detaylı API dokümantasyonu için:
- Interactive docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

## 🔐 Güvenlik Notları

- Production'da CORS ayarlarını configure edin
- Trusted hosts listesi belirleyin
- Rate limiting ekleyin (gelecek)
- Authentication/Authorization ekleyin (gelecek)
- HTTPS kullanın (reverse proxy ile)

## 🛣️ Roadmap

- [ ] Docker containerization
- [ ] Authentication & authorization
- [ ] Rate limiting
- [ ] Model versioning
- [ ] Metrics & monitoring
- [ ] Caching layer
- [ ] WebSocket support
- [ ] Video analysis
- [ ] A/B testing framework

## 🤝 Katkıda Bulunma

1. Fork the repository
2. Create feature branch
3. Add tests
4. Submit pull request

## 📄 Lisans

Aura Project - Internal Use

## 📞 Destek

Sorular ve sorunlar için:
- Technical documentation: `/docs`
- Health check: `/health`
- Model info: `/model/info`
