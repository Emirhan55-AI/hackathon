# OutfitTransformer Microservice - Aura Project

OutfitTransformer microservice, Aura AI Platform'un outfit recommendation ve compatibility analysis kısmını sağlayan gelişmiş bir yapay zeka servisidir. Transformer tabanlı derin öğrenme modeli kullanarak moda item'ları arasındaki uyumluluğu analiz eder ve kişiselleştirilmiş outfit önerileri oluşturur.

## Key Features

### 🎯 Core Functionality
- **Outfit Compatibility Analysis**: Item'lar arasındaki uyumluluğu analiz eder
- **Outfit Recommendations**: Seed item'lara göre uyumlu outfitler önerir
- **Item Recommendations**: Belirli bir item için complementary item'lar önerir
- **Fashion Rule Engine**: Moda kurallarına göre compatibility kontrolü
- **Graph-based Analysis**: Outfit'leri graph yapısında analiz eder

### 🧠 AI Model Architecture
- **OutfitTransformer**: Custom transformer architecture for fashion
- **ResNet50 Backbone**: Image feature extraction
- **Multi-modal Fusion**: Images + attributes (category, color, style)
- **Attention Mechanisms**: Context-aware compatibility modeling
- **Graph Neural Networks**: Outfit relationship modeling

### 📊 Data Management
- **Polyvore Dataset Integration**: Fashion compatibility learning
- **FAISS Vector Search**: Efficient similarity search
- **Dynamic Item Database**: Real-time item addition/management
- **Fashion Attribute Embeddings**: Learned representations

### 🌐 API Features
- **RESTful API**: Comprehensive endpoints
- **FastAPI Framework**: High-performance async API
- **Authentication Support**: Token-based security
- **Comprehensive Testing**: Unit, integration, performance tests
- **Demo Mode**: Easy development and testing

## API Endpoints

### Generate Recommendations
- `POST /recommendations/generate` - Generate outfit recommendations
- `POST /recommendations/compatibility` - Score item compatibility
- `GET /recommendations/user/{user_id}` - Get user-specific recommendations

### Style Analysis
- `POST /style/analyze` - Analyze style preferences from wardrobe
- `PUT /style/profile/{user_id}` - Update user style profile
- `GET /style/trends` - Get current style trends

## Model Architecture

- **OutfitTransformer**: Transformer-based model for outfit compatibility
- **Style Encoder**: Encode personal style preferences
- **Context Embeddings**: Weather, occasion, and seasonal awareness

## Quick Start

```bash
# Build the service
docker build -t aura-outfit-recommendation .

# Run the service
docker run -p 8002:8002 aura-outfit-recommendation

# Test the API
curl -X POST http://localhost:8002/recommendations/generate \
  -H "Content-Type: application/json" \
  -d '{"user_id": "123", "context": {"weather": "sunny", "occasion": "casual"}}'
```
