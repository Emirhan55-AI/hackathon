# Aura AI Platform 🌟

> **Advanced Fashion AI Assistant with Computer Vision, Recommendation Engine, and Conversational AI**

A production-ready, enterprise-grade AI platform that revolutionizes fashion and style through cutting-edge artificial intelligence. The platform combines computer vision, advanced recommendation algorithms, and conversational AI to provide personalized fashion assistance.

## 🎯 Overview

The Aura AI Platform consists of three integrated microservices:

### 🔍 Visual Analysis Service
- **DETR-based Computer Vision**: State-of-the-art object detection for fashion items
- **Multi-category Detection**: Recognizes clothing, accessories, and style elements
- **Real-time Processing**: Fast inference with GPU acceleration
- **Detailed Analysis**: Color, pattern, style, and fit analysis

### 👗 Outfit Recommendation Service  
- **OutfitTransformer Architecture**: Advanced neural network for style recommendations
- **Personalized Suggestions**: Learns user preferences and style patterns
- **Context-Aware**: Considers occasion, weather, and personal style
- **Style Scoring**: Quantitative fashion compatibility assessment

### 💬 Conversational AI Service
- **Hybrid QLoRA + RAG**: Fine-tuned LLaMA-3-8B with retrieval-augmented generation
- **Fashion Expertise**: Specialized knowledge in style, trends, and fashion advice
- **Real-time Chat**: WebSocket support for interactive conversations
- **Memory & Context**: Maintains conversation history and user preferences

## Project Structure

```
aura_ai_platform/
├── README.md
├── docker-compose.yml
├── shared/                          # Shared components
│   ├── config/                      # Configuration management
│   │   ├── settings.py             # Application settings
│   │   └── logging.py              # Logging configuration
│   ├── models/                     # Shared data models
│   │   └── base.py                 # Pydantic models
│   └── utils/                      # Shared utilities
│       └── image_processing.py     # Image processing functions
├── services/
│   ├── visual_analysis/            # Visual Analysis Service
│   │   ├── app/
│   │   │   ├── main.py            # FastAPI application
│   │   │   ├── api/               # API endpoints
│   │   │   │   └── analysis.py    # Image analysis endpoints
│   │   │   └── core/              # Core logic
│   │   │       ├── model.py       # DETR model implementation
│   │   │       └── inference.py   # Inference engine
│   │   ├── tests/                 # Unit and integration tests
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── README.md
│   ├── outfit_recommendation/      # Outfit Recommendation Service
│   │   ├── app/
│   │   │   ├── main.py            # FastAPI application
│   │   │   ├── api/               # API endpoints
│   │   │   │   ├── recommendations.py # Recommendation endpoints
│   │   │   │   └── style.py       # Style analysis endpoints
│   │   │   └── core/              # Core logic
│   │   │       ├── model.py       # OutfitTransformer model
│   │   │       └── inference.py   # Recommendation engine
│   │   ├── tests/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── README.md
│   └── conversational_ai/          # Conversational AI Service
│       ├── app/
│       │   ├── main.py            # FastAPI application with WebSocket
│       │   ├── api/               # API endpoints
│       │   │   ├── chat.py        # Chat endpoints
│       │   │   ├── knowledge.py   # Knowledge base endpoints
│       │   │   └── personalization.py # User preferences
│       │   └── core/              # Core logic
│       │       ├── model.py       # RAG + LLM implementation
│       │       └── inference.py   # Conversation engine
│       ├── tests/
│       ├── Dockerfile
│       ├── requirements.txt
│       └── README.md
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.9+
- NVIDIA GPU (recommended for production)

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd aura_ai_platform

# Build and run all services
docker-compose up --build

# Or run individual services
cd services/visual_analysis
docker build -t aura-visual-analysis .
docker run -p 8001:8001 aura-visual-analysis
```

### API Testing

```bash
# Visual Analysis
curl -X POST http://localhost:8001/analyze/image \
  -F "image=@sample_image.jpg"

# Outfit Recommendations
curl -X POST http://localhost:8002/recommendations/generate \
  -H "Content-Type: application/json" \
  -d '{"user_id": "123", "context": {"occasion": "casual"}}'

# Conversational AI
curl -X POST http://localhost:8003/chat/message \
  -H "Content-Type: application/json" \
  -d '{"user_id": "123", "message": "What should I wear today?"}'
```

## Development

Each microservice can be developed and tested independently.

### Shared Components

Common components are located in the `shared/` directory and used by all services.

### API Documentation

Visit `/docs` endpoint for detailed API documentation when services are running.
