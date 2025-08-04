# Aura Conversational AI Service

A sophisticated microservice for fashion-focused conversational AI using hybrid QLoRA fine-tuning + RAG (Retrieval-Augmented Generation) architecture.

## 🎯 Architecture Overview

This service combines two powerful AI techniques:

1. **QLoRA Fine-Tuning**: Custom personality development for Meta-Llama-3-8B-Instruct
2. **RAG Integration**: Real-time knowledge retrieval from user's wardrobe data

## ✨ Features

- **Aura Fashion Personality**: Custom fine-tuned LLM with fashion expertise
- **QLoRA Efficiency**: 4-bit quantization for memory-efficient inference
- **RAG Integration**: Retrieval-augmented generation for personalized responses
- **Context Awareness**: Understands user preferences and wardrobe context
- **Multi-turn Conversations**: Maintains conversation history and context
- **Production Ready**: FastAPI-based microservice with monitoring

## 📁 Project Structure

```
conversational_ai/
├── src/
│   ├── finetune.py              # QLoRA fine-tuning script
│   ├── build_vector_store.py    # RAG vector store builder
│   ├── rag_service.py           # Hibrit RAG pipeline service
│   ├── rag_config_examples.py   # RAG configuration examples
│   ├── config_examples.py       # Fine-tuning configurations
│   ├── api/
│   │   ├── main.py             # FastAPI application
│   │   └── __init__.py         # API module
│   └── __init__.py             # Source module
├── app/                        # Legacy FastAPI application
├── data/                       # Training datasets & wardrobe data
├── vector_stores/             # Generated vector stores
├── saved_models/              # Fine-tuned models
├── requirements.txt           # Dependencies
├── test_finetune.py          # Fine-tuning tests
├── test_vector_store.py      # Vector store tests
├── test_rag_service.py       # RAG service tests
├── test_api.py               # FastAPI endpoint tests
├── Dockerfile                # Container configuration
├── docker-compose.yml        # Multi-service orchestration
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Model Fine-Tuning (First Time Setup)

```bash
# Download fashion dataset (Women's Clothing Reviews from Kaggle)
# Place womens_clothing_reviews.csv in data/ folder

# Fine-tune the model with QLoRA
python src/finetune.py \
    --dataset_path ./data/womens_clothing_reviews.csv \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-4 \
    --bf16

# This will create a fine-tuned model in saved_models/
```

### 2. Vector Store Creation (RAG Setup)

```bash
# Create vector store from user wardrobe data
python src/build_vector_store.py \
    --input_data_path ./data/user_wardrobe.json \
    --vector_store_type faiss \
    --embedding_model_name sentence-transformers/all-MiniLM-L6-v2

# This will create a vector store in vector_stores/
```

### 3. RAG Service Usage

```python
from src.rag_service import create_rag_service

# Create RAG service
service = create_rag_service(
    finetuned_model_path="./saved_models/aura_fashion_assistant",
    vector_store_path="./vector_stores/wardrobe_faiss.index"
)

# Generate personalized response
response = service.generate_response(
    query="What should I wear for a job interview?",
    user_id="user123"
)

print(response["response"])
```

### 4. FastAPI Service Usage

```bash
# Run the FastAPI service directly
python src/api/main.py

# Or use uvicorn
uvicorn src.api.main:app --host 0.0.0.0 --port 8003 --reload

# Test the API endpoints
curl -X POST http://localhost:8003/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What should I wear for a business meeting?",
    "user_id": "user123",
    "context": {"season": "winter", "occasion": "business"}
  }'
```

### 5. WebSocket Real-time Chat

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8003/ws/chat/user123');

// Send message
ws.send(JSON.stringify({
  message: "What outfit do you recommend for today?",
  session_id: "session_abc",
  context: {"weather": "rainy", "mood": "professional"}
}));

// Receive response
ws.onmessage = function(event) {
  const response = JSON.parse(event.data);
  console.log("Fashion advice:", response.response);
  console.log("Suggestions:", response.suggestions);
};
```

### 6. Service Deployment

```bash
# Build the service
docker build -t aura-conversational-ai .

# Run the service
docker run -p 8003:8003 aura-conversational-ai

# Or use docker-compose for full stack
docker-compose up -d

# Test the API
curl -X POST http://localhost:8003/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "123", "query": "What outfit should I wear for a job interview?"}'
```

## API Endpoints

### Chat Interface
- `POST /chat/message` - Send a message to the AI assistant
- `GET /chat/history/{user_id}` - Get conversation history
- `DELETE /chat/history/{user_id}` - Clear conversation history

### Knowledge Base
- `POST /knowledge/query` - Query the fashion knowledge base
- `GET /knowledge/topics` - Get available fashion topics
- `POST /knowledge/update` - Update knowledge base (admin)

### Personalization
- `GET /personalization/preferences/{user_id}` - Get user chat preferences
- `PUT /personalization/preferences/{user_id}` - Update chat preferences

## Model Architecture

- **Base LLM**: Fine-tuned language model for fashion conversations
- **RAG System**: Vector-based retrieval from fashion knowledge base
- **Context Manager**: Manages conversation history and user context

## 🚀 Quick Start

### 1. Model Fine-Tuning (First Time Setup)

```bash
# Download fashion dataset (Women's Clothing Reviews from Kaggle)
# Place womens_clothing_reviews.csv in data/ folder

# Fine-tune the model with QLoRA
python src/finetune.py \
    --dataset_path ./data/womens_clothing_reviews.csv \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-4 \
    --bf16

# This will create a fine-tuned model in saved_models/
```

### 2. Service Deployment

```bash
# Build the service
docker build -t aura-conversational-ai .

# Run the service
docker run -p 8003:8003 aura-conversational-ai

# Test the API
curl -X POST http://localhost:8003/chat/message \
  -H "Content-Type: application/json" \
  -d '{"user_id": "123", "message": "What outfit should I wear for a job interview?"}'
```

## 🔧 Fine-Tuning Configuration

The service uses QLoRA (Quantized LoRA) for efficient fine-tuning:

- **Base Model**: Meta-Llama-3-8B-Instruct
- **Quantization**: 4-bit (NF4) for memory efficiency
- **LoRA Rank**: 16 (configurable)
- **Target Modules**: Attention and MLP layers
- **Memory Usage**: ~16GB GPU VRAM

For detailed fine-tuning instructions, see [src/README.md](src/README.md).
