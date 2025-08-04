# Aura AI Platform - Production Deployment Guide

## Overview

The Aura AI Platform is a comprehensive fashion AI assistant consisting of three main microservices:

- **Visual Analysis Service**: DETR-based computer vision for fashion item detection and analysis
- **Outfit Recommendation Service**: OutfitTransformer-based style recommendations
- **Conversational AI Service**: Hybrid QLoRA + RAG chatbot for personalized fashion advice

All services are orchestrated behind an Nginx API Gateway with monitoring, caching, and database support.

## Prerequisites

### Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (RTX 3060 or better recommended)
- **RAM**: Minimum 16GB (32GB recommended for production)
- **Storage**: 100GB+ free space for models and data
- **CPU**: Multi-core processor (8+ cores recommended)

### Software Requirements

- Docker and Docker Compose
- NVIDIA Docker runtime (for GPU support)
- Git

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd aura_ai_platform

# Copy environment template
cp .env.example .env
```

### 2. Configure Environment

Edit `.env` file with your API keys and settings:

```bash
# Required: Hugging Face token for LLaMA model access
HF_TOKEN=your_huggingface_token_here

# Optional: Cloud services
PINECONE_API_KEY=your_pinecone_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Security
POSTGRES_PASSWORD=your_secure_password
GRAFANA_PASSWORD=your_grafana_password
JWT_SECRET_KEY=your_jwt_secret

# Performance
DEVICE=cuda  # or 'cpu' if no GPU
MAX_WORKERS=4
```

### 3. Deploy Platform

```bash
# Build and start all services
docker-compose up -d

# Monitor startup
docker-compose logs -f
```

### 4. Verify Deployment

- **API Gateway**: http://localhost (main entry point)
- **Grafana Monitoring**: http://localhost:3000 (admin/[your_password])
- **Prometheus Metrics**: http://localhost:9090

## API Endpoints

### Visual Analysis Service
```
POST /api/v1/visual-analysis/analyze
GET  /api/v1/visual-analysis/health
GET  /api/v1/visual-analysis/docs
```

### Outfit Recommendation Service
```
POST /api/v1/outfit-recommendation/recommend
POST /api/v1/outfit-recommendation/evaluate
GET  /api/v1/outfit-recommendation/health
GET  /api/v1/outfit-recommendation/docs
```

### Conversational AI Service
```
POST /api/v1/conversational-ai/chat
WS   /api/v1/conversational-ai/ws/chat
POST /api/v1/conversational-ai/rag/query
GET  /api/v1/conversational-ai/health
GET  /api/v1/conversational-ai/docs
```

## Service Architecture

```
┌─────────────────┐
│   Nginx Gateway │ :80
├─────────────────┤
│  Load Balancer  │
│  Rate Limiting  │
│  SSL Termination│
└─────────┬───────┘
          │
    ┌─────┴─────┐
    │           │
┌───▼───┐   ┌───▼───┐   ┌─────────▼──────────┐
│Visual │   │Outfit │   │  Conversational AI │
│Analysis│   │Recomm │   │   (QLoRA + RAG)    │
│       │   │       │   │                    │
│:8000  │   │:8001  │   │      :8003         │
└───────┘   └───────┘   └────────────────────┘
     │           │              │
     └───────────┼──────────────┘
                 │
    ┌────────────┴─────────────┐
    │                          │
┌───▼────┐  ┌─────▼─────┐  ┌───▼──────┐
│ Redis  │  │PostgreSQL │  │Prometheus│
│        │  │           │  │          │
│ :6379  │  │   :5432   │  │  :9090   │
└────────┘  └───────────┘  └──────────┘
```

## Monitoring and Observability

### Health Checks

All services include comprehensive health checks:

```bash
# Check overall platform status
curl http://localhost/health

# Check individual services
curl http://localhost/api/v1/visual-analysis/health
curl http://localhost/api/v1/outfit-recommendation/health
curl http://localhost/api/v1/conversational-ai/health
```

### Metrics and Monitoring

- **Prometheus**: Collects metrics from all services
- **Grafana**: Provides dashboards and alerting
- **Service Logs**: Centralized logging for debugging

### Performance Monitoring

Key metrics tracked:
- Request latency and throughput
- GPU utilization and memory
- Model inference times
- Queue depths and processing rates
- Error rates and availability

## Scaling and Performance

### Horizontal Scaling

Scale individual services based on load:

```bash
# Scale visual analysis service
docker-compose up -d --scale visual_analysis_service=3

# Scale outfit recommendation service
docker-compose up -d --scale outfit_recommendation_service=2
```

### Performance Tuning

1. **GPU Allocation**: Adjust GPU memory per service
2. **Worker Processes**: Tune `MAX_WORKERS` per service needs
3. **Caching**: Configure Redis for optimal cache hit rates
4. **Database**: Optimize PostgreSQL for read/write patterns

## Security Considerations

### API Security

- Rate limiting configured via Nginx
- CORS policies enforced
- JWT authentication for user sessions
- Input validation on all endpoints

### Data Security

- Database encryption at rest
- Secure password hashing (bcrypt)
- API key management via environment variables
- Network isolation via Docker networks

## Troubleshooting

### Common Issues

1. **GPU Not Detected**
   ```bash
   # Check NVIDIA Docker runtime
   docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
   ```

2. **Model Download Failures**
   ```bash
   # Verify HF_TOKEN is set correctly
   docker-compose logs conversational_ai_service
   ```

3. **Out of Memory Errors**
   ```bash
   # Reduce MAX_WORKERS or model batch sizes
   # Monitor with: docker stats
   ```

4. **Service Not Starting**
   ```bash
   # Check logs for specific service
   docker-compose logs [service_name]
   ```

### Performance Issues

1. **Slow Response Times**
   - Check GPU utilization in Grafana
   - Monitor queue depths
   - Consider scaling services

2. **High Memory Usage**
   - Adjust model quantization settings
   - Tune cache sizes
   - Monitor with `docker stats`

### Database Issues

1. **Connection Failures**
   ```bash
   # Check PostgreSQL status
   docker-compose exec postgres pg_isready -U aura_user
   ```

2. **Migration Issues**
   ```bash
   # Manually run initialization
   docker-compose exec postgres psql -U aura_user -d aura_platform -f /docker-entrypoint-initdb.d/init.sql
   ```

## Development Mode

For development and testing:

```bash
# Start with development overrides
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Enable debug logging
export LOG_LEVEL=DEBUG

# Access services directly (bypassing gateway)
curl http://localhost:8000/health  # Visual Analysis
curl http://localhost:8001/health  # Outfit Recommendation  
curl http://localhost:8003/health  # Conversational AI
```

## Backup and Recovery

### Database Backup

```bash
# Create backup
docker-compose exec postgres pg_dump -U aura_user aura_platform > backup.sql

# Restore backup
docker-compose exec -T postgres psql -U aura_user aura_platform < backup.sql
```

### Model and Data Backup

```bash
# Backup model volumes
docker run --rm -v aura_ai_platform_visual_analysis_models:/data -v $(pwd):/backup alpine tar czf /backup/models_backup.tar.gz /data

# Backup vector store
docker run --rm -v aura_ai_platform_conversational_ai_vector_store:/data -v $(pwd):/backup alpine tar czf /backup/vector_store_backup.tar.gz /data
```

## Support and Maintenance

### Regular Maintenance

1. **Update Models**: Regularly update AI models for better performance
2. **Database Maintenance**: Run VACUUM and ANALYZE on PostgreSQL
3. **Log Rotation**: Configure log rotation to prevent disk space issues
4. **Security Updates**: Keep Docker images and dependencies updated

### Monitoring Alerts

Set up alerts for:
- High error rates (>5%)
- Response time degradation (>2s)
- GPU memory exhaustion (>90%)
- Database connection issues
- Disk space usage (>80%)

## API Documentation

Complete API documentation is available at:
- Visual Analysis: http://localhost/api/v1/visual-analysis/docs
- Outfit Recommendation: http://localhost/api/v1/outfit-recommendation/docs  
- Conversational AI: http://localhost/api/v1/conversational-ai/docs

## License and Credits

Aura AI Platform - Fashion AI Assistant
Built with PyTorch, Transformers, FastAPI, and Docker.
