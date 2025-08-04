# Aura Conversational AI Service Dockerfile
# Multi-stage build for optimized production image

# Stage 1: Base Python environment
FROM python:3.10-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user
RUN groupadd -r aura && useradd -r -g aura aura && mkdir -p /home/aura

# Stage 2: Dependencies installation
FROM base as dependencies

# Copy requirements
COPY services/conversational_ai/requirements.txt /tmp/
WORKDIR /tmp

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Stage 3: Application
FROM dependencies as application

# Set working directory
WORKDIR /app

# Copy application code
COPY services/conversational_ai/src/ /app/src/
COPY shared/ /app/shared/
COPY services/conversational_ai/test_*.py /app/
COPY services/conversational_ai/*.md /app/

# Create necessary directories
RUN mkdir -p /app/saved_models \
    /app/vector_stores \
    /app/data \
    /app/logs \
    /home/aura/.cache \
    /home/aura/.cache/huggingface \
    /home/aura/.huggingface

# Set ownership to aura user
RUN chown -R aura:aura /app /home/aura

# Switch to non-root user
USER aura

# Environment variables for production
ENV PYTHONPATH=/app
ENV RAG_CONFIG=production
ENV HOST=0.0.0.0
ENV PORT=8003
ENV WORKERS=1
ENV DEBUG=false
ENV HF_HOME=/home/aura/.huggingface
ENV TRANSFORMERS_CACHE=/home/aura/.cache/huggingface

# Expose port
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# Default command - Use our RAG-enabled API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8003"]
