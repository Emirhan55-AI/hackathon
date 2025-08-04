#!/bin/bash
# Aura AI Platform - Data Directory Setup Script
# Bu script, tüm persistent data dizinlerini oluşturur

echo "🏗️ Aura AI Platform - Data directory'leri oluşturuluyor..."

# Ana data dizini
mkdir -p data

# Visual Analysis service dizinleri
mkdir -p data/visual_analysis/models
mkdir -p data/visual_analysis/logs

# Outfit Recommendation service dizinleri
mkdir -p data/outfit_recommendation/models
mkdir -p data/outfit_recommendation/logs

# Conversational AI service dizinleri
mkdir -p data/conversational_ai/models
mkdir -p data/conversational_ai/vector_store
mkdir -p data/conversational_ai/logs

# Database dizinleri
mkdir -p data/postgres
mkdir -p data/redis

# Monitoring dizinleri
mkdir -p data/prometheus
mkdir -p data/grafana

# Nginx log dizini
mkdir -p data/nginx/logs

# Dizin izinlerini ayarla (Linux/macOS için)
if [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "darwin"* ]]; then
    echo "📁 Dizin izinleri ayarlanıyor..."
    
    # AI model dizinleri için geniş izinler
    chmod -R 755 data/visual_analysis/models
    chmod -R 755 data/outfit_recommendation/models
    chmod -R 755 data/conversational_ai/models
    chmod -R 755 data/conversational_ai/vector_store
    
    # Log dizinleri için yazma izni
    chmod -R 755 data/*/logs
    chmod -R 755 data/nginx/logs
    
    # Database dizinleri için özel izinler
    chmod -R 700 data/postgres  # PostgreSQL için güvenli izinler
    chmod -R 755 data/redis     # Redis için standart izinler
    
    # Monitoring dizinleri
    chmod -R 755 data/prometheus
    chmod -R 755 data/grafana
fi

echo "✅ Data directory'leri başarıyla oluşturuldu!"
echo ""
echo "📂 Oluşturulan dizinler:"
echo "   📁 data/visual_analysis/models - DETR modelleri"
echo "   📁 data/outfit_recommendation/models - OutfitTransformer modelleri"
echo "   📁 data/conversational_ai/models - QLoRA/LLM modelleri"
echo "   📁 data/conversational_ai/vector_store - FAISS indeks dosyaları"
echo "   📁 data/postgres - PostgreSQL veritabanı"
echo "   📁 data/redis - Redis önbellek"
echo "   📁 data/prometheus - Prometheus metrikleri"
echo "   📁 data/grafana - Grafana konfigürasyonu"
echo "   📁 data/*/logs - Servis logları"
echo ""
echo "🚀 Artık 'docker-compose up' ile sistemini başlatabilirsin!"
echo "   Tüm veriler host makinada kalıcı olarak saklanacak."
