@echo off
REM Aura AI Platform - Data Directory Setup Script (Windows)
REM Bu script, tüm persistent data dizinlerini oluşturur

echo 🏗️ Aura AI Platform - Data directory'leri oluşturuluyor...

REM Ana data dizini
if not exist "data" mkdir "data"

REM Visual Analysis service dizinleri
if not exist "data\visual_analysis" mkdir "data\visual_analysis"
if not exist "data\visual_analysis\models" mkdir "data\visual_analysis\models"
if not exist "data\visual_analysis\logs" mkdir "data\visual_analysis\logs"

REM Outfit Recommendation service dizinleri
if not exist "data\outfit_recommendation" mkdir "data\outfit_recommendation"
if not exist "data\outfit_recommendation\models" mkdir "data\outfit_recommendation\models"
if not exist "data\outfit_recommendation\logs" mkdir "data\outfit_recommendation\logs"

REM Conversational AI service dizinleri
if not exist "data\conversational_ai" mkdir "data\conversational_ai"
if not exist "data\conversational_ai\models" mkdir "data\conversational_ai\models"
if not exist "data\conversational_ai\vector_store" mkdir "data\conversational_ai\vector_store"
if not exist "data\conversational_ai\logs" mkdir "data\conversational_ai\logs"

REM Database dizinleri
if not exist "data\postgres" mkdir "data\postgres"
if not exist "data\redis" mkdir "data\redis"

REM Monitoring dizinleri
if not exist "data\prometheus" mkdir "data\prometheus"
if not exist "data\grafana" mkdir "data\grafana"

REM Nginx log dizini
if not exist "data\nginx" mkdir "data\nginx"
if not exist "data\nginx\logs" mkdir "data\nginx\logs"

echo ✅ Data directory'leri başarıyla oluşturuldu!
echo.
echo 📂 Oluşturulan dizinler:
echo    📁 data\visual_analysis\models - DETR modelleri
echo    📁 data\outfit_recommendation\models - OutfitTransformer modelleri
echo    📁 data\conversational_ai\models - QLoRA/LLM modelleri
echo    📁 data\conversational_ai\vector_store - FAISS indeks dosyaları
echo    📁 data\postgres - PostgreSQL veritabanı
echo    📁 data\redis - Redis önbellek
echo    📁 data\prometheus - Prometheus metrikleri
echo    📁 data\grafana - Grafana konfigürasyonu
echo    📁 data\*\logs - Servis logları
echo.
echo 🚀 Artık 'docker-compose up' ile sistemini başlatabilirsin!
echo    Tüm veriler host makinada kalıcı olarak saklanacak.
pause
