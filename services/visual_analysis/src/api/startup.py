"""
Startup Script for Aura Visual Analysis API
Bu script, visual analysis API'sini çalıştırmak için kullanılır.
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

def check_requirements():
    """Gerekli paketlerin yüklenip yüklenmediğini kontrol et"""
    required_packages = [
        "fastapi",
        "uvicorn", 
        "torch",
        "transformers",
        "PIL",
        "numpy"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            if package == "PIL":
                try:
                    __import__("Pillow")
                except ImportError:
                    missing_packages.append("Pillow")
            else:
                missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Eksik paketler: {', '.join(missing_packages)}")
        print("Yüklemek için: pip install -r requirements.txt")
        return False
    
    print("✅ Tüm gerekli paketler yüklü")
    return True

def setup_environment():
    """Ortam değişkenlerini ayarla"""
    # Model path
    model_path = os.getenv("MODEL_PATH")
    if not model_path:
        # Varsayılan model path'ini ayarla
        current_dir = Path(__file__).parent
        default_model_path = current_dir.parent.parent / "saved_models" / "detr_fashionpedia.pth"
        os.environ["MODEL_PATH"] = str(default_model_path)
        print(f"📦 Model path ayarlandı: {default_model_path}")
    
    # API host ve port
    if not os.getenv("HOST"):
        os.environ["HOST"] = "0.0.0.0"
    
    if not os.getenv("PORT"):
        os.environ["PORT"] = "8000"
    
    # Debug mode
    if not os.getenv("DEBUG"):
        os.environ["DEBUG"] = "false"
    
    print(f"🌐 Host: {os.getenv('HOST')}:{os.getenv('PORT')}")
    print(f"🐛 Debug: {os.getenv('DEBUG')}")

def start_api_development():
    """Development mode'da API'yi başlat"""
    print("🚀 Development mode'da API başlatılıyor...")
    
    api_path = Path(__file__).parent / "main.py"
    
    cmd = [
        sys.executable, 
        str(api_path)
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 API durduruldu")
    except subprocess.CalledProcessError as e:
        print(f"❌ API başlatma hatası: {e}")

def start_api_production(workers: int = 1):
    """Production mode'da API'yi başlat"""
    print(f"🏭 Production mode'da API başlatılıyor ({workers} worker)...")
    
    host = os.getenv("HOST", "0.0.0.0")
    port = os.getenv("PORT", "8000")
    
    cmd = [
        "uvicorn",
        "main:app",
        "--host", host,
        "--port", port,
        "--workers", str(workers),
        "--log-level", "info",
        "--access-log"
    ]
    
    try:
        subprocess.run(cmd, check=True, cwd=Path(__file__).parent)
    except KeyboardInterrupt:
        print("\n🛑 API durduruldu")
    except FileNotFoundError:
        print("❌ uvicorn bulunamadı. Yüklemek için: pip install uvicorn")
    except subprocess.CalledProcessError as e:
        print(f"❌ API başlatma hatası: {e}")

def test_api():
    """API'yi test et"""
    print("🧪 API test ediliyor...")
    
    test_script_path = Path(__file__).parent / "test_api.py"
    
    if not test_script_path.exists():
        print("❌ Test scripti bulunamadı")
        return
    
    cmd = [sys.executable, str(test_script_path)]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Test hatası: {e}")

def show_api_info():
    """API hakkında bilgi göster"""
    print("📊 Aura Visual Analysis API Bilgileri")
    print("=" * 50)
    print("🔍 API Açıklaması:")
    print("   DETR tabanlı fashion analysis mikroservisi")
    print("   294 Fashionpedia kategorisini destekler")
    print()
    print("🌐 Ana Endpoint'ler:")
    print("   GET  /health        - Sağlık kontrolü")
    print("   POST /analyze       - Tek görüntü analizi")
    print("   POST /analyze/batch - Çoklu görüntü analizi")
    print("   GET  /categories    - Fashion kategorileri")
    print("   GET  /model/info    - Model bilgileri")
    print("   GET  /docs          - API dokümantasyonu")
    print()
    print("📦 Desteklenen Formatlar:")
    print("   .jpg, .jpeg, .png, .bmp, .tiff, .webp")
    print()
    print("⚙️ Ortam Değişkenleri:")
    print(f"   MODEL_PATH = {os.getenv('MODEL_PATH', 'Ayarlanmamış')}")
    print(f"   HOST = {os.getenv('HOST', '0.0.0.0')}")
    print(f"   PORT = {os.getenv('PORT', '8000')}")
    print(f"   DEBUG = {os.getenv('DEBUG', 'false')}")

def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(
        description="Aura Visual Analysis API Startup Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python startup.py dev                    # Development mode
  python startup.py prod --workers 4      # Production mode
  python startup.py test                  # API'yi test et
  python startup.py info                  # API bilgilerini göster
        """
    )
    
    parser.add_argument(
        "mode",
        choices=["dev", "prod", "test", "info"],
        help="Çalıştırma modu"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Production mode için worker sayısı (varsayılan: 1)"
    )
    
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Sadece bağımlılıkları kontrol et"
    )
    
    args = parser.parse_args()
    
    print("🔍 Aura Visual Analysis API - Startup Script")
    print("=" * 60)
    
    # Bağımlılık kontrolü
    if not check_requirements():
        if args.check_deps:
            sys.exit(1)
        
        response = input("Devam etmek istiyor musunuz? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    if args.check_deps:
        print("✅ Bağımlılık kontrolü tamamlandı")
        return
    
    # Ortam ayarları
    setup_environment()
    
    # Mode'a göre işlem
    if args.mode == "dev":
        start_api_development()
        
    elif args.mode == "prod":
        start_api_production(args.workers)
        
    elif args.mode == "test":
        test_api()
        
    elif args.mode == "info":
        show_api_info()

if __name__ == "__main__":
    main()
