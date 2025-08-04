"""
OutfitTransformer Service Startup Script - Aura Project
Bu script, OutfitTransformer microservice'ini başlatmak için kullanılır.
"""

import os
import sys
import logging
import time
import argparse
from pathlib import Path
import subprocess
import json
from typing import Dict, Any, Optional

# Environment setup
sys.path.append(str(Path(__file__).parent / "src"))

# Import after path setup
try:
    from inference import OutfitRecommendationEngine, create_demo_items_database
    from train import train_outfit_transformer
except ImportError as e:
    print(f"Import hatası: {e}")
    print("src/ dizininin Python path'te olduğundan emin olun")
    sys.exit(1)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Service configuration
SERVICE_CONFIG = {
    "service_name": "outfit-recommendation",
    "host": "0.0.0.0",
    "port": 8001,
    "reload": False,
    "workers": 1,
    "log_level": "info",
    "model_path": "./models/outfit_transformer_best.pt",
    "items_database_path": "./data/items_database.json",
    "demo_mode": False,
    "enable_auth": False
}

# Environment variables mapping
ENV_MAPPING = {
    "HOST": "host",
    "PORT": "port", 
    "MODEL_PATH": "model_path",
    "ITEMS_DB_PATH": "items_database_path",
    "DEMO_MODE": "demo_mode",
    "ENABLE_AUTH": "enable_auth",
    "LOG_LEVEL": "log_level",
    "WORKERS": "workers"
}


def load_config_from_env() -> Dict[str, Any]:
    """
    Environment variable'lardan konfigürasyonu yükler
    
    Returns:
        Dict[str, Any]: Updated configuration
    """
    config = SERVICE_CONFIG.copy()
    
    for env_var, config_key in ENV_MAPPING.items():
        env_value = os.getenv(env_var)
        if env_value is not None:
            # Type conversion
            if config_key in ["port", "workers"]:
                config[config_key] = int(env_value)
            elif config_key in ["demo_mode", "enable_auth", "reload"]:
                config[config_key] = env_value.lower() in ["true", "1", "yes"]
            else:
                config[config_key] = env_value
    
    return config


def setup_directories(config: Dict[str, Any]) -> None:
    """
    Gerekli dizinleri oluşturur
    
    Args:
        config: Service konfigürasyonu
    """
    directories = [
        "./models",
        "./data", 
        "./logs",
        "./uploads",
        "./demo_images"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Dizin oluşturuldu: {directory}")


def check_model_availability(model_path: str) -> bool:
    """
    Model dosyasının varlığını kontrol eder
    
    Args:
        model_path: Model dosya path'i
        
    Returns:
        bool: Model mevcut mu
    """
    if Path(model_path).exists():
        logger.info(f"Model bulundu: {model_path}")
        return True
    else:
        logger.warning(f"Model bulunamadı: {model_path}")
        return False


def setup_demo_environment(config: Dict[str, Any]) -> None:
    """
    Demo ortamını kurulur
    
    Args:
        config: Service konfigürasyonu
    """
    logger.info("Demo ortamı kuruluyor...")
    
    # Demo items database
    demo_db_path = "./data/demo_items_database.json"
    if not Path(demo_db_path).exists():
        logger.info("Demo items database oluşturuluyor...")
        
        try:
            create_demo_items_database("./demo_images", demo_db_path)
            config["items_database_path"] = demo_db_path
            logger.info(f"Demo database oluşturuldu: {demo_db_path}")
        except Exception as e:
            logger.error(f"Demo database oluşturulamadı: {e}")
    
    # Demo images (basit renkli görüntüler)
    demo_images_dir = Path("./demo_images")
    demo_images_dir.mkdir(exist_ok=True)
    
    if not any(demo_images_dir.iterdir()):
        logger.info("Demo görüntüler oluşturuluyor...")
        create_demo_images(demo_images_dir)


def create_demo_images(images_dir: Path) -> None:
    """
    Demo için basit görüntüler oluşturur
    
    Args:
        images_dir: Görüntü dizini
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        import random
    except ImportError:
        logger.warning("PIL mevcut değil, demo görüntüler oluşturulamıyor")
        return
    
    categories = ["tops", "bottoms", "shoes", "accessories", "outerwear"]
    colors = ["black", "white", "blue", "red", "brown", "gray"]
    styles = ["casual", "formal", "trendy", "classic", "sporty"]
    
    item_id = 1
    for category in categories:
        for color in colors[:3]:  # Limit combinations
            for style in styles[:2]:
                # Create colored image
                image = Image.new('RGB', (224, 224), color=color)
                
                # Add text
                try:
                    draw = ImageDraw.Draw(image)
                    text = f"{category}\n{color}\n{style}"
                    draw.text((10, 10), text, fill="white" if color == "black" else "black")
                except Exception:
                    pass  # Font hatası için fallback
                
                # Save image
                filename = f"demo_{category}_{color}_{style}.jpg"
                image.save(images_dir / filename)
                
                item_id += 1
                
                if item_id > 30:  # Limit total images
                    return
    
    logger.info(f"Demo görüntüler oluşturuldu: {images_dir}")


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Konfigürasyonu validate eder
    
    Args:
        config: Service konfigürasyonu
        
    Returns:
        bool: Konfigürasyon geçerli mi
    """
    required_keys = ["host", "port", "model_path", "items_database_path"]
    
    for key in required_keys:
        if key not in config:
            logger.error(f"Gerekli konfigürasyon eksik: {key}")
            return False
    
    # Port validation
    if not (1000 <= config["port"] <= 65535):
        logger.error(f"Geçersiz port: {config['port']}")
        return False
    
    # Workers validation
    if config["workers"] < 1:
        logger.error(f"Geçersiz worker sayısı: {config['workers']}")
        return False
    
    return True


def start_training(data_dir: str, 
                   image_dir: str, 
                   output_dir: str,
                   config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Model eğitimini başlatır
    
    Args:
        data_dir: Polyvore data dizini
        image_dir: Görüntü dizini  
        output_dir: Model output dizini
        config: Training konfigürasyonu
        
    Returns:
        bool: Eğitim başarılı mı
    """
    logger.info("Model eğitimi başlatılıyor...")
    
    try:
        training_summary = train_outfit_transformer(
            data_dir=data_dir,
            image_dir=image_dir,
            output_dir=output_dir,
            config=config,
            use_wandb=False  # Production'da True yapılabilir
        )
        
        logger.info("Model eğitimi tamamlandı!")
        logger.info(f"En iyi F1 skoru: {training_summary.get('best_f1_score', 'N/A')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Model eğitimi hatası: {e}")
        return False


def test_service_endpoints(host: str, port: int) -> bool:
    """
    Service endpoint'lerini test eder
    
    Args:
        host: Service host
        port: Service port
        
    Returns:
        bool: Test başarılı mı
    """
    try:
        import requests
        
        base_url = f"http://{host}:{port}"
        
        # Health check
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code != 200:
            logger.error(f"Health check başarısız: {response.status_code}")
            return False
        
        logger.info("Service endpoints test edildi: ✓")
        return True
        
    except Exception as e:
        logger.error(f"Endpoint test hatası: {e}")
        return False


def start_uvicorn_server(config: Dict[str, Any]) -> None:
    """
    Uvicorn server'ı başlatır
    
    Args:
        config: Service konfigürasyonu
    """
    try:
        import uvicorn
        
        # Set environment variables for the app
        for env_var, config_key in ENV_MAPPING.items():
            if config_key in config:
                os.environ[env_var] = str(config[config_key])
        
        logger.info(f"OutfitTransformer service başlatılıyor...")
        logger.info(f"Host: {config['host']}")
        logger.info(f"Port: {config['port']}")
        logger.info(f"Workers: {config['workers']}")
        logger.info(f"Demo mode: {config['demo_mode']}")
        logger.info(f"Model path: {config['model_path']}")
        
        # Start server
        uvicorn.run(
            "main:app",
            host=config["host"],
            port=config["port"],
            reload=config["reload"],
            workers=config["workers"],
            log_level=config["log_level"],
            access_log=True
        )
        
    except KeyboardInterrupt:
        logger.info("Service durduruldu (Ctrl+C)")
    except Exception as e:
        logger.error(f"Server başlatma hatası: {e}")
        sys.exit(1)


def main():
    """Ana startup fonksiyonu"""
    parser = argparse.ArgumentParser(description="OutfitTransformer Service Startup")
    
    parser.add_argument("--mode", choices=["serve", "train", "test"], default="serve",
                       help="Çalışma modu")
    parser.add_argument("--config", type=str, default=None,
                       help="Konfigürasyon dosyası")
    parser.add_argument("--demo", action="store_true",
                       help="Demo mode'da çalıştır")
    parser.add_argument("--host", type=str, default=None,
                       help="Server host")
    parser.add_argument("--port", type=int, default=None,
                       help="Server port")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Model dosya path'i")
    parser.add_argument("--data-dir", type=str, default=None,
                       help="Training data dizini")
    parser.add_argument("--image-dir", type=str, default=None,
                       help="Görüntü dizini")
    parser.add_argument("--output-dir", type=str, default="./models",
                       help="Model output dizini")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config_from_env()
    
    # Load config file if provided
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    # Override with CLI args
    if args.demo:
        config["demo_mode"] = True
    if args.host:
        config["host"] = args.host
    if args.port:
        config["port"] = args.port
    if args.model_path:
        config["model_path"] = args.model_path
    
    # Validate configuration
    if not validate_config(config):
        logger.error("Konfigürasyon hatası")
        sys.exit(1)
    
    # Setup directories
    setup_directories(config)
    
    # Mode-specific execution
    if args.mode == "train":
        # Training mode
        if not args.data_dir or not args.image_dir:
            logger.error("Training için --data-dir ve --image-dir gerekli")
            sys.exit(1)
        
        success = start_training(
            data_dir=args.data_dir,
            image_dir=args.image_dir,
            output_dir=args.output_dir
        )
        
        if not success:
            sys.exit(1)
    
    elif args.mode == "test":
        # Test mode
        logger.info("Test mode başlatılıyor...")
        
        # Create test environment
        if config["demo_mode"]:
            setup_demo_environment(config)
        
        # Run tests
        try:
            import pytest
            
            # Run API tests
            exit_code = pytest.main([
                "test_api.py",
                "-v",
                "--tb=short"
            ])
            
            sys.exit(exit_code)
            
        except ImportError:
            logger.error("pytest mevcut değil. pip install pytest yapın")
            sys.exit(1)
    
    else:
        # Serve mode (default)
        logger.info("Service mode başlatılıyor...")
        
        # Demo setup if needed
        if config["demo_mode"]:
            setup_demo_environment(config)
        
        # Check model availability
        model_available = check_model_availability(config["model_path"])
        
        if not model_available and not config["demo_mode"]:
            logger.warning("Model mevcut değil. Demo mode'u etkinleştirmeyi düşünün (--demo)")
        
        # Start server
        start_uvicorn_server(config)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Startup interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Startup hatası: {e}", exc_info=True)
        sys.exit(1)
