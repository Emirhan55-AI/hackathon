"""
Test Script for Aura Visual Analysis API
Bu script, visual analysis API'sinin çeşitli endpoint'lerini test eder.
"""

import requests
import json
import time
from pathlib import Path
from PIL import Image
import io
import base64

# API base URL
API_BASE_URL = "http://localhost:8000"

class APITester:
    """Visual Analysis API test sınıfı"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "AuraAPI-Tester/1.0"
        })
    
    def test_health_check(self):
        """Health check endpoint'ini test et"""
        print("🏥 Health check testi...")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Servis durumu: {data['status']}")
                print(f"📊 Model yüklü: {data['model_loaded']}")
                print(f"📅 Versiyon: {data['version']}")
            else:
                print(f"❌ Health check başarısız: {response.text}")
                
        except Exception as e:
            print(f"❌ Health check hatası: {e}")
    
    def test_model_info(self):
        """Model info endpoint'ini test et"""
        print("\n📊 Model info testi...")
        
        try:
            response = self.session.get(f"{self.base_url}/model/info")
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Model tipi: {data['model_info']['model_type']}")
                print(f"📦 Toplam kategori: {data['total_categories']}")
                print(f"🏷️ İlk 5 kategori: {list(data['categories'].values())[:5]}")
            else:
                print(f"❌ Model info başarısız: {response.text}")
                
        except Exception as e:
            print(f"❌ Model info hatası: {e}")
    
    def create_test_image(self, width: int = 640, height: int = 480) -> bytes:
        """Test için basit bir görüntü oluştur"""
        # Basit bir renkli görüntü oluştur
        image = Image.new("RGB", (width, height), color="lightblue")
        
        # Bazı geometrik şekiller ekle (fashion item benzeri)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        
        # Dikdörtgen (gömlek benzeri)
        draw.rectangle([100, 100, 300, 300], fill="darkblue", outline="black", width=2)
        
        # Daire (düğme benzeri)
        draw.ellipse([180, 180, 220, 220], fill="white", outline="black", width=1)
        
        # BytesIO'ya kaydet
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        
        return img_bytes.getvalue()
    
    def test_single_analysis(self, test_image_path: str = None):
        """Tek görüntü analizi testi"""
        print("\n🔍 Tek görüntü analizi testi...")
        
        try:
            # Test görüntüsü hazırla
            if test_image_path and Path(test_image_path).exists():
                with open(test_image_path, "rb") as f:
                    image_data = f.read()
                filename = Path(test_image_path).name
            else:
                print("📸 Test görüntüsü oluşturuluyor...")
                image_data = self.create_test_image()
                filename = "test_image.png"
            
            # Analiz parametreleri
            files = {"file": (filename, image_data, "image/png")}
            data = {
                "confidence_threshold": 0.5,
                "max_detections": 20,
                "return_masks": True,
                "include_attributes": True
            }
            
            print(f"📤 Görüntü gönderiliyor: {filename}")
            start_time = time.time()
            
            response = self.session.post(
                f"{self.base_url}/analyze",
                files=files,
                data=data
            )
            
            processing_time = time.time() - start_time
            print(f"⏱️ İstek süresi: {processing_time:.2f}s")
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Analiz başarılı!")
                print(f"🎯 Tespit sayısı: {len(result['detections'])}")
                print(f"⚡ Sunucu işlem süresi: {result['processing_time']:.2f}s")
                
                # İlk detection'ı göster
                if result['detections']:
                    first_detection = result['detections'][0]
                    print(f"🏷️ İlk tespit: {first_detection['label']} (güven: {first_detection['confidence']:.2f})")
                
                # Özet bilgiler
                summary = result['summary']
                print(f"📊 Benzersiz kategori: {summary['unique_categories']}")
                print(f"📏 Görüntü boyutu: {summary['image_dimensions']}")
                
            else:
                print(f"❌ Analiz başarısız: {response.text}")
                
        except Exception as e:
            print(f"❌ Analiz testi hatası: {e}")
    
    def test_batch_analysis(self, num_images: int = 3):
        """Batch analiz testi"""
        print(f"\n📦 Batch analiz testi ({num_images} görüntü)...")
        
        try:
            # Çoklu test görüntüsü oluştur
            files = []
            for i in range(num_images):
                image_data = self.create_test_image(
                    width=400 + i*50, 
                    height=300 + i*50
                )
                files.append(
                    ("files", (f"test_image_{i+1}.png", image_data, "image/png"))
                )
            
            # Batch parametreleri
            data = {
                "confidence_threshold": 0.6,
                "max_detections": 15,
                "return_masks": False,  # Batch'te mask'ler kapalı
                "include_attributes": True
            }
            
            print(f"📤 {num_images} görüntü gönderiliyor...")
            start_time = time.time()
            
            response = self.session.post(
                f"{self.base_url}/analyze/batch",
                files=files,
                data=data
            )
            
            processing_time = time.time() - start_time
            print(f"⏱️ Toplam istek süresi: {processing_time:.2f}s")
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                results = response.json()
                print(f"✅ Batch analiz başarılı!")
                print(f"📊 Sonuç sayısı: {len(results)}")
                
                # Her sonucu özetle
                for i, result in enumerate(results):
                    if result['success']:
                        print(f"  📸 Görüntü {i+1}: {len(result['detections'])} tespit")
                    else:
                        print(f"  ❌ Görüntü {i+1}: Hata")
                        
            else:
                print(f"❌ Batch analiz başarısız: {response.text}")
                
        except Exception as e:
            print(f"❌ Batch analiz testi hatası: {e}")
    
    def test_categories_endpoint(self):
        """Categories endpoint'ini test et"""
        print("\n🏷️ Categories endpoint testi...")
        
        try:
            response = self.session.get(f"{self.base_url}/categories")
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Kategoriler alındı!")
                print(f"📊 Toplam kategori: {data['total_count']}")
                print(f"🎨 Attribute türleri: {list(data['attributes'].keys())}")
            else:
                print(f"❌ Categories başarısız: {response.text}")
                
        except Exception as e:
            print(f"❌ Categories testi hatası: {e}")
    
    def test_stats_endpoint(self):
        """Stats endpoint'ini test et"""
        print("\n📈 Stats endpoint testi...")
        
        try:
            response = self.session.get(f"{self.base_url}/stats")
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ İstatistikler alındı!")
                print(f"🔧 API versiyonu: {data['api_version']}")
                print(f"📊 Model durumu: {data['model_loaded']}")
                print(f"📁 Desteklenen formatlar: {len(data['supported_formats'])}")
                print(f"🌐 Endpoint sayısı: {len(data['endpoints'])}")
            else:
                print(f"❌ Stats başarısız: {response.text}")
                
        except Exception as e:
            print(f"❌ Stats testi hatası: {e}")
    
    def test_error_handling(self):
        """Hata işleme testi"""
        print("\n❌ Hata işleme testi...")
        
        # 1. Geçersiz dosya formatı
        print("  1️⃣ Geçersiz dosya formatı testi...")
        try:
            files = {"file": ("test.txt", b"This is a text file", "text/plain")}
            response = self.session.post(f"{self.base_url}/analyze", files=files)
            print(f"     Status: {response.status_code} (beklenen: 415)")
        except Exception as e:
            print(f"     Hata: {e}")
        
        # 2. Çok büyük dosya (simüle)
        print("  2️⃣ Büyük dosya testi...")
        try:
            # 15MB fake data (limit 10MB)
            large_data = b"0" * (15 * 1024 * 1024)
            files = {"file": ("large.jpg", large_data, "image/jpeg")}
            response = self.session.post(f"{self.base_url}/analyze", files=files)
            print(f"     Status: {response.status_code} (beklenen: 413)")
        except Exception as e:
            print(f"     Hata: {e}")
        
        # 3. Geçersiz parametreler
        print("  3️⃣ Geçersiz parametreler testi...")
        try:
            image_data = self.create_test_image()
            files = {"file": ("test.png", image_data, "image/png")}
            data = {"confidence_threshold": 1.5}  # Geçersiz threshold
            response = self.session.post(f"{self.base_url}/analyze", files=files, data=data)
            print(f"     Status: {response.status_code} (beklenen: 422)")
        except Exception as e:
            print(f"     Hata: {e}")
    
    def run_all_tests(self, test_image_path: str = None):
        """Tüm testleri çalıştır"""
        print("🧪 Aura Visual Analysis API - Kapsamlı Test Süreci")
        print("=" * 60)
        
        # Temel endpoint testleri
        self.test_health_check()
        self.test_model_info()
        self.test_categories_endpoint()
        self.test_stats_endpoint()
        
        # Analiz testleri
        self.test_single_analysis(test_image_path)
        self.test_batch_analysis()
        
        # Hata handling testleri
        self.test_error_handling()
        
        print("\n" + "=" * 60)
        print("🎉 Tüm testler tamamlandı!")


def main():
    """Ana test fonksiyonu"""
    print("Aura Visual Analysis API Test Scripti")
    print("API URL'si kontrol ediliyor...")
    
    tester = APITester()
    
    # Önce sunucunun çalışıp çalışmadığını kontrol et
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API sunucusu erişilebilir")
        else:
            print(f"⚠️ API sunucusu yanıt veriyor ama durum: {response.status_code}")
    except Exception as e:
        print(f"❌ API sunucusuna erişilemiyor: {e}")
        print("Sunucuyu başlatmak için: python main.py")
        return
    
    # Test image path (opsiyonel)
    test_image_path = None
    
    # Eğer test klasöründe örnek görüntü varsa kullan
    possible_test_images = [
        "test_image.jpg",
        "sample.png", 
        "fashion_test.jpg",
        "../test_images/sample.jpg"
    ]
    
    for path in possible_test_images:
        if Path(path).exists():
            test_image_path = path
            print(f"📸 Test görüntüsü bulundu: {path}")
            break
    
    if not test_image_path:
        print("📸 Test görüntüsü bulunamadı, sentetik görüntü kullanılacak")
    
    # Tüm testleri çalıştır
    tester.run_all_tests(test_image_path)


if __name__ == "__main__":
    main()
