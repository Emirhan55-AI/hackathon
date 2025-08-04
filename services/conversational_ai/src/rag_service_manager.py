"""
RAG Service Manager - Thread-Safe RAG Service Yönetimi
Bu modül, RAGService'in güvenli ve asenkron başlatılmasını sağlar.
Yarış koşullarını önler ve lazy loading ile performansı optimize eder.
"""

import asyncio
import logging
import time
from typing import Optional, TYPE_CHECKING
from contextlib import asynccontextmanager

if TYPE_CHECKING:
    from src.rag_service import RAGService, RAGConfig

# Logger setup
logger = logging.getLogger(__name__)

# Local imports
try:
    from src.rag_service import RAGService, RAGConfig
except ImportError:
    try:
        from rag_service import RAGService, RAGConfig
    except ImportError:
        RAGService = None
        RAGConfig = None
        logger.error("RAGService ve RAGConfig import edilemedi")


class RAGServiceManager:
    """
    Thread-safe ve asenkron RAG Service yöneticisi.
    Yarış koşullarını önler ve lazy loading ile performansı optimize eder.
    """
    
    def __init__(self, config):
        """
        RAGServiceManager'ı başlatır
        
        Args:
            config: RAG konfigürasyon nesnesi
        """
        self.config = config
        self.service = None
        self._initializing: bool = False
        self._initialized: bool = False
        self._lock = asyncio.Lock()
        self._initialization_error: Optional[Exception] = None
        self._start_time: Optional[float] = None
        
        logger.info("🔧 RAGServiceManager oluşturuldu")
    
    async def initialize(self) -> None:
        """
        RAGService'i asenkron olarak başlatır.
        Thread-safe ve idempotent (tekrar çağrılabilir).
        
        Raises:
            RuntimeError: Başlatma sırasında hata oluşursa
        """
        async with self._lock:
            # Zaten başlatılmışsa, tekrar başlatma
            if self._initialized:
                logger.debug("✅ RAG Service zaten başlatılmış")
                return
            
            # Başka bir thread başlatma yapıyorsa, bekleme
            if self._initializing:
                logger.info("⏳ RAG Service başlatılıyor - bekleniyor...")
                
                # Başlatma bitene kadar bekle
                max_wait_time = 300  # 5 dakika maksimum bekleme
                wait_start = time.time()
                
                while self._initializing and (time.time() - wait_start) < max_wait_time:
                    await asyncio.sleep(0.1)
                
                # Timeout kontrolü
                if self._initializing:
                    raise RuntimeError("RAG Service başlatma timeout'a uğradı")
                
                # Başlatma tamamlandıktan sonra durum kontrolü
                if self._initialized:
                    logger.info("✅ RAG Service başlatma tamamlandı")
                    return
                elif self._initialization_error:
                    raise self._initialization_error
                else:
                    raise RuntimeError("RAG Service başlatma bilinmeyen bir nedenle başarısız oldu")
            
            # Başlatma işlemini başlat
            self._initializing = True
            self._initialization_error = None
            self._start_time = time.time()
            
            logger.info("🚀 RAG Service başlatılıyor...")
            
            try:
                # RAGService oluştur
                if RAGService is None:
                    raise RuntimeError("RAGService sınıfı import edilemedi")
                
                self.service = RAGService(self.config)
                
                # Başlatma süresini hesapla
                if self._start_time:
                    initialization_time = time.time() - self._start_time
                    logger.info(f"✅ RAG Service başarıyla başlatıldı ({initialization_time:.2f}s)")
                else:
                    logger.info("✅ RAG Service başarıyla başlatıldı")
                
                self._initialized = True
                
            except Exception as e:
                # Hata durumunda state'i temizle
                self._initialization_error = e
                self._initialized = False
                self.service = None
                
                error_msg = f"❌ RAG Service başlatma hatası: {str(e)}"
                logger.error(error_msg)
                
                # Hatayı yeniden fırlat
                raise RuntimeError(error_msg) from e
                
            finally:
                # Her durumda initializing flag'ini temizle
                self._initializing = False
    
    def is_ready(self) -> bool:
        """
        RAG Service'in hazır olup olmadığını kontrol eder
        
        Returns:
            bool: Service hazırsa True, aksi halde False
        """
        return self._initialized and self.service is not None
    
    def is_initializing(self) -> bool:
        """
        RAG Service'in şu anda başlatılmakta olup olmadığını kontrol eder
        
        Returns:
            bool: Başlatılıyorsa True, aksi halde False
        """
        return self._initializing
    
    def get_service(self):
        """
        RAG Service instance'ını döndürür
        
        Returns:
            RAGService: Hazır RAG service instance'ı
            
        Raises:
            RuntimeError: Service hazır değilse
        """
        if not self.is_ready():
            if self._initialization_error:
                raise RuntimeError(f"RAG Service başlatılamadı: {str(self._initialization_error)}")
            else:
                raise RuntimeError("RAG Service henüz hazır değil veya başlatılamadı.")
        
        return self.service
    
    def get_status(self) -> dict:
        """
        RAG Service Manager'ın durumu hakkında bilgi döndürür
        
        Returns:
            dict: Durum bilgileri
        """
        status = {
            "initialized": self._initialized,
            "initializing": self._initializing,
            "ready": self.is_ready(),
            "has_error": self._initialization_error is not None,
            "service_loaded": self.service is not None
        }
        
        if self._initialization_error:
            status["error"] = str(self._initialization_error)
        
        if self._start_time and self._initialized:
            status["initialization_time"] = time.time() - self._start_time
        
        return status
    
    async def shutdown(self) -> None:
        """
        RAG Service'i güvenli bir şekilde kapatır
        """
        async with self._lock:
            if self.service:
                try:
                    # RAG Service'in cleanup metodları varsa çağır
                    if hasattr(self.service, 'clear_cache'):
                        self.service.clear_cache()
                    
                    logger.info("🔄 RAG Service kapatılıyor...")
                    
                except Exception as e:
                    logger.error(f"⚠️ RAG Service kapatma sırasında hata: {str(e)}")
                
                finally:
                    self.service = None
                    self._initialized = False
                    self._initializing = False
                    self._initialization_error = None
                    
                    logger.info("✅ RAG Service kapatıldı")
    
    def __del__(self):
        """
        Destructor - cleanup işlemleri
        """
        if self.service:
            try:
                if hasattr(self.service, 'clear_cache'):
                    self.service.clear_cache()
            except:
                pass


# Global instance yönetimi
rag_service_manager_instance: Optional[RAGServiceManager] = None


def get_rag_service_manager() -> RAGServiceManager:
    """
    Global RAG Service Manager instance'ını döndürür
    
    Returns:
        RAGServiceManager: Global manager instance'ı
        
    Raises:
        RuntimeError: Manager henüz oluşturulmamışsa
    """
    global rag_service_manager_instance
    
    if rag_service_manager_instance is None:
        raise RuntimeError(
            "RAG Service Manager henüz oluşturulmamış. "
            "Önce create_rag_service_manager() fonksiyonunu çağırın."
        )
    
    return rag_service_manager_instance


def create_rag_service_manager(config) -> RAGServiceManager:
    """
    Global RAG Service Manager instance'ını oluşturur
    
    Args:
        config: RAG konfigürasyon nesnesi
        
    Returns:
        RAGServiceManager: Yeni oluşturulan manager instance'ı
        
    Raises:
        RuntimeError: Manager zaten oluşturulmuşsa
    """
    global rag_service_manager_instance
    
    if rag_service_manager_instance is not None:
        raise RuntimeError(
            "RAG Service Manager zaten oluşturulmuş. "
            "Tekrar oluşturmak için önce reset_rag_service_manager() çağırın."
        )
    
    if config is None:
        raise ValueError("RAGConfig nesnesi gerekli")
    
    rag_service_manager_instance = RAGServiceManager(config)
    logger.info("🔧 Global RAG Service Manager oluşturuldu")
    
    return rag_service_manager_instance


def reset_rag_service_manager() -> None:
    """
    Global RAG Service Manager'ı sıfırlar (test amaçlı)
    """
    global rag_service_manager_instance
    
    if rag_service_manager_instance:
        logger.info("🔄 Global RAG Service Manager sıfırlanıyor...")
        rag_service_manager_instance = None
        logger.info("✅ Global RAG Service Manager sıfırlandı")


async def ensure_rag_service_ready():
    """
    RAG Service'in hazır olduğundan emin olur ve service'i döndürür.
    Convenience function for endpoints.
    
    Returns:
        RAGService: Hazır RAG service instance'ı
        
    Raises:
        RuntimeError: Service hazırlanamadığında
    """
    try:
        manager = get_rag_service_manager()
        
        if not manager.is_ready():
            logger.info("🔄 RAG Service hazır değil, başlatılıyor...")
            await manager.initialize()
        
        return manager.get_service()
        
    except Exception as e:
        logger.error(f"❌ RAG Service hazırlama hatası: {str(e)}")
        raise RuntimeError(f"RAG Service hazırlanamadı: {str(e)}") from e


# Health check fonksiyonu
def get_rag_service_health() -> dict:
    """
    RAG Service'in sağlık durumunu döndürür
    
    Returns:
        dict: Sağlık durumu bilgileri
    """
    try:
        manager = get_rag_service_manager()
        return {
            "status": "healthy" if manager.is_ready() else "initializing" if manager.is_initializing() else "unhealthy",
            "details": manager.get_status()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "details": {"initialized": False, "ready": False}
        }
