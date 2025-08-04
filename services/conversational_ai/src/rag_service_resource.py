"""
RAG Service Resource Manager - Bellek Sızıntısı Koruması
Bu modül, RAGService'in kaynaklarının güvenli bir şekilde yönetilmesini sağlar.
Büyük modeller ve GPU belleği için bellek sızıntısı koruması sunar.
"""

import weakref
import logging
import gc
import time
from typing import Optional, TYPE_CHECKING
from contextlib import contextmanager

if TYPE_CHECKING:
    from src.rag_service import RAGService

# Logger setup
logger = logging.getLogger(__name__)

# Global service tracking for debugging
_service_refs = weakref.WeakSet()


class RAGServiceResource:
    """
    RAGService için context manager ve kaynak yöneticisi.
    Bellek sızıntısını önlemek için büyük modellerin referanslarını yönetir.
    """
    
    def __init__(self, service: 'RAGService'):
        """
        RAGServiceResource'ı başlatır
        
        Args:
            service: Yönetilecek RAGService instance'ı
        """
        self.service = service
        self._service_id = id(service) if service else None
        self._entered = False
        self._cleanup_called = False
        
        if service:
            logger.debug(f"🔧 RAGServiceResource oluşturuldu: {self._service_id}")
            # Servisi takip et
            track_service(service)
        else:
            logger.warning("⚠️ RAGServiceResource None service ile oluşturuldu")
    
    def __enter__(self):
        """
        Context manager giriş noktası
        
        Returns:
            RAGService: Yönetilen service instance'ı
        """
        self._entered = True
        
        if self.service:
            logger.debug(f"🔄 RAGServiceResource girildi: {self._service_id}")
        else:
            logger.warning("⚠️ RAGServiceResource None service ile girildi")
        
        return self.service
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager çıkış noktası - Kaynak temizleme
        
        Args:
            exc_type: Exception tipi
            exc_val: Exception değeri  
            exc_tb: Exception traceback
        """
        logger.debug(f"🔄 RAGServiceResource çıkılıyor: {self._service_id}")
        
        try:
            # Service cleanup çağır
            if self.service and hasattr(self.service, 'cleanup'):
                try:
                    logger.debug(f"🧹 RAGService cleanup çağrılıyor: {self._service_id}")
                    self.service.cleanup()
                    self._cleanup_called = True
                except Exception as e:
                    logger.error(f"❌ RAGService cleanup hatası: {str(e)}")
            
            # PyTorch GPU bellek temizliği
            try:
                import torch
                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    logger.debug("🖥️ GPU bellek cache temizleniyor...")
                    torch.cuda.empty_cache()
                    
                    # GPU bellek istatistikleri
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated()
                        cached = torch.cuda.memory_reserved()
                        logger.debug(f"📊 GPU Bellek - Allocated: {allocated/1024**2:.1f}MB, Cached: {cached/1024**2:.1f}MB")
                        
            except ImportError:
                logger.debug("📝 PyTorch bulunamadı, GPU temizleme atlandı")
            except Exception as e:
                logger.warning(f"⚠️ GPU bellek temizleme hatası: {str(e)}")
            
            # Service referansını temizle
            if self.service:
                logger.debug(f"🔗 Service referansı temizleniyor: {self._service_id}")
                self.service = None
            
            # Agresif garbage collection (sadece gerektiğinde)
            if self._cleanup_called:
                logger.debug("🗑️ Garbage collection tetikleniyor...")
                collected = gc.collect()
                if collected > 0:
                    logger.debug(f"🗑️ {collected} nesne garbage collection ile temizlendi")
                    
        except Exception as e:
            logger.error(f"❌ RAGServiceResource çıkış hatası: {str(e)}")
        
        finally:
            self._entered = False
            logger.debug(f"✅ RAGServiceResource çıkıldı: {self._service_id}")
            
        # Exception'ı yeniden fırlatma (False döndür)
        return False
    
    def force_cleanup(self):
        """
        Zorunlu temizleme işlemi (emergency use)
        """
        if not self._cleanup_called and self.service:
            logger.warning(f"⚠️ Force cleanup çağrıldı: {self._service_id}")
            self.__exit__(None, None, None)
    
    def is_valid(self) -> bool:
        """
        Resource'un geçerli olup olmadığını kontrol eder
        
        Returns:
            bool: Resource geçerliyse True
        """
        return self.service is not None and not self._cleanup_called
    
    def get_memory_stats(self) -> dict:
        """
        Bellek istatistiklerini döndürür
        
        Returns:
            dict: Bellek durumu bilgileri
        """
        stats = {
            "service_valid": self.is_valid(),
            "cleanup_called": self._cleanup_called,
            "service_id": self._service_id,
            "tracked_services": get_tracked_service_count()
        }
        
        # PyTorch GPU stats
        try:
            import torch
            if torch.cuda.is_available():
                stats.update({
                    "gpu_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                    "gpu_cached_mb": torch.cuda.memory_reserved() / 1024**2,
                    "gpu_device_count": torch.cuda.device_count()
                })
        except ImportError:
            stats["gpu_info"] = "PyTorch not available"
        except Exception as e:
            stats["gpu_error"] = str(e)
        
        return stats
    
    def __del__(self):
        """
        Destructor - son güvenlik önlemi
        """
        if self.service and not self._cleanup_called:
            logger.warning(f"⚠️ RAGServiceResource destructor cleanup: {self._service_id}")
            try:
                self.force_cleanup()
            except:
                pass


# Service tracking functions
def track_service(service: 'RAGService') -> None:
    """
    RAGService instance'ını weak reference ile takip eder
    
    Args:
        service: Takip edilecek RAGService instance'ı
    """
    if service:
        _service_refs.add(service)
        service_id = id(service)
        logger.debug(f"📊 Service tracked: {service_id} (Toplam: {len(_service_refs)})")


def get_tracked_service_count() -> int:
    """
    Takip edilen service sayısını döndürür
    
    Returns:
        int: Aktif service sayısı
    """
    return len(_service_refs)


def get_tracked_services_info() -> dict:
    """
    Takip edilen servislerin detaylı bilgilerini döndürür
    
    Returns:
        dict: Service bilgileri
    """
    services_info = []
    
    for service in _service_refs:
        try:
            service_info = {
                "id": id(service),
                "type": type(service).__name__,
                "model_loaded": hasattr(service, 'model') and service.model is not None,
                "tokenizer_loaded": hasattr(service, 'tokenizer') and service.tokenizer is not None,
                "embedding_model_loaded": hasattr(service, 'embedding_model') and service.embedding_model is not None,
                "vector_store_loaded": hasattr(service, 'vector_store') and service.vector_store is not None,
            }
            services_info.append(service_info)
        except Exception as e:
            logger.warning(f"⚠️ Service info alma hatası: {str(e)}")
    
    return {
        "total_tracked": len(_service_refs),
        "services": services_info,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }


def clear_all_tracked_services() -> int:
    """
    Tüm takip edilen servisleri temizler (debugging için)
    
    Returns:
        int: Temizlenen service sayısı
    """
    count = len(_service_refs)
    _service_refs.clear()
    logger.info(f"🧹 {count} tracked service temizlendi")
    return count


@contextmanager
def rag_service_context(service: 'RAGService'):
    """
    RAGService için context manager helper fonksiyonu
    
    Args:
        service: Yönetilecek RAGService instance'ı
        
    Yields:
        RAGService: Yönetilen service instance'ı
    """
    resource = RAGServiceResource(service)
    try:
        with resource as managed_service:
            yield managed_service
    finally:
        # Resource otomatik olarak temizlenecek
        pass


def force_cleanup_all_resources() -> dict:
    """
    Acil durum bellek temizleme fonksiyonu
    
    Returns:
        dict: Temizleme sonucu bilgileri
    """
    logger.warning("🚨 Force cleanup all resources çağrıldı")
    
    cleanup_stats = {
        "tracked_services_before": get_tracked_service_count(),
        "gpu_cleanup": False,
        "gc_collected": 0,
        "errors": []
    }
    
    try:
        # GPU bellek temizliği
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                cleanup_stats["gpu_cleanup"] = True
                logger.info("🖥️ GPU bellek force cleanup tamamlandı")
        except Exception as e:
            cleanup_stats["errors"].append(f"GPU cleanup error: {str(e)}")
        
        # Aggressive garbage collection
        for _ in range(3):  # 3 defa gc.collect() çağır
            collected = gc.collect()
            cleanup_stats["gc_collected"] += collected
        
        # Tracked services temizle
        cleared_count = clear_all_tracked_services()
        cleanup_stats["tracked_services_cleared"] = cleared_count
        
        logger.info(f"🧹 Force cleanup tamamlandı: {cleanup_stats}")
        
    except Exception as e:
        cleanup_stats["errors"].append(f"General cleanup error: {str(e)}")
        logger.error(f"❌ Force cleanup hatası: {str(e)}")
    
    return cleanup_stats


# Health check fonksiyonu
def get_resource_health() -> dict:
    """
    Kaynak yönetimi sağlık durumunu döndürür
    
    Returns:
        dict: Sağlık durumu bilgileri
    """
    health_info = {
        "status": "healthy",
        "tracked_services": get_tracked_service_count(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # GPU durumu
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            cached = torch.cuda.memory_reserved()
            max_memory = torch.cuda.get_device_properties(0).total_memory
            
            health_info["gpu"] = {
                "available": True,
                "allocated_mb": allocated / 1024**2,
                "cached_mb": cached / 1024**2,
                "total_mb": max_memory / 1024**2,
                "utilization_percent": (allocated / max_memory) * 100
            }
            
            # Yüksek bellek kullanımı uyarısı
            if (allocated / max_memory) > 0.8:
                health_info["status"] = "warning"
                health_info["warning"] = "High GPU memory utilization"
                
        else:
            health_info["gpu"] = {"available": False}
            
    except ImportError:
        health_info["gpu"] = {"available": False, "reason": "PyTorch not installed"}
    except Exception as e:
        health_info["gpu"] = {"available": False, "error": str(e)}
    
    return health_info
