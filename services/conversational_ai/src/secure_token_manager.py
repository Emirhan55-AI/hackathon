"""
Secure Token Manager - Güvenli Token Yönetimi Modülü
Bu modül, HuggingFace API token'ları gibi hassas bilgilerin güvenli bir şekilde
yönetilmesini sağlar ve log dosyalarında açık metin olarak görünmelerini engeller.
"""

import os
import logging
from typing import Optional

# Logger setup
logger = logging.getLogger(__name__)


class SecureTokenManager:
    """
    Güvenli token yönetimi için sınıf.
    Token'ları ortam değişkenlerinden yükler, doğrular ve maskeleyerek loglar.
    """
    
    def __init__(self, env_var_name: str = "HF_TOKEN"):
        """
        SecureTokenManager'ı başlatır
        
        Args:
            env_var_name: Token'ın bulunduğu ortam değişkeninin adı
        """
        self.env_var_name = env_var_name
        self._token: Optional[str] = None
        self._is_validated: bool = False
    
    def get_token(self) -> str:
        """
        Token'ı döndürür. İlk çağrıda token'ı yükler ve doğrular.
        
        Returns:
            str: Geçerli token
            
        Raises:
            ValueError: Token bulunamadığında veya geçersiz olduğunda
        """
        if self._token is None:
            self._load_and_validate_token()
        
        return self._token
    
    def _load_and_validate_token(self):
        """
        Token'ı ortam değişkeninden yükler ve temel doğrulama yapar
        
        Raises:
            ValueError: Token bulunamadığında veya geçersiz olduğunda
        """
        # Ortam değişkeninden token'ı al
        token = os.getenv(self.env_var_name)
        
        if not token:
            raise ValueError(f"{self.env_var_name} ortam değişkeni bulunamadı.")
        
        # Temel doğrulama - boş string kontrolü
        if not token.strip():
            raise ValueError(f"{self.env_var_name} ortam değişkeni boş.")
        
        # Varsayılan değer kontrolü
        if token == "your_huggingface_token_here":
            raise ValueError(f"{self.env_var_name} ortam değişkeni varsayılan değerde bırakılmış.")
        
        # Minimum uzunluk kontrolü (HuggingFace token'ları genellikle en az 20 karakter)
        if len(token) < 10:
            raise ValueError(f"{self.env_var_name} çok kısa. Geçerli bir token olduğundan emin olun.")
        
        # Token'ı ata ve doğrulandı olarak işaretle
        self._token = token
        self._is_validated = True
        
        # Maskelenmiş token ile log
        masked = self.mask_token(token)
        logger.info(f"🔐 {self.env_var_name} yüklendi (maskelenmiş: {masked})")
    
    def mask_token(self, token: str = None) -> str:
        """
        Token'ı maskeler - güvenlik için sadece başlangıç ve sonu görünür
        
        Args:
            token: Maskelenecek token. None ise, yüklü token kullanılır.
            
        Returns:
            str: Maskelenmiş token
        """
        if token is None:
            token = self._token
        
        if not token:
            return "***"
        
        token_length = len(token)
        
        # Çok kısa token'lar için tamamını maskele
        if token_length <= 6:
            return "*" * token_length
        
        # İlk 2 ve son 2 karakter görünür, ortası maskelenmiş
        visible_start = 2
        visible_end = 2
        
        if token_length <= 8:
            # Kısa token'lar için sadece ilk ve son karakter
            visible_start = 1
            visible_end = 1
        
        masked_middle_length = token_length - visible_start - visible_end
        masked_middle = "*" * masked_middle_length
        
        return token[:visible_start] + masked_middle + token[-visible_end:]
    
    def is_valid(self) -> bool:
        """
        Token'ın doğrulanmış olup olmadığını kontrol eder
        
        Returns:
            bool: Token doğrulanmışsa True
        """
        return self._is_validated and self._token is not None
    
    def clear_token(self):
        """
        Bellek güvenliği için token'ı temizler
        """
        if self._token:
            self._token = None
            self._is_validated = False
            logger.debug(f"🧹 {self.env_var_name} temizlendi")


# Global instance yönetimi (Singleton pattern)
_hf_token_manager: Optional[SecureTokenManager] = None


def get_hf_token_manager() -> SecureTokenManager:
    """
    Global HuggingFace token manager instance'ını döndürür.
    İlk çağrıda instance'ı oluşturur (Singleton pattern).
    
    Returns:
        SecureTokenManager: Global token manager instance'ı
    """
    global _hf_token_manager
    
    if _hf_token_manager is None:
        _hf_token_manager = SecureTokenManager(env_var_name="HF_TOKEN")
    
    return _hf_token_manager


def get_token_manager(env_var_name: str) -> SecureTokenManager:
    """
    Belirtilen ortam değişkeni için yeni bir token manager oluşturur
    
    Args:
        env_var_name: Ortam değişkeninin adı
        
    Returns:
        SecureTokenManager: Yeni token manager instance'ı
    """
    return SecureTokenManager(env_var_name=env_var_name)


# Cleanup fonksiyonu
def cleanup_all_tokens():
    """
    Tüm global token manager'ları temizler
    """
    global _hf_token_manager
    
    if _hf_token_manager:
        _hf_token_manager.clear_token()
        _hf_token_manager = None
        logger.info("🧹 Tüm global token manager'lar temizlendi")
