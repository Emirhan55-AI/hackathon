"""
Shared configuration settings for Aura AI Platform
"""

import os
from typing import Optional
from pydantic import BaseSettings


class DatabaseSettings(BaseSettings):
    """Database configuration"""
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5432"))
    username: str = os.getenv("DB_USERNAME", "aura")
    password: str = os.getenv("DB_PASSWORD", "aura_password")
    database: str = os.getenv("DB_NAME", "aura_ai")

    @property
    def url(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class RedisSettings(BaseSettings):
    """Redis configuration"""
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    password: Optional[str] = os.getenv("REDIS_PASSWORD")
    db: int = int(os.getenv("REDIS_DB", "0"))

    @property
    def url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class ModelSettings(BaseSettings):
    """Model configuration"""
    model_path: str = os.getenv("MODEL_PATH", "/app/models")
    cache_ttl: int = int(os.getenv("MODEL_CACHE_TTL", "3600"))
    batch_size: int = int(os.getenv("MODEL_BATCH_SIZE", "8"))


class APISettings(BaseSettings):
    """API configuration"""
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


class Settings(BaseSettings):
    """Main settings class"""
    service_name: str = os.getenv("SERVICE_NAME", "aura-ai-service")
    environment: str = os.getenv("ENVIRONMENT", "development")
    version: str = "1.0.0"
    
    # Component settings
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    model: ModelSettings = ModelSettings()
    api: APISettings = APISettings()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
