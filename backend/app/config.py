from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import validator
import os

class Settings(BaseSettings):
    APP_NAME: str = "StarWhisper"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    SECRET_KEY: str = "change-this-in-production"
    REDIS_URL: str = "redis://redis:6379/0"
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/0"
    MODEL_PATH: str = "/app/backend/app/ml/artifacts/xgboost_model.joblib"
    FEATURE_NAMES_PATH: str = "/app/backend/app/ml/artifacts/feature_names.joblib"
    SCALER_PATH: str = "/app/backend/app/ml/artifacts/scaler.joblib"
    FRONTEND_URL: str = "http://localhost:5173"
    MAX_UPLOAD_SIZE: int = 104857600  # 100 MB
    CATALOG_PATH: str = "/data/catalog.csv"

    @validator('SECRET_KEY')
    def check_secret_key(cls, v):
        if v == "change-this-in-production" and os.getenv("ENVIRONMENT") == "production":
            raise ValueError("SECRET_KEY must be changed in production")
        return v

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
