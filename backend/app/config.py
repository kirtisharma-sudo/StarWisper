from pydantic import validator

class Settings(BaseSettings):
    ...
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100 MB
    CATALOG_PATH: str = "/data/catalog.csv"
    
    @validator('SECRET_KEY')
    def check_secret_key(cls, v):
        if v == "change-this-in-production" and ENVIRONMENT == "production":
            raise ValueError("SECRET_KEY must be changed in production")
        return v
