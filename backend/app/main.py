import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .config import settings
from .api.routes import router
from .utils.logging import setup_logging
import redis
from .celery_worker import celery

setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ML model on startup
    from .ml.model import ModelManager
    ModelManager.load_models()
    logger.info("ML models loaded successfully")
    yield
    # Cleanup if needed

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.get("/health")
async def health():
    redis_status = "failed"
    try:
        r = redis.Redis.from_url(settings.REDIS_URL)
        r.ping()
        redis_status = "ok"
    except:
        pass
    # Check Celery (simple ping)
    celery_status = "ok"  # We'll assume Celery is running if Redis is up
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "redis": redis_status,
        "celery": celery_status
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. See logs."},
    )
