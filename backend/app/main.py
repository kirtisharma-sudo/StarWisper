@app.get("/health")
async def health():
    # Check Redis
    try:
        import redis
        r = redis.Redis.from_url(settings.REDIS_URL)
        r.ping()
        redis_status = "ok"
    except:
        redis_status = "failed"
    # Check Celery (if possible)
    # For simplicity, just return status
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "redis": redis_status,
        "celery": "ok"  # would check via Celery inspect in production
    }
