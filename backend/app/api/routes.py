import os
from fastapi import HTTPException, UploadFile
import re

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Validate size
    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(413, "File too large")
    # Sanitize filename
    filename = re.sub(r'[^a-zA-Z0-9._-]', '', file.filename)
    # ...
