from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from typing import List
import uuid
import os
import shutil
import re
import json
import numpy as np
from datetime import datetime
from ..models.schemas import (
    UploadResponse, ProcessingStatus, ProcessedData, TransitParameters,
    FeatureVector, ClassificationResult, ValidationMetrics, AudioAnalysis,
    ReportRequest, ReportResponse, LightCurveMetadata, PhaseFoldData,
    ProcessingStatusResponse
)
from ..services.file_ingestion import FileIngestionService
from ..services.lightcurve_processor import LightCurveProcessor
from ..services.transit_detector import TransitDetector
from ..services.feature_extractor import FeatureExtractor
from ..services.classifier import ClassifierService
from ..services.audio_generator import AudioGenerator
from ..services.report_generator import ReportGenerator
from ..services.validation import ValidationService
from ..celery_worker import process_lightcurve_task, generate_report_task
from ..utils.logging import get_logger
from ..config import settings

router = APIRouter()
logger = get_logger(__name__)

UPLOAD_DIR = "/data/uploads"
RESULTS_DIR = "/data/results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(413, "File too large")
    filename = re.sub(r'[^a-zA-Z0-9._-]', '', file.filename)
    if not filename:
        filename = "upload"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".fits", ".fit", ".csv", ".txt"]:
        raise HTTPException(400, "Unsupported file format. Use FITS or CSV.")
    file_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    with open(save_path, "wb") as buffer:
        buffer.write(content)
    try:
        metadata = FileIngestionService.extract_metadata(save_path)
    except Exception as e:
        os.remove(save_path)
        raise HTTPException(400, f"Invalid file: {str(e)}")
    process_lightcurve_task.delay(file_id, save_path)
    return UploadResponse(
        file_id=file_id,
        filename=filename,
        status=ProcessingStatus.PENDING,
        message="File uploaded. Processing started."
    )

@router.get("/analysis/{file_id}", response_model=ProcessedData)
async def get_analysis(file_id: str):
    result_path = os.path.join(RESULTS_DIR, f"{file_id}_processed.npz")
    if not os.path.exists(result_path):
        raise HTTPException(404, "Processing not completed or file not found")
    data = FileIngestionService.load_processed_result(result_path)
    return data

@router.get("/transit/{file_id}", response_model=TransitParameters)
async def get_transit_parameters(file_id: str):
    result_path = os.path.join(RESULTS_DIR, f"{file_id}_transit.npz")
    if not os.path.exists(result_path):
        raise HTTPException(404, "Transit search not completed")
    params = TransitDetector.load_transit_result(result_path)
    return params

@router.get("/phase/{file_id}", response_model=PhaseFoldData)
async def get_phase_fold(file_id: str):
    result_path = os.path.join(RESULTS_DIR, f"{file_id}_phase.npz")
    if not os.path.exists(result_path):
        raise HTTPException(404, "Phase fold not computed")
    data = np.load(result_path, allow_pickle=True)
    return PhaseFoldData(
        phase=data['phase'].tolist(),
        flux=data['flux'].tolist(),
        period=float(data['period']),
        epoch=float(data['epoch'])
    )

@router.get("/features/{file_id}", response_model=FeatureVector)
async def get_features(file_id: str):
    result_path = os.path.join(RESULTS_DIR, f"{file_id}_features.npz")
    if not os.path.exists(result_path):
        raise HTTPException(404, "Feature extraction not completed")
    features = FeatureExtractor.load_features(result_path)
    return features

@router.get("/classification/{file_id}", response_model=ClassificationResult)
async def get_classification(file_id: str):
    result_path = os.path.join(RESULTS_DIR, f"{file_id}_classification.json")
    if not os.path.exists(result_path):
        raise HTTPException(404, "Classification not completed")
    return ClassifierService.load_classification(result_path)

@router.get("/validation/{file_id}", response_model=ValidationMetrics)
async def get_validation(file_id: str):
    # Try to get TIC from metadata
    meta_path = os.path.join(RESULTS_DIR, f"{file_id}_processed.npz")
    tic_id = None
    if os.path.exists(meta_path):
        data = np.load(meta_path, allow_pickle=True)
        meta = data.get('metadata', None)
        if meta is not None and isinstance(meta, np.ndarray):
            meta = meta.item()
        if meta and hasattr(meta, 'get'):
            tic_id = meta.get('tic')
    result = await ValidationService.compute_metrics(file_id, tic_id)
    return result

@router.get("/audio/{file_id}", response_model=AudioAnalysis)
async def get_audio(file_id: str):
    audio_path = os.path.join(RESULTS_DIR, f"{file_id}_audio.wav")
    if not os.path.exists(audio_path):
        # Generate if not exists
        processed = FileIngestionService.load_processed_result(
            os.path.join(RESULTS_DIR, f"{file_id}_processed.npz")
        )
        # Load transit params for highlighting
        transit_path = os.path.join(RESULTS_DIR, f"{file_id}_transit.npz")
        transit_params = None
        if os.path.exists(transit_path):
            transit_params = TransitDetector.load_transit_result(transit_path)
        audio_result = AudioGenerator.generate_sonification(
            processed.dict() if hasattr(processed, 'dict') else processed,
            transit_params
        )
        AudioGenerator.save_audio(audio_result, audio_path)
    analysis = AudioGenerator.analyze_audio(audio_path)
    return analysis

@router.post("/report/{file_id}", response_model=ReportResponse)
async def generate_report(file_id: str, request: ReportRequest):
    report_path = os.path.join(RESULTS_DIR, f"{file_id}_report.pdf")
    if not os.path.exists(report_path):
        generator = ReportGenerator()
        generator.generate(file_id, request.dict(), report_path)
    return ReportResponse(
        report_url=f"/api/v1/report/download/{file_id}",
        file_size=os.path.getsize(report_path),
        created_at=datetime.now()
    )

@router.get("/report/download/{file_id}")
async def download_report(file_id: str):
    report_path = os.path.join(RESULTS_DIR, f"{file_id}_report.pdf")
    if not os.path.exists(report_path):
        raise HTTPException(404, "Report not found")
    return FileResponse(report_path, media_type="application/pdf", filename=f"starwhisper_report_{file_id}.pdf")

@router.get("/status/{file_id}", response_model=ProcessingStatusResponse)
async def get_processing_status(file_id: str):
    processed_path = os.path.join(RESULTS_DIR, f"{file_id}_processed.npz")
    if os.path.exists(processed_path):
        return ProcessingStatusResponse(
            file_id=file_id,
            status=ProcessingStatus.COMPLETED,
            progress=1.0,
            message="Processing complete"
        )
    return ProcessingStatusResponse(
        file_id=file_id,
        status=ProcessingStatus.PENDING,
        progress=0.0,
        message="Queued for processing"
    )
