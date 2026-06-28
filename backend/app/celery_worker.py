from celery import Celery
from .config import settings
import os
import numpy as np
import json
from .services.file_ingestion import FileIngestionService
from .services.lightcurve_processor import LightCurveProcessor
from .services.transit_detector import TransitDetector
from .services.feature_extractor import FeatureExtractor
from .services.classifier import ClassifierService
from .services.audio_generator import AudioGenerator
from .utils.logging import get_logger

logger = get_logger(__name__)

celery = Celery(
    "starwhisper",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)
celery.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_retry_delay=60,
    task_max_retries=3,
)

@celery.task(bind=True, max_retries=3)
def process_lightcurve_task(self, file_id: str, file_path: str):
    logger.info(f"Processing {file_id}")
    try:
        metadata = FileIngestionService.extract_metadata(file_path)
        # Load data
        if file_path.endswith('.fits'):
            from astropy.io import fits
            with fits.open(file_path) as hdul:
                if len(hdul) > 1:
                    data = hdul[1].data
                else:
                    data = hdul[0].data
                time = data[metadata.time_columns[0]]
                flux = data[metadata.flux_columns[0]]
                quality = data[metadata.quality_column] if metadata.quality_column else None
        else:
            import pandas as pd
            df = pd.read_csv(file_path)
            time = df[metadata.time_columns[0]].values
            flux = df[metadata.flux_columns[0]].values
            quality = df[metadata.quality_column].values if metadata.quality_column else None

        # Process
        processed = LightCurveProcessor.process(time, flux, quality)
        np.savez_compressed(f"/data/results/{file_id}_processed.npz", **processed)

        # Transit
        transit_params = TransitDetector.detect(processed['time'], processed['flux'])
        np.savez_compressed(f"/data/results/{file_id}_transit.npz", **transit_params)

        # Phase fold
        phase_data = TransitDetector.phase_fold(
            processed['time'], processed['flux'],
            transit_params['period'], transit_params['epoch']
        )
        np.savez_compressed(f"/data/results/{file_id}_phase.npz", **phase_data)

        # Features
        features = FeatureExtractor.extract(
            processed['time'], processed['flux'], transit_params
        )
        np.savez_compressed(f"/data/results/{file_id}_features.npz", **features.dict())

        # Classification
        cls_result = ClassifierService.classify(features)
        with open(f"/data/results/{file_id}_classification.json", 'w') as f:
            json.dump(cls_result.dict(), f)

        # Audio
        audio_result = AudioGenerator.generate_sonification(processed, transit_params)
        audio_path = f"/data/results/{file_id}_audio.wav"
        AudioGenerator.save_audio(audio_result, audio_path)

        logger.info(f"Processing {file_id} completed")
        return {"status": "completed", "file_id": file_id}
    except Exception as e:
        logger.error(f"Processing {file_id} failed: {e}", exc_info=True)
        self.retry(exc=e, countdown=60)

@celery.task(bind=True, max_retries=3)
def generate_report_task(self, file_id: str, request_dict: dict):
    from .services.report_generator import ReportGenerator
    output_path = f"/data/results/{file_id}_report.pdf"
    try:
        gen = ReportGenerator()
        gen.generate(file_id, request_dict, output_path)
        return {"status": "completed", "file_path": output_path}
    except Exception as e:
        logger.error(f"Report generation failed: {e}", exc_info=True)
        self.retry(exc=e, countdown=60)
