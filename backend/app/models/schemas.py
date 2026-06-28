from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class UploadResponse(BaseModel):
    file_id: str
    filename: str
    status: ProcessingStatus
    message: Optional[str] = None

class LightCurveMetadata(BaseModel):
    target: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    tic: Optional[int] = None
    sector: Optional[int] = None
    camera: Optional[int] = None
    ccd: Optional[int] = None
    time_columns: List[str]
    flux_columns: List[str]
    quality_column: Optional[str] = None
    n_points: int
    time_min: float
    time_max: float

class ProcessedData(BaseModel):
    time: List[float]
    flux: List[float]
    flux_err: Optional[List[float]] = None
    detrended_flux: Optional[List[float]] = None
    quality_mask: Optional[List[bool]] = None
    metadata: LightCurveMetadata

class TransitParameters(BaseModel):
    period: float
    duration: float
    depth: float
    epoch: float
    signal_to_noise: float
    reduced_chi2: float
    bls_power: float
    confidence_interval_lower: float
    confidence_interval_upper: float

class FeatureVector(BaseModel):
    period: float
    duration: float
    depth: float
    epoch: float
    snr: float
    chi2: float
    bls_power: float
    transit_signal_positive: float
    even_odd_ratio: float
    secondary_eclipse_depth: float
    blend_probability: float
    contamination: float

class ClassificationResult(BaseModel):
    predicted_label: str
    probability: float
    confidence: float
    feature_importance: Dict[str, float]
    interpretation: str

class ValidationMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: List[List[int]]
    roc_auc: float

class AudioAnalysis(BaseModel):
    waveform_url: str
    duration: float
    sample_rate: int
    frequency_peaks: List[float]
    transit_rhythm: float
    noise_floor: float

class PhaseFoldData(BaseModel):
    phase: List[float]
    flux: List[float]
    period: float
    epoch: float

class ProcessingStatusResponse(BaseModel):
    file_id: str
    status: ProcessingStatus
    progress: Optional[float] = None
    message: Optional[str] = None

class ReportRequest(BaseModel):
    include_plots: bool = True
    include_tables: bool = True
    include_interpretation: bool = True
    include_validation: bool = True
    format: str = "pdf"

class ReportResponse(BaseModel):
    report_url: str
    file_size: int
    created_at: datetime
