from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np

@dataclass
class LightCurve:
    time: np.ndarray
    flux: np.ndarray
    flux_err: Optional[np.ndarray] = None
    quality: Optional[np.ndarray] = None
    target: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    tic: Optional[int] = None
    sector: Optional[int] = None

@dataclass
class TransitSignal:
    period: float
    duration: float
    depth: float
    epoch: float
    snr: float
    chi2: float
    bls_power: float

@dataclass
class MLFeatures:
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
