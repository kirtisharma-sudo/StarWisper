import numpy as np
from scipy import stats, signal
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def extract_morphological_features(time: np.ndarray, flux: np.ndarray, transit_mask: np.ndarray, period: float, duration: float, epoch: float) -> Dict[str, float]:
    in_idx = np.where(transit_mask)[0]
    if len(in_idx) < 3:
        return {
            'depth': 0.0,
            'width': duration,
            'ingress_duration': duration * 0.1,
            'egress_duration': duration * 0.1,
            'ingress_slope': 0.0,
            'egress_slope': 0.0,
            'asymmetry': 0.0,
            'residual_std': 0.0,
            'sharpness': 0.0,
        }
    flux_in = flux[transit_mask]
    flux_out = flux[~transit_mask]
    median_in = np.median(flux_in) if len(flux_in) > 0 else 1.0
    median_out = np.median(flux_out) if len(flux_out) > 0 else 1.0
    depth = (median_out - median_in) / median_out
    grad = np.gradient(flux)
    boundaries = np.diff(transit_mask.astype(int))
    ingress_idx = np.where(boundaries == 1)[0]
    egress_idx = np.where(boundaries == -1)[0]
    if len(ingress_idx) > 0 and len(egress_idx) > 0:
        ingress_start = ingress_idx[0]
        egress_start = egress_idx[0]
        ingress_length = min(10, egress_start - ingress_start)
        egress_length = min(10, len(flux) - egress_start)
        if ingress_length > 1:
            ingress_duration = time[ingress_start + ingress_length] - time[ingress_start]
            ingress_slope = (flux[ingress_start + ingress_length] - flux[ingress_start]) / ingress_duration
        else:
            ingress_duration = duration * 0.1
            ingress_slope = 0.0
        if egress_length > 1:
            egress_duration = time[egress_start + egress_length] - time[egress_start]
            egress_slope = (flux[egress_start + egress_length] - flux[egress_start]) / egress_duration
        else:
            egress_duration = duration * 0.1
            egress_slope = 0.0
        asymmetry = ingress_duration / (ingress_duration + egress_duration + 1e-12)
    else:
        ingress_duration = duration * 0.1
        egress_duration = duration * 0.1
        ingress_slope = 0.0
        egress_slope = 0.0
        asymmetry = 0.5
    res_in = flux_in - median_in
    res_out = flux_out - median_out
    residual_std = np.std(np.concatenate([res_in, res_out])) if len(res_in) + len(res_out) > 0 else 0.0
    if len(in_idx) > 5:
        mid_idx = in_idx[len(in_idx)//2]
        if mid_idx > 1 and mid_idx < len(flux)-1:
            second_deriv = flux[mid_idx+1] - 2*flux[mid_idx] + flux[mid_idx-1]
            sharpness = -second_deriv / (duration**2)
        else:
            sharpness = 0.0
    else:
        sharpness = 0.0
    return {
        'depth': depth,
        'width': duration,
        'ingress_duration': ingress_duration,
        'egress_duration': egress_duration,
        'ingress_slope': ingress_slope,
        'egress_slope': egress_slope,
        'asymmetry': asymmetry,
        'residual_std': residual_std,
        'sharpness': sharpness,
    }

def extract_statistical_features(flux: np.ndarray) -> Dict[str, float]:
    flux = np.asarray(flux)
    return {
        'mean': np.mean(flux),
        'median': np.median(flux),
        'std': np.std(flux),
        'skew': stats.skew(flux),
        'kurtosis': stats.kurtosis(flux),
        'q25': np.percentile(flux, 25),
        'q75': np.percentile(flux, 75),
        'iqr': np.percentile(flux, 75) - np.percentile(flux, 25),
        'min': np.min(flux),
        'max': np.max(flux),
    }

def extract_time_series_features(time: np.ndarray, flux: np.ndarray) -> Dict[str, float]:
    if len(flux) > 1:
        acf = np.correlate(flux - np.mean(flux), flux - np.mean(flux), mode='full')
        acf = acf / (np.var(flux) * len(flux))
        lag1 = acf[len(acf)//2 + 1] if len(acf) > 1 else 0.0
    else:
        lag1 = 0.0
    if len(flux) > 10:
        f, Pxx = signal.periodogram(flux, fs=1.0 / (time[1] - time[0] if len(time) > 1 else 1.0))
        peak_freq = f[np.argmax(Pxx)]
        peak_power = np.max(Pxx)
    else:
        peak_freq = 0.0
        peak_power = 0.0
    return {
        'autocorr_lag1': lag1,
        'power_peak_freq': peak_freq,
        'power_peak_amplitude': peak_power,
        'rms': np.sqrt(np.mean(flux**2)),
        'peak_to_peak': np.max(flux) - np.min(flux),
    }

def extract_all_features(time: np.ndarray, flux: np.ndarray, transit_mask: Optional[np.ndarray] = None, period: Optional[float] = None, duration: Optional[float] = None, epoch: Optional[float] = None) -> Dict[str, float]:
    features = {}
    features.update(extract_statistical_features(flux))
    features.update(extract_time_series_features(time, flux))
    if transit_mask is not None and period is not None and duration is not None and epoch is not None:
        morph = extract_morphological_features(time, flux, transit_mask, period, duration, epoch)
        features.update(morph)
    return features

def feature_vector_to_array(features: Dict[str, float], feature_names: List[str]) -> np.ndarray:
    return np.array([features.get(name, 0.0) for name in feature_names])
