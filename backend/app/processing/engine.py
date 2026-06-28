import numpy as np
from scipy import signal, stats
from scipy.interpolate import interp1d
from typing import Tuple, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

def normalize(flux: np.ndarray, method: str = 'median') -> np.ndarray:
    flux = np.asarray(flux)
    if method == 'median':
        baseline = np.median(flux)
    elif method == 'mean':
        baseline = np.mean(flux)
    else:
        raise ValueError("method must be 'median' or 'mean'")
    if baseline == 0:
        raise ZeroDivisionError("Flux baseline is zero; cannot normalize.")
    return flux / baseline

def savitzky_golay(y: np.ndarray, window: int = 11, order: int = 3, deriv: int = 0, delta: float = 1.0) -> np.ndarray:
    y = np.asarray(y)
    if window % 2 == 0:
        window += 1
    if window > len(y):
        window = len(y) if len(y) % 2 == 1 else len(y) - 1
    if window < 3:
        return y
    try:
        return signal.savgol_filter(y, window, order, deriv=deriv, delta=delta)
    except Exception as e:
        logger.warning(f"Savitzky‑Golay failed: {e}. Falling back to identity.")
        return y

def sigma_clip(flux: np.ndarray, sigma: float = 5.0, max_iter: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    flux = np.asarray(flux)
    mask = np.ones(flux.shape, dtype=bool)
    data = flux.flatten()
    for _ in range(max_iter):
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            break
        new_mask = np.abs(data - mean) < sigma * std
        if np.all(new_mask):
            break
        data = data[new_mask]
        new_flat = np.zeros(len(mask.flatten()), dtype=bool)
        new_flat[mask.flatten()] = new_mask
        mask = new_flat.reshape(mask.shape)
    return mask, flux[mask]

def phase_fold(time: np.ndarray, flux: np.ndarray, period: float, epoch: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    time = np.asarray(time)
    flux = np.asarray(flux)
    if period <= 0:
        raise ValueError("Period must be positive.")
    phase = ((time - epoch) % period) / period
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    idx = np.argsort(phase)
    return phase[idx], flux[idx]

def transit_detection(time: np.ndarray, flux: np.ndarray, period_grid: Optional[np.ndarray] = None, duration_grid: Optional[np.ndarray] = None) -> Dict[str, Union[float, np.ndarray]]:
    try:
        from lightkurve import LightCurve
        lc = LightCurve(time=time, flux=flux)
        if period_grid is None:
            period_grid = np.linspace(0.5, 20, 1000)
        if duration_grid is None:
            duration_grid = np.linspace(0.02, 0.2, 20)
        bls = lc.to_periodogram(method='bls', period=period_grid, duration=duration_grid)
        best = bls.get_highest_peak()
        if best is None:
            raise ValueError("No significant peak found.")
        period = best.period.value
        duration = best.duration.value
        depth = best.depth.value
        epoch = best.transit_time.value
        snr = getattr(best, 'snr', np.nan)
        power = best.power.value
        model = bls.get_transit_model(period=period, duration=duration, transit_time=epoch)
        return {
            'period': period,
            'duration': duration,
            'depth': depth,
            'epoch': epoch,
            'snr': snr,
            'power': power,
            'transit_model': model,
        }
    except ImportError:
        logger.warning("lightkurve not available. Using fallback BLS.")
        return _bls_fallback(time, flux, period_grid, duration_grid)

def _bls_fallback(time: np.ndarray, flux: np.ndarray, period_grid: Optional[np.ndarray] = None, duration_grid: Optional[np.ndarray] = None) -> Dict[str, Union[float, np.ndarray]]:
    flux_norm = flux / np.median(flux)
    if period_grid is None:
        period_grid = np.linspace(0.5, 20, 500)
    if duration_grid is None:
        duration_grid = np.linspace(0.02, 0.2, 10)
    best_power = -np.inf
    best_params = None
    t = time
    for p in period_grid:
        for d in duration_grid:
            if d >= p:
                continue
            phase = (t % p) / p
            in_transit = (phase < d/p) | (phase > 1 - d/p)
            if np.sum(in_transit) == 0:
                continue
            y_in = flux_norm[in_transit]
            y_out = flux_norm[~in_transit]
            if len(y_out) == 0:
                continue
            mean_in = np.mean(y_in)
            mean_out = np.mean(y_out)
            total = np.sum((flux_norm - np.mean(flux_norm))**2)
            residual = np.sum((flux_norm[in_transit] - mean_in)**2) + np.sum((flux_norm[~in_transit] - mean_out)**2)
            power = (total - residual) / total
            if power > best_power:
                best_power = power
                best_params = {
                    'period': p,
                    'duration': d,
                    'depth': (mean_out - mean_in) / mean_out,
                    'epoch': 0.0,
                    'snr': np.nan,
                    'power': power,
                }
    if best_params is None:
        raise ValueError("No transit detected.")
    model = np.ones_like(flux_norm)
    if best_params:
        p = best_params['period']
        d = best_params['duration']
        phase = (t % p) / p
        in_transit = (phase < d/p) | (phase > 1 - d/p)
        depth = best_params['depth']
        model[in_transit] = 1 - depth
    best_params['transit_model'] = model
    return best_params

def feature_extraction(time: np.ndarray, flux: np.ndarray, transit_params: Dict[str, float]) -> Dict[str, float]:
    period = transit_params['period']
    duration = transit_params['duration']
    depth = transit_params['depth']
    epoch = transit_params['epoch']
    snr = transit_params.get('snr', np.nan)
    chi2 = transit_params.get('chi2', np.nan)
    power = transit_params.get('power', np.nan)

    # Additional features
    dur_half = duration / 2
    phase = ((time - epoch) % period) / period
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    in_transit = np.abs(phase) < dur_half / period
    median_in = np.median(flux[in_transit]) if np.any(in_transit) else 1.0
    median_out = np.median(flux[~in_transit]) if np.any(~in_transit) else 1.0
    transit_signal_positive = (median_out - median_in) / median_out if median_out != 0 else 0

    n_transits = int(np.floor((time[-1] - epoch) / period))
    depths_even = []
    depths_odd = []
    for i in range(n_transits):
        t0 = epoch + i * period
        mask = np.abs((time - t0) % period - period/2) < dur_half
        if np.any(mask):
            med = np.median(flux[mask])
            depth_ppm = (1 - med) * 1e6
            if i % 2 == 0:
                depths_even.append(depth_ppm)
            else:
                depths_odd.append(depth_ppm)
    even_odd_ratio = np.mean(depths_even) / np.mean(depths_odd) if depths_odd else 1.0

    sec_epoch = epoch + period/2
    mask_sec = np.abs((time - sec_epoch) % period - period/2) < dur_half
    median_sec = np.median(flux[mask_sec]) if np.any(mask_sec) else 1.0
    secondary_eclipse_depth = (median_out - median_sec) / median_out if median_out != 0 else 0

    blend_prob = 0.1
    contamination = 0.05

    return {
        'period': period,
        'duration': duration * 24,
        'depth': depth * 1e6,
        'epoch': epoch,
        'snr': snr,
        'chi2': chi2,
        'bls_power': power,
        'transit_signal_positive': transit_signal_positive,
        'even_odd_ratio': even_odd_ratio,
        'secondary_eclipse_depth': secondary_eclipse_depth,
        'blend_probability': blend_prob,
        'contamination': contamination,
    }
