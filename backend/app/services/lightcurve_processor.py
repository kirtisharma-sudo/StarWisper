import numpy as np
from scipy import signal
from ..processing import sigma_clip, savitzky_golay, normalize
from ..utils.logging import get_logger

logger = get_logger(__name__)

class LightCurveProcessor:
    @staticmethod
    def process(time: np.ndarray, flux: np.ndarray, quality: np.ndarray = None) -> dict:
        mask = np.isfinite(time) & np.isfinite(flux)
        if quality is not None:
            mask &= (quality == 0)
        time = time[mask]
        flux = flux[mask]
        if len(time) == 0:
            raise ValueError("No good data points after quality filtering")
        flux_norm = normalize(flux, method='median')
        window = min(201, len(flux_norm)//10)
        if window % 2 == 0:
            window += 1
        if window > 5:
            detrended = savitzky_golay(flux_norm, window=window, order=3)
            flux_detrend = flux_norm - detrended + 1.0
        else:
            flux_detrend = flux_norm
        mask_clip, flux_clip = sigma_clip(flux_detrend, sigma=5.0, max_iter=3)
        time_clip = time[mask_clip]
        return {
            "time": time_clip,
            "flux": flux_clip,
            "detrended_flux": flux_clip,
            "quality_mask": mask_clip,
            "metadata": {
                "n_points": len(time_clip),
                "time_min": float(time_clip.min()),
                "time_max": float(time_clip.max())
            }
        }
