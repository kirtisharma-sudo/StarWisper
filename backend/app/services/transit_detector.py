import numpy as np
from lightkurve import LightCurve as LKLightCurve
from ..processing import transit_detection, phase_fold
from ..utils.logging import get_logger

logger = get_logger(__name__)

class TransitDetector:
    @staticmethod
    def detect(time: np.ndarray, flux: np.ndarray) -> dict:
        result = transit_detection(time, flux)
        # Add bootstrap confidence intervals
        periods = np.linspace(0.5, 20, 1000)
        n_bootstrap = 100
        periods_boot = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(time), len(time), replace=True)
            try:
                lc_boot = LKLightCurve(time=time[idx], flux=flux[idx])
                bls_boot = lc_boot.to_periodogram(method='bls', period=periods, duration=0.1)
                peak = bls_boot.get_highest_peak()
                if peak is not None:
                    periods_boot.append(peak.period.value)
            except:
                continue
        if periods_boot:
            ci_lower = np.percentile(periods_boot, 2.5)
            ci_upper = np.percentile(periods_boot, 97.5)
        else:
            ci_lower = result['period'] - 0.01
            ci_upper = result['period'] + 0.01
        result['confidence_interval_lower'] = float(ci_lower)
        result['confidence_interval_upper'] = float(ci_upper)
        return result

    @staticmethod
    def phase_fold(time: np.ndarray, flux: np.ndarray, period: float, epoch: float) -> dict:
        phase, folded_flux = phase_fold(time, flux, period, epoch)
        return {
            "phase": phase.tolist(),
            "flux": folded_flux.tolist(),
            "period": period,
            "epoch": epoch
        }

    @staticmethod
    def load_transit_result(result_path: str) -> dict:
        data = np.load(result_path, allow_pickle=True).item()
        return data
