import numpy as np

class TransitHighlighter:
    @staticmethod
    def amplitude_boost(audio: np.ndarray, transit_mask: np.ndarray, boost_factor: float = 2.0, fade: int = 0) -> np.ndarray:
        audio_out = audio.copy()
        if fade > 0:
            from scipy.ndimage import gaussian_filter1d
            mask = transit_mask.astype(float)
            mask_smooth = gaussian_filter1d(mask, sigma=fade/2)
            audio_out *= 1.0 + (boost_factor - 1.0) * mask_smooth
        else:
            audio_out[transit_mask] *= boost_factor
        return audio_out

class StereoProcessor:
    @staticmethod
    def pan_by_flux(audio: np.ndarray, flux: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        if len(flux) != len(audio):
            from scipy.interpolate import interp1d
            t_audio = np.linspace(0, len(audio)/sample_rate, len(audio))
            if len(flux) > 1:
                t_flux = np.linspace(0, t_audio[-1], len(flux))
                interp = interp1d(t_flux, flux, kind='linear', fill_value='extrapolate')
                flux_resampled = interp(t_audio)
            else:
                flux_resampled = np.full(len(audio), flux[0])
        else:
            flux_resampled = flux
        flux_norm = 2 * (flux_resampled - np.min(flux_resampled)) / (np.max(flux_resampled) - np.min(flux_resampled) + 1e-12) - 1
        pan = flux_norm
        left = audio * (1 - pan) / 2
        right = audio * (1 + pan) / 2
        return np.column_stack((left, right))
