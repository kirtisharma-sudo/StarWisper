import numpy as np

class PitchMapper:
    @staticmethod
    def linear(flux: np.ndarray, min_freq: float = 200.0, max_freq: float = 800.0, baseline: float = 1.0) -> np.ndarray:
        delta = np.max(np.abs(flux - baseline))
        if delta == 0:
            return np.full_like(flux, (min_freq + max_freq) / 2)
        norm = (flux - (baseline - delta)) / (2 * delta)
        return min_freq + norm * (max_freq - min_freq)

    @staticmethod
    def logarithmic(flux: np.ndarray, min_freq: float = 100.0, max_freq: float = 1000.0, baseline: float = 1.0) -> np.ndarray:
        flux_safe = np.clip(flux, 1e-6, None)
        log_flux = np.log(flux_safe / baseline)
        if np.max(np.abs(log_flux)) == 0:
            return np.full_like(flux, (min_freq + max_freq) / 2)
        norm = (log_flux - np.min(log_flux)) / (np.max(log_flux) - np.min(log_flux))
        return min_freq + norm * (max_freq - min_freq)

    @staticmethod
    def exponential(flux: np.ndarray, min_freq: float = 200.0, max_freq: float = 800.0, base: float = 1.0) -> np.ndarray:
        flux_norm = (flux - np.min(flux)) / (np.max(flux) - np.min(flux) + 1e-12)
        exp_val = np.exp(flux_norm * np.log(max_freq / min_freq))
        return min_freq * exp_val

class AmplitudeMapper:
    @staticmethod
    def linear(flux: np.ndarray, min_amp: float = 0.1, max_amp: float = 1.0, baseline: float = 1.0) -> np.ndarray:
        delta = np.max(np.abs(flux - baseline))
        if delta == 0:
            return np.full_like(flux, (min_amp + max_amp) / 2)
        norm = (flux - (baseline - delta)) / (2 * delta)
        return min_amp + norm * (max_amp - min_amp)

    @staticmethod
    def logarithmic(flux: np.ndarray, min_amp: float = 0.05, max_amp: float = 0.9) -> np.ndarray:
        flux_safe = np.clip(flux, 1e-6, None)
        log_flux = np.log(flux_safe)
        if np.max(log_flux) == np.min(log_flux):
            return np.full_like(flux, (min_amp + max_amp) / 2)
        norm = (log_flux - np.min(log_flux)) / (np.max(log_flux) - np.min(log_flux))
        return min_amp + norm * (max_amp - min_amp)

    @staticmethod
    def threshold(flux: np.ndarray, threshold: float = 0.95, min_amp: float = 0.0, max_amp: float = 1.0) -> np.ndarray:
        amp = np.zeros_like(flux)
        mask = flux > threshold
        if np.any(mask):
            norm = (flux[mask] - threshold) / (np.max(flux) - threshold + 1e-12)
            amp[mask] = min_amp + norm * (max_amp - min_amp)
        return amp

class NoiseMapper:
    @staticmethod
    def pink_noise(time: np.ndarray, flux: np.ndarray, amplitude: float = 0.02, sample_rate: int = 44100) -> np.ndarray:
        def _pink_noise(n: int) -> np.ndarray:
            out = np.zeros(n)
            for i in range(6):
                stride = 2**i
                white = np.random.randn(n)
                window = np.ones(stride) / stride
                smoothed = np.convolve(white, window, mode='same')
                out += smoothed / (2**i)
            return out / np.std(out) * 0.5

        audio_time = np.linspace(0, time[-1]-time[0], int(sample_rate * (time[-1]-time[0])))
        if len(audio_time) > 1:
            from scipy.interpolate import interp1d
            interp = interp1d(time, flux, kind='linear', fill_value='extrapolate')
            flux_audio = interp(audio_time)
        else:
            flux_audio = np.full(len(audio_time), np.mean(flux))
        amp = 0.5 + 0.5 * ((flux_audio - np.min(flux_audio)) / (np.max(flux_audio) - np.min(flux_audio) + 1e-12))
        noise = _pink_noise(len(amp))
        return noise * amp * amplitude

    @staticmethod
    def white_noise(flux: np.ndarray, amplitude: float = 0.01) -> np.ndarray:
        noise = np.random.randn(len(flux))
        flux_norm = (flux - np.min(flux)) / (np.max(flux) - np.min(flux) + 1e-12)
        return noise * flux_norm * amplitude
