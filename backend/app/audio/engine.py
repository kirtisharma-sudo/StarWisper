import numpy as np
from typing import Optional, Dict, Any
from scipy import signal
import logging
from .mappings import PitchMapper, AmplitudeMapper, NoiseMapper
from .effects import TransitHighlighter, StereoProcessor
from .export import WaveformGenerator, AudioExporter

logger = logging.getLogger(__name__)

class AudioEngine:
    def __init__(
        self,
        sample_rate: int = 44100,
        duration_scale: float = 60.0,
        pitch_mapper: str = 'linear',
        amplitude_mapper: str = 'linear',
        noise_mapper: str = 'pink',
        stereo: bool = True,
        highlight_transits: bool = True,
    ):
        self.sample_rate = sample_rate
        self.duration_scale = duration_scale
        self.pitch_mapper = pitch_mapper
        self.amplitude_mapper = amplitude_mapper
        self.noise_mapper = noise_mapper
        self.stereo = stereo
        self.highlight_transits = highlight_transits

    def generate(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        transit_mask: Optional[np.ndarray] = None,
        period: Optional[float] = None,
        epoch: Optional[float] = None
    ) -> Dict[str, Any]:
        time = np.asarray(time)
        flux = np.asarray(flux)
        if len(time) == 0:
            raise ValueError("Empty time array")
        flux_norm = flux / np.median(flux)
        time_range = time[-1] - time[0]
        if time_range <= 0:
            audio_duration = 5.0
        else:
            audio_duration = min(30.0, self.duration_scale * time_range)
        if audio_duration < 1:
            audio_duration = 1.0
        n_samples = int(self.sample_rate * audio_duration)
        audio_time = np.linspace(0, audio_duration, n_samples)
        from scipy.interpolate import interp1d
        if time_range > 0:
            interp_func = interp1d(time, flux_norm, kind='linear', fill_value='extrapolate')
            audio_t_mapped = time[0] + (time[-1] - time[0]) * audio_time / audio_duration
            flux_audio = interp_func(audio_t_mapped)
        else:
            flux_audio = np.full(n_samples, np.median(flux_norm))

        if self.pitch_mapper == 'linear':
            freq = PitchMapper.linear(flux_audio, min_freq=200, max_freq=800, baseline=1.0)
        elif self.pitch_mapper == 'log':
            freq = PitchMapper.logarithmic(flux_audio, min_freq=100, max_freq=1000, baseline=1.0)
        elif self.pitch_mapper == 'exp':
            freq = PitchMapper.exponential(flux_audio, min_freq=200, max_freq=800)
        else:
            freq = PitchMapper.linear(flux_audio)

        if self.amplitude_mapper == 'linear':
            amp = AmplitudeMapper.linear(flux_audio, min_amp=0.1, max_amp=1.0, baseline=1.0)
        elif self.amplitude_mapper == 'log':
            amp = AmplitudeMapper.logarithmic(flux_audio, min_amp=0.05, max_amp=0.9)
        elif self.amplitude_mapper == 'threshold':
            amp = AmplitudeMapper.threshold(flux_audio, threshold=0.95)
        else:
            amp = AmplitudeMapper.linear(flux_audio)

        phase = np.cumsum(2 * np.pi * freq / self.sample_rate)
        audio = np.sin(phase) * amp

        if self.noise_mapper == 'pink':
            noise = NoiseMapper.pink_noise(time, flux_norm, amplitude=0.01, sample_rate=self.sample_rate)
            if len(noise) > len(audio):
                noise = noise[:len(audio)]
            elif len(noise) < len(audio):
                noise = np.pad(noise, (0, len(audio)-len(noise)), mode='wrap')
            audio += noise
        elif self.noise_mapper == 'white':
            noise = NoiseMapper.white_noise(flux_audio, amplitude=0.005)
            audio += noise

        if self.highlight_transits and transit_mask is not None and np.any(transit_mask):
            if period is not None and epoch is not None:
                phase_audio = ((audio_t_mapped - epoch) % period) / period
                phase_audio = np.where(phase_audio > 0.5, phase_audio - 1.0, phase_audio)
                dur_half = 0.05 * period
                transit_mask_audio = np.abs(phase_audio) < dur_half / period
                audio = TransitHighlighter.amplitude_boost(audio, transit_mask_audio, boost_factor=2.0, fade=10)
            else:
                from scipy.interpolate import interp1d
                interp_mask = interp1d(time, transit_mask.astype(float), kind='nearest', fill_value='extrapolate')
                mask_audio_float = interp_mask(audio_t_mapped)
                transit_mask_audio = mask_audio_float > 0.5
                audio = TransitHighlighter.amplitude_boost(audio, transit_mask_audio, boost_factor=2.0, fade=10)

        if self.stereo:
            audio = StereoProcessor.pan_by_flux(audio, flux_audio, sample_rate=self.sample_rate)
        else:
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

        max_val = np.max(np.abs(audio))
        if max_val > 0.5:
            audio = audio / max_val * 0.5

        waveform_image = WaveformGenerator.generate_image(
            audio if audio.ndim == 1 else np.mean(audio, axis=1),
            self.sample_rate
        )
        waveform_data = WaveformGenerator.generate_waveform_data(
            audio if audio.ndim == 1 else np.mean(audio, axis=1),
            self.sample_rate
        )

        return {
            'audio': audio,
            'sample_rate': self.sample_rate,
            'waveform_image': waveform_image,
            'waveform_data': waveform_data,
            'duration': audio_duration,
        }

    def save(self, result: Dict[str, Any], path: str):
        audio = result['audio']
        sample_rate = result['sample_rate']
        AudioExporter.save_wav(audio, sample_rate, path)
