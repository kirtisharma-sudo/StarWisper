import numpy as np
import os
from ..audio import AudioEngine
from ..models.schemas import AudioAnalysis
from ..utils.logging import get_logger
from ..audio.export import AudioExporter
import librosa
from scipy.signal import spectrogram

logger = get_logger(__name__)

class AudioGenerator:
    @staticmethod
    def generate_sonification(processed_data: dict, transit_params: dict = None) -> dict:
        time = np.array(processed_data["time"])
        flux = np.array(processed_data["detrended_flux"] or processed_data["flux"])
        period = transit_params.get('period') if transit_params else None
        epoch = transit_params.get('epoch') if transit_params else None
        engine = AudioEngine(
            sample_rate=44100,
            duration_scale=60.0,
            pitch_mapper='linear',
            amplitude_mapper='linear',
            noise_mapper='pink',
            stereo=True,
            highlight_transits=True if period else False
        )
        transit_mask = None
        if period is not None and epoch is not None:
            phase = ((time - epoch) % period) / period
            phase = np.where(phase > 0.5, phase - 1.0, phase)
            dur_half = 0.05 * period
            transit_mask = np.abs(phase) < dur_half / period
        result = engine.generate(time, flux, transit_mask, period, epoch)
        return result

    @staticmethod
    def save_audio(audio_data: dict, path: str):
        engine = AudioEngine()
        engine.save(audio_data, path)

    @staticmethod
    def analyze_audio(audio_path: str) -> AudioAnalysis:
        audio, sr = librosa.load(audio_path, sr=None)
        duration = len(audio) / sr
        f, t, Sxx = spectrogram(audio, sr, nperseg=256)
        mean_spectrum = np.mean(Sxx, axis=1)
        peaks = librosa.util.peak_pick(mean_spectrum, 3, 3, 3, 5, 0.1, 0.5)
        freq_peaks = f[peaks].tolist()[:5] if len(peaks) > 0 else [440.0]
        envelope = np.abs(librosa.stft(audio))
        onset_frames = librosa.onset.onset_detect(onset_envelope=envelope, sr=sr)
        if len(onset_frames) > 1:
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            intervals = np.diff(onset_times)
            transit_rhythm = np.median(intervals)
        else:
            transit_rhythm = 1.0
        noise_floor = float(np.median(np.abs(audio)))
        return AudioAnalysis(
            waveform_url=f"/audio/{os.path.basename(audio_path)}",
            duration=duration,
            sample_rate=sr,
            frequency_peaks=freq_peaks,
            transit_rhythm=transit_rhythm,
            noise_floor=noise_floor
        )
