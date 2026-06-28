import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import io
import base64
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class WaveformGenerator:
    @staticmethod
    def generate_image(audio: np.ndarray, sample_rate: int, width: int = 800, height: int = 200, format: str = 'png', color: str = '#64b5f6', bgcolor: str = '#1a1e24') -> str:
        duration = len(audio) / sample_rate
        max_samples = 2000
        if len(audio) > max_samples:
            step = max(1, len(audio) // max_samples)
            audio_display = audio[::step]
        else:
            audio_display = audio
        t = np.linspace(0, duration, len(audio_display))
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        ax.fill_between(t, 0, audio_display, color=color, alpha=0.6)
        ax.plot(t, audio_display, color=color, linewidth=0.5, alpha=0.8)
        ax.set_xlim(0, duration)
        ax.set_ylim(-1.1, 1.1)
        ax.set_facecolor(bgcolor)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis='both', colors='#3a424c')
        ax.set_xlabel('Time (s)', color='#3a424c')
        ax.set_ylabel('Amplitude', color='#3a424c')
        buf = io.BytesIO()
        plt.savefig(buf, format=format, bbox_inches='tight', facecolor=bgcolor)
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        return f'data:image/{format};base64,{img_b64}'

    @staticmethod
    def generate_waveform_data(audio: np.ndarray, sample_rate: int, target_points: int = 1000) -> Dict[str, Any]:
        duration = len(audio) / sample_rate
        if len(audio) > target_points:
            step = max(1, len(audio) // target_points)
            audio_down = audio[::step]
            time = np.linspace(0, duration, len(audio_down))
        else:
            audio_down = audio
            time = np.linspace(0, duration, len(audio))
        return {
            'time': time.tolist(),
            'amplitude': audio_down.tolist(),
        }

class AudioExporter:
    @staticmethod
    def save_wav(audio: np.ndarray, sample_rate: int, path: str):
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
        sf.write(path, audio, sample_rate)
        logger.info(f"Audio saved to {path}")
