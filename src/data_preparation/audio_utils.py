import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import noisereduce as nr
from scipy import signal

class AudioValidator:
    def __init__(self, audio_config: Dict[str, Any]):
        self.sample_rate = audio_config.get("sample_rate", 16000)
        self.max_duration = audio_config.get("duration_range", [1.0, 10.0])[1]
        self.min_duration = audio_config.get("duration_range", [1.0, 10.0])[0]
        
    def validate_audio(self, audio_path: str) -> Tuple[bool, float]:
        """Validar archivo de audio"""
        try:
            # Verificar que existe
            if not Path(audio_path).exists():
                return False, 0.0
                
            # Cargar audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            duration = len(audio) / sr
            
            # Validar duración
            if duration < self.min_duration or duration > self.max_duration:
                return False, duration
                
            # Validar que no esté silencioso
            rms = np.sqrt(np.mean(audio**2))
            if rms < 0.001:  # Muy silencioso
                return False, duration
                
            # Validar que no sea todo NaN o Inf
            if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
                return False, duration
                
            return True, duration
            
        except Exception as e:
            print(f"Error validando {audio_path}: {e}")
            return False, 0.0

class AudioProcessor:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def load_audio(self, audio_path: str) -> np.ndarray:
        """Cargar audio y normalizar sample rate"""
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        return audio
        
    def save_audio(self, audio: np.ndarray, output_path: str):
        """Guardar audio como WAV"""
        sf.write(output_path, audio, self.sample_rate)
        
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalizar audio a rango [-1, 1]"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio
        
    def trim_silence(self, audio: np.ndarray, top_db: int = 20) -> np.ndarray:
        """Recortar silencios del audio"""
        return librosa.effects.trim(audio, top_db=top_db)[0]
        
    def reduce_noise(self, audio: np.ndarray, noise_sample: Optional[np.ndarray] = None) -> np.ndarray:
        """Reducir ruido usando noisereduce"""
        if noise_sample is not None:
            return nr.reduce_noise(y=audio, y_noise=noise_sample, sr=self.sample_rate)
        else:
            return nr.reduce_noise(y=audio, sr=self.sample_rate)
            
    def compute_audio_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Calcular características del audio"""
        features = {}
        
        # RMS (volumen)
        features['rms'] = np.sqrt(np.mean(audio**2))
        
        # Energía
        features['energy'] = np.sum(audio**2)
        
        # Zero-crossing rate
        features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(audio))
        
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        features['spectral_centroid'] = np.mean(spectral_centroids)
        
        # Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)
        features['spectral_bandwidth'] = np.mean(spectral_bandwidth)
        
        return features

class AudioAugmentor:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def time_stretch(self, audio: np.ndarray, rate: float) -> np.ndarray:
        """Estirar/encoger tiempo sin afectar pitch"""
        return librosa.effects.time_stretch(audio, rate=rate)
        
    def pitch_shift(self, audio: np.ndarray, n_steps: float) -> np.ndarray:
        """Cambiar pitch sin afectar tempo"""
        return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)
        
    def add_gaussian_noise(self, audio: np.ndarray, snr_db: float = 20) -> np.ndarray:
        """Añadir ruido gaussiano con SNR específico"""
        rms_signal = np.sqrt(np.mean(audio**2))
        rms_noise = rms_signal / (10**(snr_db / 20))
        noise = np.random.normal(0, rms_noise, audio.shape)
        return audio + noise
        
    def apply_bandpass_filter(self, audio: np.ndarray, lowcut: float, highcut: float) -> np.ndarray:
        """Aplicar filtro paso banda"""
        nyquist = self.sample_rate / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        return signal.filtfilt(b, a, audio)