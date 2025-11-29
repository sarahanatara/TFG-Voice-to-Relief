import numpy as np
import librosa
import random
from pathlib import Path
from typing import List, Dict, Any
import soundfile as sf

class EmergencyAugmentor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sr = config["audio"]["sample_rate"]
        self.noise_files = self._load_noise_files()
        
    def _load_noise_files(self) -> Dict[str, List[str]]:
        """Cargar y categorizar archivos de ruido"""
        noise_dir = Path("data/noise")
        noise_files = {}
        
        for category_dir in noise_dir.iterdir():
            if category_dir.is_dir():
                category = category_dir.name
                noise_files[category] = [
                    str(f) for f in category_dir.glob("*.wav")
                ]
                
        return noise_files
    
    def _apply_telephone_effect(self, audio: np.ndarray) -> np.ndarray:
        """Efecto de calidad telefónica (banda limitada)"""
        # Filtro paso banda 300-3400 Hz (telefono tradicional)
        from scipy.signal import butter, filtfilt
        nyquist = self.sr / 2
        low = 300 / nyquist
        high = 3400 / nyquist
        
        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, audio)
    
    def _apply_packet_loss(self, audio: np.ndarray, loss_prob: float = 0.05) -> np.ndarray:
        """Simular pérdida de paquetes en llamadas VoIP"""
        result = audio.copy()
        chunk_size = int(0.02 * self.sr)  # 20ms chunks
        
        for i in range(0, len(audio), chunk_size):
            if random.random() < loss_prob:
                end = min(i + chunk_size, len(audio))
                result[i:end] = 0  # Silenciar chunk
                
        return result
    
    def _select_emergency_scenario(self) -> Dict[str, Any]:
        """Seleccionar escenario de emergencia realista"""
        scenarios = self.config["augmentation"]["emergency_scenarios"]
        scenario_name = random.choice(list(scenarios.keys()))
        noise_types = scenarios[scenario_name]
        
        return {
            "name": scenario_name,
            "noise_types": noise_types,
            "dual_noise_prob": 0.3 if scenario_name in ["street_accident", "public_space"] else 0.1
        }
    
    def generate_variants(self, audio_path: str, transcription: str, 
                         language: str, speaker_id: str) -> List[Dict[str, Any]]:
        """Generar variantes aumentadas para un audio"""
        # Cargar audio limpio
        audio, _ = librosa.load(audio_path, sr=self.sr)
        
        variants = []
        n_variants = self.config["augmentation"]["n_variants_per_file"]
        
        for i in range(n_variants):
            # Seleccionar escenario
            scenario = self._select_emergency_scenario()
            
            # Aplicar augmentaciones
            augmented = self._apply_augmentation_chain(audio, scenario)
            
            # Guardar variante
            variant_info = self._save_variant(
                audio=augmented,
                base_name=f"{speaker_id}_{language}_{Path(audio_path).stem}",
                variant_id=i,
                transcription=transcription,
                language=language,
                scenario=scenario
            )
            variants.append(variant_info)
            
        return variants
    
    def _apply_augmentation_chain(self, audio: np.ndarray, scenario: Dict[str, Any]) -> np.ndarray:
        """Cadena completa de augmentación"""
        result = audio.copy()
        effects_config = self.config["augmentation"]["effects"]
        
        # 1. Mezclar ruidos según escenario
        result = self._mix_emergency_noises(result, scenario)
        
        # 2. Aplicar efectos acústicos
        if random.random() < effects_config["reverb_prob"]:
            result = self._apply_reverb(result)
            
        if random.random() < effects_config["pitch_shift_prob"]:
            result = librosa.effects.pitch_shift(result, sr=self.sr, n_steps=random.uniform(-1, 1))
            
        if random.random() < effects_config["telephone_effect_prob"]:
            result = self._apply_telephone_effect(result)
            
        if random.random() < effects_config["packet_loss_prob"]:
            result = self._apply_packet_loss(result)
            
        # 3. Normalización final
        result = self._normalize_audio(result)
        
        return result
    
    def _mix_emergency_noises(self, audio: np.ndarray, scenario: Dict[str, Any]) -> np.ndarray:
        """Mezclar ruidos de emergencia realistas"""
        # Implementación similar a tu versión pero mejorada
        # ... (código para mezcla de ruidos)
        return audio  # Placeholder
    
    def _save_variant(self, audio: np.ndarray, base_name: str, variant_id: int,
                     transcription: str, language: str, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Guardar variante y retornar metadata"""
        # ... implementación para guardar archivos
        return {}  # Placeholder