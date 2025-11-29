import torch
import numpy as np
from typing import Dict, List, Any, Optional
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from src.utils.logger import setup_logger
from src.data_preparation.audio_utils import AudioProcessor

class EmergencyTranscriber:
    def __init__(self, model_dir: str, languages_config: Dict[str, Any]):
        self.model_dir = model_dir
        self.languages_config = languages_config
        self.logger = setup_logger("transcriber")
        
        # Cargar modelo y processor
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.model = WhisperForConditionalGeneration.from_pretrained(model_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        
        self.audio_processor = AudioProcessor()
        
        # Compilar frases de emergencia por idioma
        self.emergency_phrases = {}
        for lang_code, lang_info in languages_config["languages"].items():
            self.emergency_phrases[lang_code] = lang_info.get("emergency_phrases", [])
    
    def transcribe(self, 
                  audio_path: str, 
                  language: Optional[str] = None,
                  detect_emergency: bool = True) -> Dict[str, Any]:
        """Transcribir audio y analizar contenido"""
        
        try:
            # Cargar y preprocesar audio
            audio = self.audio_processor.load_audio(audio_path)
            
            # Preparar inputs
            inputs = self.processor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Configurar generación forzando idioma si se especifica
            forced_decoder_ids = None
            if language and language in self.languages_config["languages"]:
                lang_code = self.languages_config["languages"][language]["whisper_code"]
                forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                    language=lang_code, 
                    task="transcribe"
                )
            
            # Generar transcripción
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    inputs.input_features,
                    forced_decoder_ids=forced_decoder_ids,
                    max_length=225,
                    num_beams=5,
                    temperature=0.0
                )
            
            transcription = self.processor.decode(predicted_ids[0], skip_special_tokens=True)
            
            # Post-procesamiento
            result = {
                "transcription": transcription.strip(),
                "detected_language": self._detect_language(transcription),
                "confidence": self._estimate_confidence(predicted_ids[0]),
                "audio_duration": len(audio) / 16000,
                "emergency_phrases": [],
                "is_emergency": False
            }
            
            # Detección de emergencia
            if detect_emergency:
                emergency_info = self._detect_emergency_phrases(transcription)
                result["emergency_phrases"] = emergency_info["detected_phrases"]
                result["is_emergency"] = emergency_info["is_emergency"]
                result["emergency_confidence"] = emergency_info["confidence"]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error transcribiendo {audio_path}: {e}")
            return {
                "transcription": "",
                "error": str(e),
                "emergency_phrases": [],
                "is_emergency": False
            }
    
    def transcribe_batch(self, 
                        audio_paths: List[str], 
                        language: Optional[str] = None) -> List[Dict[str, Any]]:
        """Transcribir múltiples audios en lote"""
        results = []
        
        for audio_path in audio_paths:
            result = self.transcribe(audio_path, language)
            results.append(result)
            
        return results
    
    def _detect_language(self, transcription: str) -> str:
        """Detectar idioma basado en la transcripción"""
        transcription_lower = transcription.lower()
        
        for lang_code, lang_info in self.languages_config["languages"].items():
            # Buscar frases características del idioma
            for phrase in lang_info.get("emergency_phrases", []):
                if phrase.lower() in transcription_lower:
                    return lang_code
        
        return "unknown"
    
    def _detect_emergency_phrases(self, transcription: str) -> Dict[str, Any]:
        """Detectar frases de emergencia en la transcripción"""
        transcription_lower = transcription.lower()
        detected_phrases = []
        
        # Buscar en todos los idiomas
        for lang_code, phrases in self.emergency_phrases.items():
            for phrase in phrases:
                phrase_lower = phrase.lower()
                if phrase_lower in transcription_lower:
                    detected_phrases.append({
                        "phrase": phrase,
                        "language": lang_code,
                        "position": transcription_lower.find(phrase_lower)
                    })
        
        # Calcular confianza de emergencia
        confidence = min(1.0, len(detected_phrases) * 0.3)  # Máximo 1.0
        
        return {
            "detected_phrases": detected_phrases,
            "is_emergency": len(detected_phrases) > 0,
            "confidence": confidence
        }
    
    def _estimate_confidence(self, token_ids: torch.Tensor) -> float:
        """Estimar confianza de la transcripción (simplificado)"""
        # En una implementación real, podrías usar las probabilidades del modelo
        # Esta es una aproximación simplificada
        unique_tokens = len(set(token_ids.cpu().numpy()))
        total_tokens = len(token_ids)
        
        if total_tokens == 0:
            return 0.0
            
        # Ratio de tokens únicos (proxy de confianza)
        diversity_ratio = unique_tokens / total_tokens
        confidence = min(1.0, diversity_ratio * 1.5)  # Ajustar según necesidad
        
        return confidence
    
    def real_time_transcription(self, audio_chunk: np.ndarray) -> Dict[str, Any]:
        """Transcripción en tiempo real para streaming"""
        try:
            # Asegurar que el audio esté en el formato correcto
            if len(audio_chunk.shape) > 1:
                audio_chunk = np.mean(audio_chunk, axis=1)  # Convertir a mono
            
            # Normalizar
            audio_chunk = audio_chunk / np.max(np.abs(audio_chunk)) if np.max(np.abs(audio_chunk)) > 0 else audio_chunk
            
            # Procesar
            inputs = self.processor(
                audio_chunk, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    inputs.input_features,
                    max_length=100,  # Más corto para tiempo real
                    num_beams=3,
                    temperature=0.0
                )
            
            transcription = self.processor.decode(predicted_ids[0], skip_special_tokens=True)
            
            # Detección rápida de emergencia
            emergency_info = self._detect_emergency_phrases(transcription)
            
            return {
                "transcription": transcription,
                "is_emergency": emergency_info["is_emergency"],
                "emergency_phrases": emergency_info["detected_phrases"],
                "timestamp": len(audio_chunk) / 16000
            }
            
        except Exception as e:
            self.logger.error(f"Error en transcripción tiempo real: {e}")
            return {
                "transcription": "",
                "is_emergency": False,
                "emergency_phrases": [],
                "error": str(e)
            }