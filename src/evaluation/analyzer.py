import pandas as pd
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from jiwer import wer
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import numpy as np
from src.utils.logger import setup_logger
from src.data_preparation.audio_utils import AudioProcessor

class ComprehensiveEvaluator:
    def __init__(self, model_dir: str, languages_config: Dict[str, Any]):
        self.model_dir = model_dir
        self.languages_config = languages_config
        self.logger = setup_logger("evaluator")
        
        # Cargar modelo y processor
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.model = WhisperForConditionalGeneration.from_pretrained(model_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        self.audio_processor = AudioProcessor()
        
    def run_comprehensive_evaluation(self, test_metadata_path: str) -> Dict[str, Any]:
        """Ejecutar evaluación comprehensiva"""
        df_test = pd.read_csv(test_metadata_path)
        
        results = {
            "overall": self._evaluate_subset(df_test),
            "by_language": {},
            "by_noise_level": {},
            "by_scenario": {},
            "by_emergency_phrase": {}
        }
        
        # Evaluar por idioma
        for lang in df_test['language'].unique():
            lang_subset = df_test[df_test['language'] == lang]
            results["by_language"][lang] = self._evaluate_subset(lang_subset)
            
        # Evaluar por nivel de ruido
        if 'noise_level' in df_test.columns:
            for level in df_test['noise_level'].unique():
                level_subset = df_test[df_test['noise_level'] == level]
                results["by_noise_level"][level] = self._evaluate_subset(level_subset)
                
        # Evaluar por escenario
        if 'scenario' in df_test.columns:
            for scenario in df_test['scenario'].unique():
                scenario_subset = df_test[df_test['scenario'] == scenario]
                results["by_scenario"][scenario] = self._evaluate_subset(scenario_subset)
                
        # Evaluar frases de emergencia
        results["by_emergency_phrase"] = self._evaluate_emergency_phrases(df_test)
        
        return results
    
    def _evaluate_subset(self, df_subset: pd.DataFrame) -> Dict[str, float]:
        """Evaluar un subconjunto de datos"""
        if len(df_subset) == 0:
            return {"wer": 1.0, "samples": 0}
            
        wers = []
        for _, row in df_subset.iterrows():
            try:
                wer_score = self._transcribe_and_compare(
                    row['path'], 
                    row['transcription']
                )
                wers.append(wer_score)
            except Exception as e:
                self.logger.warning(f"Error evaluando {row['path']}: {e}")
                wers.append(1.0)  # Penalización por error
                
        return {
            "wer": float(np.mean(wers)),
            "wer_std": float(np.std(wers)),
            "samples": len(wers)
        }
    
    def _transcribe_and_compare(self, audio_path: str, reference: str) -> float:
        """Transcribir audio y comparar con referencia"""
        # Cargar y preprocesar audio
        audio = self.audio_processor.load_audio(audio_path)
        
        # Transcribir
        inputs = self.processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            predicted_ids = self.model.generate(inputs.input_features)
            
        transcription = self.processor.decode(predicted_ids[0], skip_special_tokens=True)
        
        # Calcular WER
        return wer(reference.lower(), transcription.lower())
    
    def _evaluate_emergency_phrases(self, df_test: pd.DataFrame) -> Dict[str, float]:
        """Evaluar reconocimiento de frases de emergencia específicas"""
        phrase_results = {}
        
        for lang_code, lang_info in self.languages_config["languages"].items():
            lang_phrases = lang_info.get("emergency_phrases", [])
            lang_subset = df_test[df_test['language'] == lang_code]
            
            for phrase in lang_phrases:
                # Buscar ejemplos que contengan esta frase
                phrase_examples = lang_subset[
                    lang_subset['transcription'].str.contains(phrase, case=False, na=False)
                ]
                
                if len(phrase_examples) > 0:
                    phrase_wer = self._evaluate_subset(phrase_examples)["wer"]
                    phrase_results[f"{lang_code}_{phrase}"] = phrase_wer
                    
        return phrase_results
    
    def generate_report(self, results: Dict[str, Any], output_path: str):
        """Generar reporte HTML de evaluación"""
        # Implementación para generar reporte visual
        # (puedes usar matplotlib/seaborn para gráficos)
        pass
        
    def analyze_errors(self, test_metadata_path: str, n_examples: int = 10):
        """Analizar errores más comunes"""
        df_test = pd.read_csv(test_metadata_path)
        errors = []
        
        for _, row in df_test.head(n_examples).iterrows():
            try:
                audio = self.audio_processor.load_audio(row['path'])
                inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    predicted_ids = self.model.generate(inputs.input_features)
                    
                prediction = self.processor.decode(predicted_ids[0], skip_special_tokens=True)
                reference = row['transcription']
                
                error_rate = wer(reference, prediction)
                errors.append({
                    'audio': row['path'],
                    'reference': reference,
                    'prediction': prediction,
                    'wer': error_rate,
                    'language': row.get('language', 'unknown'),
                    'noise_level': row.get('noise_level', 'unknown')
                })
                
            except Exception as e:
                self.logger.warning(f"Error analizando {row['path']}: {e}")
                
        return pd.DataFrame(errors)