import evaluate
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from jiwer import wer, cer
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

class EmergencyMetrics:
    def __init__(self):
        self.wer_metric = evaluate.load("wer")
        self.cer_metric = evaluate.load("cer")
        
    def calculate_detailed_metrics(self, references: List[str], predictions: List[str]) -> Dict[str, Any]:
        """Calcular métricas detalladas de evaluación"""
        
        # Métricas básicas
        wer_score = self.wer_metric.compute(
            predictions=predictions, 
            references=references
        )
        cer_score = self.cer_metric.compute(
            predictions=predictions, 
            references=references
        )
        
        # Métricas de detección de emergencia
        emergency_metrics = self._calculate_emergency_detection_metrics(
            references, predictions
        )
        
        # Análisis de errores por longitud
        length_analysis = self._analyze_length_errors(references, predictions)
        
        return {
            "wer": wer_score,
            "cer": cer_score,
            "emergency_detection": emergency_metrics,
            "length_analysis": length_analysis,
            "sample_count": len(references)
        }
    
    def _calculate_emergency_detection_metrics(self, references: List[str], predictions: List[str]) -> Dict[str, float]:
        """Calcular métricas específicas para detección de emergencias"""
        
        # Lista de palabras clave de emergencia (puede extenderse)
        emergency_keywords = [
            'ayuda', 'emergencia', 'ambulancia', 'policía', 'fuego', 'accidente',
            'herido', 'sangrando', 'respirar', 'auxilio', 'urgencia', 'hospital'
        ]
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for ref, pred in zip(references, predictions):
            ref_lower = ref.lower()
            pred_lower = pred.lower()
            
            # Detectar emergencia en referencia
            ref_emergency = any(keyword in ref_lower for keyword in emergency_keywords)
            # Detectar emergencia en predicción
            pred_emergency = any(keyword in pred_lower for keyword in emergency_keywords)
            
            if ref_emergency and pred_emergency:
                true_positives += 1
            elif ref_emergency and not pred_emergency:
                false_negatives += 1
            elif not ref_emergency and pred_emergency:
                false_positives += 1
        
        # Calcular métricas
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }
    
    def _analyze_length_errors(self, references: List[str], predictions: List[str]) -> Dict[str, Any]:
        """Analizar errores relacionados con la longitud del texto"""
        ref_lengths = [len(text.split()) for text in references]
        pred_lengths = [len(text.split()) for text in predictions]
        
        length_diffs = [pred - ref for ref, pred in zip(ref_lengths, pred_lengths)]
        
        return {
            "mean_reference_length": np.mean(ref_lengths),
            "mean_prediction_length": np.mean(pred_lengths),
            "mean_length_difference": np.mean(length_diffs),
            "std_length_difference": np.std(length_diffs),
            "correlation_length": np.corrcoef(ref_lengths, pred_lengths)[0, 1] if len(ref_lengths) > 1 else 0
        }
    
    def calculate_metrics_by_category(self, 
                                    references: List[str], 
                                    predictions: List[str],
                                    categories: List[str]) -> Dict[str, Dict[str, Any]]:
        """Calcular métricas agrupadas por categoría"""
        
        unique_categories = set(categories)
        results = {}
        
        for category in unique_categories:
            # Filtrar muestras por categoría
            cat_indices = [i for i, cat in enumerate(categories) if cat == category]
            cat_references = [references[i] for i in cat_indices]
            cat_predictions = [predictions[i] for i in cat_indices]
            
            if cat_references:  # Solo calcular si hay muestras
                results[category] = self.calculate_detailed_metrics(
                    cat_references, cat_predictions
                )
        
        return results

    def generate_error_analysis_report(self, 
                                     references: List[str], 
                                     predictions: List[str],
                                     metadata: Optional[List[Dict]] = None) -> pd.DataFrame:
        """Generar reporte detallado de análisis de errores"""
        
        error_data = []
        
        for i, (ref, pred) in enumerate(zip(references, predictions)):
            error_rate = wer(ref, pred)
            
            error_info = {
                'index': i,
                'reference': ref,
                'prediction': pred,
                'wer': error_rate,
                'length_reference': len(ref.split()),
                'length_prediction': len(pred.split()),
                'length_difference': abs(len(ref.split()) - len(pred.split()))
            }
            
            # Añadir metadatos si están disponibles
            if metadata and i < len(metadata):
                for key, value in metadata[i].items():
                    error_info[key] = value
            
            error_data.append(error_info)
        
        return pd.DataFrame(error_data)

# Instancia global para uso fácil
emergency_metrics = EmergencyMetrics()