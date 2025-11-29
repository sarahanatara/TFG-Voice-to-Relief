#!/usr/bin/env python3
"""
Evaluación comprehensiva por idioma, nivel de ruido y tipo de escenario
"""
from src.evaluation.analyzer import ComprehensiveEvaluator
from src.utils.config_loader import load_config

def main():
    evaluator = ComprehensiveEvaluator(
        model_dir="models/whisper-lora",
        languages_config=load_config("languages")
    )
    
    # Evaluar en test set
    results = evaluator.run_comprehensive_evaluation(
        test_metadata_path="data/generated/test_metadata.csv"
    )
    
    # Generar reporte detallado
    evaluator.generate_report(results, "evaluation_report.html")
    
    print("Evaluación completada. Ver evaluation_report.html")

if __name__ == "__main__":
    main()