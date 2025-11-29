#!/usr/bin/env python3
"""
Script para preparar y validar datos multilingües de emergencia
"""
import os
import json
import pandas as pd
from pathlib import Path
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.data_preparation.audio_utils import AudioValidator

def main():
    logger = setup_logger("data_preparation")
    config = load_config("augment_config")
    languages = load_config("languages")["languages"]
    
    validator = AudioValidator(config["audio"])
    
    # Escanear todos los archivos de audio
    audio_files = []
    raw_dir = Path("data/raw")
    
    for lang_code in languages.keys():
        lang_dir = raw_dir / lang_code
        if not lang_dir.exists():
            logger.warning(f"Directorio no encontrado: {lang_dir}")
            continue
            
        for audio_file in lang_dir.glob("*.wav"):
            # Buscar transcripción
            txt_file = audio_file.with_suffix('.txt')
            if not txt_file.exists():
                logger.warning(f"Transcripción no encontrada para: {audio_file}")
                continue
                
            # Validar audio
            is_valid, duration = validator.validate_audio(audio_file)
            if not is_valid:
                logger.warning(f"Audio inválido: {audio_file}")
                continue
                
            # Leer transcripción
            with open(txt_file, 'r', encoding='utf-8') as f:
                transcription = f.read().strip()
                
            audio_files.append({
                'path': str(audio_file),
                'language': lang_code,
                'transcription': transcription,
                'duration': duration,
                'speaker': audio_file.stem.split('_')[0]
            })
    
    # Crear dataframe y estadísticas
    df = pd.DataFrame(audio_files)
    stats = df.groupby('language').agg({
        'path': 'count',
        'duration': ['sum', 'mean', 'std']
    }).round(2)
    
    logger.info(f"Archivos encontrados: {len(df)}")
    logger.info(f"Distribución por idioma:\n{stats}")
    
    # Guardar metadata
    df.to_csv('data/raw_metadata.csv', index=False, encoding='utf-8')
    logger.info("Metadata guardada en data/raw_metadata.csv")

if __name__ == "__main__":
    main()