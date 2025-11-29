#!/usr/bin/env python3
"""
Script avanzado de augmentación para escenarios de emergencia multilingües
"""
import os
import json
import pandas as pd
from pathlib import Path
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.data_preparation.augmentor import EmergencyAugmentor

def main():
    logger = setup_logger("augmentation")
    config = load_config("augment_config")
    languages = load_config("languages")["languages"]
    
    # Cargar metadata original
    raw_metadata_path = "data/raw_metadata.csv"
    if not os.path.exists(raw_metadata_path):
        logger.error("Primero ejecuta 01_prepare_data.py")
        return
        
    df_raw = pd.read_csv(raw_metadata_path)
    
    # Inicializar augmentador
    augmentor = EmergencyAugmentor(config)
    
    # Generar datos aumentados
    results = []
    for _, row in df_raw.iterrows():
        logger.info(f"Procesando: {row['path']} ({row['language']})")
        
        variants = augmentor.generate_variants(
            audio_path=row['path'],
            transcription=row['transcription'],
            language=row['language'],
            speaker_id=row['speaker']
        )
        results.extend(variants)
    
    # Crear dataset final
    df_final = pd.DataFrame(results)
    
    # Dividir en train/val/test
    split_config = config["data_split"]
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    n_total = len(df_final)
    n_train = int(n_total * split_config["train_ratio"])
    n_val = int(n_total * split_config["val_ratio"])
    
    df_train = df_final.iloc[:n_train]
    df_val = df_final.iloc[n_train:n_train + n_val]
    df_test = df_final.iloc[n_train + n_val:]
    
    # Guardar splits
    output_dir = Path("data/generated")
    df_train.to_csv(output_dir / "train_metadata.csv", index=False)
    df_val.to_csv(output_dir / "val_metadata.csv", index=False) 
    df_test.to_csv(output_dir / "test_metadata.csv", index=False)
    
    logger.info(f"Augmentación completada:")
    logger.info(f"  - Train: {len(df_train)} samples")
    logger.info(f"  - Val: {len(df_val)} samples") 
    logger.info(f"  - Test: {len(df_test)} samples")
    
    # Estadísticas por idioma y nivel de ruido
    stats_lang = df_final.groupby('language').size()
    stats_noise = df_final.groupby('noise_level').size()
    
    logger.info(f"Distribución por idioma:\n{stats_lang}")
    logger.info(f"Distribución por nivel de ruido:\n{stats_noise}")

if __name__ == "__main__":
    main()