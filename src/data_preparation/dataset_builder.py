import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datasets import Dataset, Audio
import json
from src.utils.logger import setup_logger

class EmergencyDatasetBuilder:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logger("dataset_builder")
        
    def build_dataset_from_metadata(self, metadata_path: str) -> Dataset:
        """Construir dataset Hugging Face desde metadata"""
        df = pd.read_csv(metadata_path)
        
        # Validar datos
        df = self._validate_dataframe(df)
        
        # Crear dataset
        dataset_dict = {
            "path": df["path"].tolist(),
            "transcription": df["transcription"].tolist(),
            "language": df.get("language", ["unknown"] * len(df)).tolist(),
            "audio": df["path"].tolist()  # Para casteo posterior
        }
        
        # Añadir metadatos adicionales si existen
        for col in ["noise_level", "scenario", "snr", "effect"]:
            if col in df.columns:
                dataset_dict[col] = df[col].tolist()
        
        dataset = Dataset.from_dict(dataset_dict)
        
        # Castear columna de audio
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        self.logger.info(f"Dataset construido: {len(dataset)} muestras")
        return dataset
    
    def _validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validar y limpiar dataframe"""
        # Verificar columnas requeridas
        required_cols = ["path", "transcription"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Columna requerida faltante: {col}")
        
        # Filtrar filas con paths inválidos
        valid_mask = df["path"].apply(lambda x: Path(x).exists() if pd.notna(x) else False)
        invalid_count = len(df) - valid_mask.sum()
        
        if invalid_count > 0:
            self.logger.warning(f"Eliminadas {invalid_count} filas con paths inválidos")
            df = df[valid_mask]
        
        # Filtrar transcripciones vacías
        empty_transcripts = df["transcription"].isna() | (df["transcription"].str.strip() == "")
        empty_count = empty_transcripts.sum()
        
        if empty_count > 0:
            self.logger.warning(f"Eliminadas {empty_count} filas con transcripciones vacías")
            df = df[~empty_transcripts]
        
        return df.reset_index(drop=True)
    
    def create_data_splits(self, metadata_path: str, output_dir: str):
        """Crear splits de train/val/test y guardarlos"""
        df = pd.read_csv(metadata_path)
        df = self._validate_dataframe(df)
        
        # Configuración de splits
        split_config = self.config.get("data_split", {
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1
        })
        
        # Mezclar datos
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Calcular tamaños
        n_total = len(df)
        n_train = int(n_total * split_config["train_ratio"])
        n_val = int(n_total * split_config["val_ratio"])
        
        # Crear splits
        train_df = df.iloc[:n_train]
        val_df = df.iloc[n_train:n_train + n_val]
        test_df = df.iloc[n_train + n_val:]
        
        # Guardar splits
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_df.to_csv(output_path / "train_metadata.csv", index=False)
        val_df.to_csv(output_path / "val_metadata.csv", index=False)
        test_df.to_csv(output_path / "test_metadata.csv", index=False)
        
        self.logger.info(f"Splits creados en {output_dir}:")
        self.logger.info(f"  - Train: {len(train_df)} muestras")
        self.logger.info(f"  - Val: {len(val_df)} muestras")
        self.logger.info(f"  - Test: {len(test_df)} muestras")
        
        # Estadísticas por idioma
        if "language" in df.columns:
            self._log_language_stats(train_df, "Train")
            self._log_language_stats(val_df, "Validation")
            self._log_language_stats(test_df, "Test")
    
    def _log_language_stats(self, df: pd.DataFrame, split_name: str):
        """Log estadísticas por idioma"""
        if "language" in df.columns:
            lang_stats = df["language"].value_counts()
            self.logger.info(f"{split_name} - Distribución por idioma:")
            for lang, count in lang_stats.items():
                self.logger.info(f"    {lang}: {count} muestras")
    
    def create_huggingface_dataset(self, metadata_path: str, split: str = "train") -> Dataset:
        """Crear dataset listo para Hugging Face"""
        split_paths = {
            "train": "data/generated/train_metadata.csv",
            "val": "data/generated/val_metadata.csv",
            "test": "data/generated/test_metadata.csv"
        }
        
        if split not in split_paths:
            raise ValueError(f"Split debe ser uno de: {list(split_paths.keys())}")
        
        dataset_path = split_paths[split]
        if not Path(dataset_path).exists():
            self.logger.warning(f"Split {split} no encontrado, creando splits...")
            self.create_data_splits(metadata_path, "data/generated")
        
        return self.build_dataset_from_metadata(dataset_path)

# Función de conveniencia
def load_emergency_dataset(split: str = "train", config: Optional[Dict] = None) -> Dataset:
    """Cargar dataset de emergencias"""
    if config is None:
        from src.utils.config_loader import load_config
        config = load_config("augment_config")
    
    builder = EmergencyDatasetBuilder(config)
    return builder.create_huggingface_dataset(
        "data/generated/metadata.csv", 
        split=split
    )