#!/usr/bin/env python3
"""
Entrenamiento con LoRA multilingüe y manejo de idiomas
"""
from src.training.lora_trainer import MultilingualLoRATrainer
from src.utils.config_loader import load_config

def main():
    # Cargar configuraciones
    train_config = load_config("training_config")
    augment_config = load_config("augment_config")
    languages = load_config("languages")["languages"]
    
    # Inicializar trainer
    trainer = MultilingualLoRATrainer(
        train_config=train_config,
        languages=languages,
        base_model="openai/whisper-small"
    )
    
    # Entrenar
    trainer.train(
        train_metadata_path="data/generated/train_metadata.csv",
        val_metadata_path="data/generated/val_metadata.csv"
    )
    
    # Evaluar en validation set
    results = trainer.evaluate("data/generated/val_metadata.csv")
    print("Resultados de entrenamiento:", results)

if __name__ == "__main__":
    main()