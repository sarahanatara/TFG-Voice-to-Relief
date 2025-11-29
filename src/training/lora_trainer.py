from pathlib import Path
import torch
import pandas as pd
from transformers import (
    WhisperForConditionalGeneration, 
    WhisperProcessor, 
    TrainingArguments, 
    Trainer,
    Seq2SeqTrainingArguments
)
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset, Audio
import evaluate
from typing import Dict, Any, List, Optional
import numpy as np
from src.utils.logger import setup_logger

class MultilingualLoRATrainer:
    def __init__(
        self, 
        train_config: Dict[str, Any],
        languages: Dict[str, Any],
        base_model: str = "openai/whisper-small"
    ):
        self.train_config = train_config
        self.languages = languages
        self.base_model = base_model
        self.logger = setup_logger("lora_trainer")
        
        # Inicializar modelo y processor
        self.processor = WhisperProcessor.from_pretrained(
            base_model, 
            language="multilingual", 
            task="transcribe"
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(base_model)
        
        # Configurar LoRA
        self._setup_lora()
        
        # Métrica de evaluación
        self.wer_metric = evaluate.load("wer")
        
    def _setup_lora(self):
        """Configurar adaptación LoRA"""
        lora_config = self.train_config["lora"]
        
        peft_config = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["lora_alpha"],
            lora_dropout=lora_config["lora_dropout"],
            target_modules=lora_config["target_modules"],
            bias=lora_config["bias"],
        )
        
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        
    def _prepare_dataset(self, metadata_path: str) -> Dataset:
        """Preparar dataset para entrenamiento"""
        df = pd.read_csv(metadata_path)
        
        # Filtrar audios válidos
        valid_audio = []
        valid_text = []
        
        for _, row in df.iterrows():
            try:
                # Verificar que el archivo existe
                if not pd.isna(row['path']) and Path(row['path']).exists():
                    valid_audio.append(row['path'])
                    valid_text.append(row['transcription'])
            except Exception as e:
                self.logger.warning(f"Error procesando fila: {e}")
                continue
                
        # Crear dataset
        dataset = Dataset.from_dict({
            "audio": valid_audio,
            "transcription": valid_text
        })
        
        # Cargar audio
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        return dataset
    
    def _preprocess_function(self, examples):
        """Preprocesar ejemplos para el modelo"""
        # Cargar audio
        audio = examples["audio"]
        
        # Procesar audio
        inputs = self.processor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
            padding=True,
            max_length=480000,  # 30 segundos a 16kHz
            truncation=True
        )
        
        # Procesar texto
        labels = self.processor.tokenizer(
            examples["transcription"],
            return_tensors="pt",
            padding=True,
            max_length=448,  # Máximo largo para Whisper
            truncation=True
        ).input_ids
        
        examples["input_features"] = inputs.input_features[0]
        examples["labels"] = labels[0]
        
        return examples
    
    def _compute_metrics(self, eval_pred):
        """Calcular métricas de evaluación"""
        pred_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids
        
        # Reemplazar padding tokens
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        
        # Decodificar predicciones
        pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.batch_decode(label_ids, skip_special_tokens=True)
        
        # Calcular WER
        wer = self.wer_metric.compute(predictions=pred_str, references=label_str)
        
        return {"wer": wer}
    
    def train(
        self, 
        train_metadata_path: str, 
        val_metadata_path: str,
        output_dir: Optional[str] = None
    ):
        """Entrenar el modelo"""
        
        # Preparar datasets
        train_dataset = self._prepare_dataset(train_metadata_path)
        val_dataset = self._prepare_dataset(val_metadata_path)
        
        # Preprocesar
        train_dataset = train_dataset.map(
            self._preprocess_function,
            remove_columns=train_dataset.column_names,
            batched=False
        )
        
        val_dataset = val_dataset.map(
            self._preprocess_function,
            remove_columns=val_dataset.column_names,
            batched=False
        )
        
        # Configurar entrenamiento
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir or self.train_config["output"]["output_dir"],
            num_train_epochs=self.train_config["training"]["num_train_epochs"],
            per_device_train_batch_size=self.train_config["training"]["per_device_train_batch_size"],
            per_device_eval_batch_size=self.train_config["training"]["per_device_eval_batch_size"],
            gradient_accumulation_steps=self.train_config["training"]["gradient_accumulation_steps"],
            warmup_steps=self.train_config["training"]["warmup_steps"],
            learning_rate=self.train_config["training"]["learning_rate"],
            weight_decay=self.train_config["training"]["weight_decay"],
            lr_scheduler_type=self.train_config["training"]["lr_scheduler_type"],
            evaluation_strategy="steps",
            eval_steps=self.train_config["training"]["eval_steps"],
            save_steps=self.train_config["training"]["save_steps"],
            logging_steps=self.train_config["training"]["logging_steps"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            predict_with_generate=True,
            generation_max_length=225,
            report_to=["tensorboard"],
            push_to_hub=False,
        )
        
        # Crear trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.processor,
            compute_metrics=self._compute_metrics,
        )
        
        # Entrenar
        self.logger.info("Iniciando entrenamiento...")
        trainer.train()
        
        # Guardar modelo final
        trainer.save_model()
        self.logger.info(f"Modelo guardado en {output_dir}")
        
        return trainer
    
    def evaluate(self, test_metadata_path: str) -> Dict[str, float]:
        """Evaluar modelo en test set"""
        test_dataset = self._prepare_dataset(test_metadata_path)
        test_dataset = test_dataset.map(
            self._preprocess_function,
            remove_columns=test_dataset.column_names,
            batched=False
        )
        
        training_args = Seq2SeqTrainingArguments(
            output_dir="./tmp",
            per_device_eval_batch_size=self.train_config["training"]["per_device_eval_batch_size"],
            predict_with_generate=True,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            compute_metrics=self._compute_metrics,
        )
        
        results = trainer.evaluate(test_dataset)
        return results