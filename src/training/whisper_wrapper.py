import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
from typing import Dict, Any, Optional
from src.utils.logger import setup_logger
import numpy as np

class WhisperWrapper:
    def __init__(self, model_name: str = "openai/whisper-small", use_lora: bool = False):
        self.model_name = model_name
        self.use_lora = use_lora
        self.logger = setup_logger("whisper_wrapper")
        
        # Cargar modelo base y processor
        self.processor = WhisperProcessor.from_pretrained(
            model_name, 
            language="multilingual", 
            task="transcribe"
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
    def load_lora_adapter(self, adapter_path: str):
        """Cargar adaptador LoRA entrenado"""
        try:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.use_lora = True
            self.logger.info(f"Adaptador LoRA cargado desde {adapter_path}")
        except Exception as e:
            self.logger.error(f"Error cargando adaptador LoRA: {e}")
            raise
    
    def prepare_for_training(self, training_config: Dict[str, Any]):
        """Preparar modelo para entrenamiento"""
        # Congelar parámetros del modelo base excepto los heads adaptativos
        if self.use_lora:
            # Con LoRA, solo los parámetros LoRA son entrenables
            self.model.print_trainable_parameters()
        else:
            # Fine-tuning completo o parcial
            self._setup_fine_tuning(training_config)
    
    def _setup_fine_tuning(self, training_config: Dict[str, Any]):
        """Configurar fine-tuning del modelo"""
        # Congelar encoder o decoder según configuración
        freeze_encoder = training_config.get("freeze_encoder", False)
        freeze_decoder = training_config.get("freeze_decoder", False)
        
        if freeze_encoder:
            for param in self.model.model.encoder.parameters():
                param.requires_grad = False
                
        if freeze_decoder:
            for param in self.model.model.decoder.parameters():
                param.requires_grad = False
        
        self.logger.info("Parámetros entrenables:")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"  Total: {total_params:,}")
        self.logger.info(f"  Entrenables: {trainable_params:,}")
        self.logger.info(f"  Porcentaje: {100 * trainable_params / total_params:.2f}%")
    
    def get_model(self) -> WhisperForConditionalGeneration:
        """Obtener el modelo para entrenamiento"""
        return self.model
    
    def get_processor(self) -> WhisperProcessor:
        """Obtener el processor para preprocesamiento"""
        return self.processor
    
    def save_model(self, output_dir: str, save_full_model: bool = False):
        """Guardar modelo y processor"""
        if self.use_lora and not save_full_model:
            # Guardar solo adaptador LoRA
            self.model.save_pretrained(output_dir)
        else:
            # Guardar modelo completo
            self.model.save_pretrained(output_dir)
        
        self.processor.save_pretrained(output_dir)
        self.logger.info(f"Modelo guardado en {output_dir}")
    
    def transcribe(self, 
                  audio: np.ndarray, 
                  language: Optional[str] = None,
                  task: str = "transcribe") -> str:
        """Transcribir audio usando el modelo"""
        # Preparar inputs
        inputs = self.processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).to(self.device)
        
        # Configurar generación
        forced_decoder_ids = None
        if language:
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language=language, 
                task=task
            )
        
        # Generar
        with torch.no_grad():
            predicted_ids = self.model.generate(
                inputs.input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_length=225
            )
        
        return self.processor.decode(predicted_ids[0], skip_special_tokens=True)
    
    def get_embedding(self, audio: np.ndarray) -> torch.Tensor:
        """Obtener embeddings del encoder para análisis"""
        inputs = self.processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            encoder_outputs = self.model.model.encoder(
                input_features=inputs.input_features
            )
        
        return encoder_outputs.last_hidden_state

# Función de conveniencia para cargar modelo entrenado
def load_trained_model(model_dir: str, use_lora: bool = True) -> WhisperWrapper:
    """Cargar modelo entrenado desde directorio"""
    wrapper = WhisperWrapper(use_lora=use_lora)
    
    if use_lora:
        wrapper.load_lora_adapter(model_dir)
    else:
        wrapper.model = WhisperForConditionalGeneration.from_pretrained(model_dir)
        wrapper.model.to(wrapper.device)
    
    wrapper.processor = WhisperProcessor.from_pretrained(model_dir)
    return wrapper