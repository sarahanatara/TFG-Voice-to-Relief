#!/usr/bin/env python3
"""
Demo en tiempo real del sistema de transcripción de emergencias
"""
import torch
import gradio as gr
from src.inference.transcriber import EmergencyTranscriber
from src.utils.config_loader import load_config

def main():
    # Cargar configuraciones
    languages = load_config("languages")["languages"]
    
    # Inicializar transcriber
    transcriber = EmergencyTranscriber(
        model_dir="models/whisper-lora",
        languages_config=languages
    )
    
    def transcribe_audio(audio_path):
        if audio_path is None:
            return "Por favor, sube un archivo de audio"
            
        try:
            result = transcriber.transcribe(audio_path)
            return f"""
            **Transcripción:** {result['transcription']}
            **Idioma detectado:** {result['detected_language']}
            **Confianza:** {result['confidence']:.2f}
            **Frases de emergencia detectadas:** {', '.join(result['emergency_phrases'])}
            """
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Crear interfaz Gradio
    iface = gr.Interface(
        fn=transcribe_audio,
        inputs=gr.Audio(sources="upload", type="filepath"),
        outputs="markdown",
        title="Sistema de Transcripción de Emergencias Multilingüe",
        description="Sube un audio de emergencia para transcribirlo y detectar frases críticas"
    )
    
    iface.launch(share=True)

if __name__ == "__main__":
    main()