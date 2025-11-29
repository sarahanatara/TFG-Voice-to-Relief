🚨 Emergency ASR - Sistema de Transcripción de Emergencias Multilingüe
Sistema robusto de Automatic Speech Recognition (ASR) para transcripción de llamadas de emergencia en múltiples idiomas, diseñado como Trabajo de Fin de Grado (TFG).

🌟 Características Principales
Multilingüismo: Soporte para Español, Asturiano, Árabe, Guaraní y Francés

Robustez: Entrenado con augmentación realista para condiciones adversas

Detección de Emergencias: Identificación automática de frases críticas

Fine-tuning Efficient: Usa LoRA (Low-Rank Adaptation) para entrenamiento eficiente

Evaluación Comprehensiva: Métricas por idioma, nivel de ruido y escenario

🏗️ Estructura del Proyecto
text
emergency-asr-tfg/
├── configs/                 # Configuraciones
│   ├── augment_config.json
│   ├── training_config.json
│   └── languages.json
├── data/                   # Datos y audios
│   ├── raw/               # Audios originales por idioma
│   ├── noise/             # Ruidos para augmentación
│   └── generated/         # Datos aumentados
├── src/                   # Código fuente
│   ├── data_preparation/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   └── utils/
├── scripts/               # Scripts de ejecución
│   ├── 01_prepare_data.py
│   ├── 02_run_augmentation.py
│   ├── 03_train_model.py
│   ├── 04_evaluate.py
│   └── 05_demo.py
├── notebooks/             # Análisis y experimentos
│   ├── 01_data_exploration.ipynb
│   ├── 02_augmentation_analysis.ipynb
│   └── 03_results_analysis.ipynb
├── models/               # Modelos entrenados
└── logs/                 # Logs de entrenamiento
🚀 Instalación Rápida
1. Clonar y configurar entorno
bash
git clone <repository-url>
cd emergency-asr-tfg
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows
2. Instalar dependencias
bash
pip install -r requirements.txt
3. Preparar estructura de datos
bash
mkdir -p data/raw data/noise data/generated models logs
📊 Pipeline Completo
Paso 1: Preparación de Datos
bash
python scripts/01_prepare_data.py
Valida archivos de audio y transcripciones

Genera metadata inicial

Analiza distribución por idioma

Paso 2: Augmentación
bash
python scripts/02_run_augmentation.py
Mezcla ruidos realistas (tráfico, sirenas, multitudes)

Aplica efectos acústicos (reverb, telephone effect)

Genera múltiples variantes por audio

Paso 3: Entrenamiento
bash
python scripts/03_train_model.py
Fine-tuning con LoRA para eficiencia

Entrenamiento multilingüe

Early stopping y checkpointing

Paso 4: Evaluación
bash
python scripts/04_evaluate.py
WER por idioma y nivel de ruido

Detección de frases de emergencia

Reporte comprehensivo

Paso 5: Demo
bash
python scripts/05_demo.py
Interfaz Gradio para pruebas

Transcripción en tiempo real

Detección de emergencias

🗣️ Idiomas Soportados
Idioma	Código	Ejemplo Frases Emergencia
Español	es	"ayuda", "emergencia", "ambulancia"
Asturiano	ast	"ayuda", "emerxencia", "ambulancia"
Árabe	ar	"مساعدة", "طوارئ", "إسعاف"
Guaraní	gn	"pytyvõ", "emergencia", "ambulancia"
Francés	fr	"aide", "urgence", "ambulance"
🎯 Augmentación de Audio
El sistema aplica augmentación realista para simular condiciones de emergencia:

Niveles de Ruido
clean: SNR 30-40 dB (condiciones ideales)

low_noise: SNR 20-30 dB (ruido bajo)

medium_noise: SNR 10-20 dB (ruido moderado)

high_noise: SNR 0-10 dB (ruido alto)

extreme_noise: SNR -5-5 dB (condiciones extremas)

Escenarios de Emergencia
street_accident: Tráfico, sirenas, gritos

home_emergency: Electrodomésticos, voces, teléfono

public_space: Música, multitudes, ambiente

nature_emergency: Viento, lluvia, truenos

Efectos Acústicos
Reverb (simulación de espacios)

Pitch shift (variación de tono)

Time stretch (cambio de velocidad)

Telephone effect (filtro banda limitada)

Packet loss (simulación VoIP)

📈 Métricas de Evaluación
Métricas Principales
WER (Word Error Rate): Precisión general de transcripción

CER (Character Error Rate): Precisión a nivel de caracteres

Detección de Emergencias: Precision, Recall, F1-score

Robustez: Degradación bajo diferentes niveles de ruido

Evaluación por Categoría
Por idioma

Por nivel de ruido

Por escenario de emergencia

Por frase de emergencia específica

🛠️ Uso Avanzado
Entrenamiento Personalizado
python
from src.training.lora_trainer import MultilingualLoRATrainer
from src.utils.config_loader import load_config

# Cargar configuraciones
train_config = load_config("training_config")
languages = load_config("languages")["languages"]

# Inicializar trainer
trainer = MultilingualLoRATrainer(
    train_config=train_config,
    languages=languages,
    base_model="openai/whisper-small"
)

# Entrenar
trainer.train("data/generated/train_metadata.csv", "data/generated/val_metadata.csv")
Transcripción Programática
python
from src.inference.transcriber import EmergencyTranscriber
from src.utils.config_loader import load_config

# Cargar configuraciones
languages_config = load_config("languages")

# Inicializar transcriber
transcriber = EmergencyTranscriber("models/whisper-lora", languages_config)

# Transcribir audio
result = transcriber.transcribe("audio_emergencia.wav")
print(f"Transcripción: {result['transcription']}")
print(f"Emergencia detectada: {result['is_emergency']}")
print(f"Frases detectadas: {result['emergency_phrases']}")
Análisis de Resultados
python
from src.evaluation.analyzer import ComprehensiveEvaluator
from src.utils.config_loader import load_config

# Cargar configuraciones
languages_config = load_config("languages")

# Evaluar modelo
evaluator = ComprehensiveEvaluator("models/whisper-lora", languages_config)
results = evaluator.run_comprehensive_evaluation("data/generated/test_metadata.csv")

# Generar reporte
evaluator.generate_report(results, "evaluation_report.html")
🔧 Configuración
Ajuste de Hiperparámetros
Editar configs/training_config.json:

json
{
  "training": {
    "num_train_epochs": 5,
    "learning_rate": 1e-4,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 2
  },
  "lora": {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05
  }
}
Personalización de Augmentación
Editar configs/augment_config.json:

json
{
  "augmentation": {
    "n_variants_per_file": 25,
    "snr_levels": {
      "clean": [30, 40],
      "high_noise": [0, 10]
    },
    "emergency_scenarios": {
      "street_accident": ["trafico", "sirena", "gritos"]
    }
  }
}
📊 Resultados Esperados
Rendimiento General
WER en condiciones limpias: < 0.15

WER en condiciones extremas: < 0.30

Detección de emergencias: > 0.85 F1-score

Consistencia multilingüe: Rendimiento similar entre idiomas

Métricas de Robustez
Degradación máxima de WER: < 0.15 entre condiciones limpias y extremas

Detección confiable de frases críticas incluso con ruido

Transcripción aceptable con SNR hasta 0 dB

🐛 Solución de Problemas
Error: Memoria insuficiente
bash
# Reducir batch size
python scripts/03_train_model.py --batch_size 2

# Usar modelo más pequeño
python scripts/03_train_model.py --model_name openai/whisper-tiny
Error: Archivos de audio no encontrados
bash
# Verificar estructura de datos
python scripts/01_prepare_data.py

# Regenerar metadata
rm data/raw_metadata.csv
python scripts/01_prepare_data.py
Error: Dependencias faltantes
bash
# Reinstalar requirements
pip install -r requirements.txt --force-reinstall

# Instalar individualmente paquetes problemáticos
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
🤝 Contribución
Para contribuir al proyecto:

Fork el repositorio

Crea una rama para tu feature (git checkout -b feature/AmazingFeature)

Commit tus cambios (git commit -m 'Add some AmazingFeature')

Push a la rama (git push origin feature/AmazingFeature)

Abre un Pull Request

📝 Licencia
Este proyecto está licenciado bajo la Licencia MIT - ver el archivo LICENSE para detalles.

🙏 Agradecimientos
OpenAI por el modelo Whisper

Hugging Face por la biblioteca Transformers

PyTorch por el framework de deep learning

Agradecimientos especiales a todos los colaboradores que grabaron audios en múltiples idiomas

📞 Contacto y Soporte
Para preguntas académicas o técnicas sobre este proyecto TFG:

Autor: [Sara Hanafy Tárano]

Email: [UO287527@uniovi.es]

Universidad: [Universidad de Oviedo]

Departamento: [Departamento de informática, Grado en Ciencia e Ingeniería de Datos]

Nota: Este proyecto es parte de un Trabajo de Fin de Grado. Los resultados pueden variar dependiendo de la calidad y cantidad de datos de entrenamiento disponibles.#   T F G - V o i c e - t o - R e l i e f  
 