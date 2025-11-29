import json
import os
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Cargar configuración desde archivo JSON"""
        config_path = self.config_dir / f"{config_name}.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        return config
    
    def save_config(self, config: Dict[str, Any], config_name: str):
        """Guardar configuración en archivo JSON"""
        config_path = self.config_dir / f"{config_name}.json"
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
    def get_available_configs(self) -> list:
        """Obtener lista de configuraciones disponibles"""
        return [f.stem for f in self.config_dir.glob("*.json")]

# Instancia global para uso fácil
config_loader = ConfigLoader()

def load_config(config_name: str) -> Dict[str, Any]:
    """Función helper para cargar configuración"""
    return config_loader.load_config(config_name)