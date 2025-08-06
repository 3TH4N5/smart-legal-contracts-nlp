"""
Configuration settings for the Smart Legal Contracts project
Handles loading and validation of configuration from YAML files
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Model configuration settings"""
    base_model: str
    legal_model: str
    max_length: int
    batch_size: int
    learning_rate: float
    num_epochs: int

@dataclass
class TrainingConfig:
    """Training configuration settings"""
    test_size: float
    val_size: float
    random_seed: int
    early_stopping_patience: int
    save_best_model: bool

@dataclass
class PathConfig:
    """Path configuration settings"""
    raw_data: Path
    processed_data: Path
    samples: Path
    templates: Path
    saved_models: Path
    checkpoints: Path
    logs: Path
    outputs: Path

class ConfigManager:
    """Manages configuration loading and access"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to config file. If None, uses default config/cuad_config.yaml
        """
        if config_path is None:
            self.config_path = Path(__file__).parent / "cuad_config.yaml"
        else:
            self.config_path = Path(config_path)
            
        self.config = self._load_config()
        self._validate_config()
        self._setup_paths()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            return self._get_default_config()
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if file loading fails"""
        return {
            'clause_mappings': {},
            'text_cleaning': {
                'remove_patterns': [],
                'non_answers': ['none', 'n/a'],
                'min_answer_length': 10
            },
            'models': {
                'classification': {
                    'base_model': 'nlpaueb/legal-bert-base-uncased',
                    'max_length': 512,
                    'batch_size': 16,
                    'learning_rate': 2e-5,
                    'num_epochs': 3
                }
            },
            'paths': {
                'data': {
                    'raw': 'data/raw',
                    'processed': 'data/processed'
                },
                'models': {
                    'saved': 'models/saved_models'
                }
            }
        }
    
    def _validate_config(self):
        """Validate configuration structure and values"""
        required_sections = ['clause_mappings', 'text_cleaning', 'models', 'paths']
        
        for section in required_sections:
            if section not in self.config:
                logger.warning(f"Missing configuration section: {section}")
                
        # Validate model config
        if 'models' in self.config and 'classification' in self.config['models']:
            model_config = self.config['models']['classification']
            required_model_fields = ['base_model', 'max_length', 'batch_size']
            
            for field in required_model_fields:
                if field not in model_config:
                    logger.warning(f"Missing model configuration field: {field}")
    
    def _setup_paths(self):
        """Setup and create necessary directories"""
        if 'paths' not in self.config:
            return
            
        # Create directories from config
        for category, paths in self.config['paths'].items():
            if isinstance(paths, dict):
                for path_name, path_value in paths.items():
                    path_obj = Path(path_value)
                    path_obj.mkdir(parents=True, exist_ok=True)
                    logger.debug(f"Created directory: {path_obj}")
    
    # Property accessors for easy access to config sections
    @property
    def clause_mappings(self) -> Dict[str, str]:
        """Get clause type mappings"""
        return self.config.get('clause_mappings', {})
    
    @property
    def clause_types(self) -> List[str]:
        """Get list of all clause types"""
        return list(self.clause_mappings.values())
    
    @property
    def text_cleaning(self) -> Dict[str, Any]:
        """Get text cleaning configuration"""
        return self.config.get('text_cleaning', {})
    
    @property
    def remove_patterns(self) -> List[str]:
        """Get text removal patterns"""
        return self.text_cleaning.get('remove_patterns', [])
    
    @property
    def non_answers(self) -> List[str]:
        """Get non-answer patterns"""
        return self.text_cleaning.get('non_answers', [])
    
    @property
    def min_answer_length(self) -> int:
        """Get minimum answer length"""
        return self.text_cleaning.get('min_answer_length', 10)
    
    @property
    def max_context_length(self) -> int:
        """Get maximum context length"""
        return self.text_cleaning.get('max_context_length', 2048)
    
    @property
    def classification_model_config(self) -> ModelConfig:
        """Get classification model configuration"""
        model_config = self.config.get('models', {}).get('classification', {})
        
        return ModelConfig(
            base_model=model_config.get('base_model', 'bert-base-uncased'),
            legal_model=model_config.get('legal_model', 'nlpaueb/legal-bert-base-uncased'),
            max_length=model_config.get('max_length', 512),
            batch_size=model_config.get('batch_size', 16),
            learning_rate=model_config.get('learning_rate', 2e-5),
            num_epochs=model_config.get('num_epochs', 3)
        )
    
    @property
    def training_config(self) -> TrainingConfig:
        """Get training configuration"""
        training_config = self.config.get('training', {})
        
        return TrainingConfig(
            test_size=training_config.get('test_size', 0.2),
            val_size=training_config.get('val_size', 0.1),
            random_seed=training_config.get('random_seed', 42),
            early_stopping_patience=training_config.get('early_stopping_patience', 3),
            save_best_model=training_config.get('save_best_model', True)
        )
    
    @property
    def paths(self) -> PathConfig:
        """Get path configuration"""
        paths_config = self.config.get('paths', {})
        data_paths = paths_config.get('data', {})
        model_paths = paths_config.get('models', {})
        output_paths = paths_config.get('outputs', {})
        
        return PathConfig(
            raw_data=Path(data_paths.get('raw', 'data/raw')),
            processed_data=Path(data_paths.get('processed', 'data/processed')),
            samples=Path(data_paths.get('samples', 'data/raw/samples')),
            templates=Path(data_paths.get('templates', 'data/templates')),
            saved_models=Path(model_paths.get('saved', 'models/saved_models')),
            checkpoints=Path(model_paths.get('checkpoints', 'models/checkpoints')),
            logs=Path(model_paths.get('logs', 'models/logs')),
            outputs=Path(output_paths.get('contracts', 'outputs/generated_contracts'))
        )
    
    def get_clause_type(self, question: str) -> str:
        """
        Extract clause type from question text using configured mappings
        
        Args:
            question: Question text to analyze
            
        Returns:
            Standardized clause type or 'unknown'
        """
        question_lower = question.lower()
        
        # Check each mapping
        for original_name, standardized_name in self.clause_mappings.items():
            if original_name.lower() in question_lower:
                return standardized_name
        
        # Fallback to pattern matching
        question_patterns = self.config.get('question_patterns', [])
        for pattern in question_patterns:
            import re
            match = re.search(pattern, question_lower)
            if match:
                clause_text = match.group(1)
                # Try to find matching clause type
                for original_name, standardized_name in self.clause_mappings.items():
                    if original_name.lower() in clause_text:
                        return standardized_name
        
        return 'unknown'
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled"""
        debug_config = self.config.get('debug', {})
        return debug_config.get('verbose_logging', False)
    
    def get_sample_size(self) -> Optional[int]:
        """Get debug sample size"""
        debug_config = self.config.get('debug', {})
        return debug_config.get('sample_size')
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict:
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.config, updates)
        logger.info("Configuration updated")
    
    def save_config(self, output_path: Optional[str] = None):
        """Save current configuration to file"""
        if output_path is None:
            output_path = self.config_path
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {output_path}")

# Global configuration instance
_config_manager = None

def get_config(config_path: Optional[str] = None) -> ConfigManager:
    """
    Get global configuration manager instance
    
    Args:
        config_path: Path to config file (only used on first call)
        
    Returns:
        ConfigManager instance
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    
    return _config_manager

def reload_config(config_path: Optional[str] = None):
    """Reload configuration from file"""
    global _config_manager
    _config_manager = ConfigManager(config_path)
    return _config_manager

# Convenience functions for common config access
def get_clause_mappings() -> Dict[str, str]:
    """Get clause type mappings"""
    return get_config().clause_mappings

def get_clause_types() -> List[str]:
    """Get list of clause types"""
    return get_config().clause_types

def get_model_config() -> ModelConfig:
    """Get model configuration"""
    return get_config().classification_model_config

def get_paths() -> PathConfig:
    """Get path configuration"""
    return get_config().paths

if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    
    print("Configuration loaded successfully!")
    print(f"Number of clause types: {len(config.clause_types)}")
    print(f"Model: {config.classification_model_config.base_model}")
    print(f"Data path: {config.paths.raw_data}")
