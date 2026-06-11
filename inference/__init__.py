from .tts_engine import (
    TTSEngine,
    PyttxTTSEngine,
    GTTSTTSEngine,
    PrerecordedTTSEngine,
    TTSManager
)

from .word_detector import WordDetector
from .alphabet_detector import AlphabetDetector
from .sequence_handler import SequenceHandler
from .response_engine import ResponseEngine

try:
    from .hybrid_detector import HybridDetector
except Exception:
    HybridDetector = None

# Multi-class detectors
from .multiclass_detector import (
    MultiClassWordDetector,
    MultiClassAlphabetDetector,
    load_multiclass_model
)

__version__ = "2.0.0"  # Updated for multi-class support

__all__ = [
    # TTS Engines
    'TTSEngine',
    'PyttxTTSEngine',
    'GTTSTTSEngine',
    'PrerecordedTTSEngine',
    'TTSManager',
    
    # Binary Detectors (single gesture)
    'WordDetector',
    'AlphabetDetector',
    
    # Multi-class Detectors (multiple gestures)
    'MultiClassWordDetector',
    'MultiClassAlphabetDetector',
    'load_multiclass_model',
    
    # Handlers
    'SequenceHandler',
    'ResponseEngine',
    
    # Main System
    'HybridDetector'
]
