from .callbacks import (
    TrainingMonitor,
    MetricsLogger,
    BestModelSaver,
    ConfusionMatrixCallback,
    create_standard_callbacks
)

from .train_word_model import (
    train_word_model,
    load_preprocessed_data as load_word_data
)

from .train_alphabet_model import (
    train_alphabet_model,
    load_preprocessed_data as load_alphabet_data
)

# Pelatihan multi-kelas
from .train_multiclass_word import train_multiclass_word_model
from .train_multiclass_alphabet import train_multiclass_alphabet_model

from .evaluate import (
    load_model_and_data,
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_prediction_distribution,
    save_evaluation_report
)

__version__ = "2.0.0"  # Diperbarui untuk dukungan multi-kelas

__all__ = [
    # Callbacks
    'TrainingMonitor',
    'MetricsLogger',
    'BestModelSaver',
    'ConfusionMatrixCallback',
    'create_standard_callbacks',
    
    # Fungsi pelatihan biner (gestur tunggal)
    'train_word_model',
    'train_alphabet_model',
    'load_word_data',
    'load_alphabet_data',
    
    # Fungsi pelatihan multi-kelas (banyak gestur)
    'train_multiclass_word_model',
    'train_multiclass_alphabet_model',
    
    # Fungsi evaluasi
    'load_model_and_data',
    'evaluate_model',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_prediction_distribution',
    'save_evaluation_report'
]
