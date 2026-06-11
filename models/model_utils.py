import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, List, Tuple, Optional
import os
import json
import matplotlib.pyplot as plt


def get_model_config() -> Dict:
    """
    Dapatkan konfigurasi model default
    
    Returns:
        Dictionary dengan konfigurasi model
    """
    return {
        # Konfigurasi model kata
        'word': {
            'input_shape': (45, 126),  # (frames, features)
            'num_classes': 1,  # Klasifikasi biner untuk POC
            'cnn_filters': [32, 64],
            'cnn_kernel_size': 3,
            'lstm_units': [64, 32],
            'dropout_rate': 0.3,
            'dense_units': [32],
            'learning_rate': 0.001
        },
        
        # Konfigurasi model alfabet
        'alphabet': {
            'input_shape': (1, 126),  # (frames, features) - statis
            'num_classes': 1,  # Klasifikasi biner untuk POC
            'cnn_filters': [32, 64],
            'cnn_kernel_size': 3,
            'lstm_units': [32],
            'dropout_rate': 0.3,
            'dense_units': [32, 16],
            'learning_rate': 0.001
        },
        
        # Konfigurasi pelatihan
        'training': {
            'epochs': 50,
            'batch_size': 8,
            'validation_split': 0.0,  # Kita punya set validasi terpisah
            'early_stopping_patience': 10,
            'reduce_lr_patience': 5
        }
    }


def create_callbacks(
    model_name: str,
    output_dir: str = 'trained_models',
    monitor: str = 'val_accuracy',
    patience: int = 10
) -> List[keras.callbacks.Callback]:
    """
    Buat callback pelatihan
    
    Args:
        model_name: Nama model
        output_dir: Direktori untuk menyimpan model
        monitor: Metrik untuk dipantau untuk checkpointing (default: 'val_accuracy')
        patience: Jumlah epoch tanpa peningkatan sebelum penghentian awal (default: 10)
        
    Returns:
        Daftar callback
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Checkpoint model - simpan model terbaik
    checkpoint_path = os.path.join(output_dir, f'{model_name}_best.keras')
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=monitor,
        save_best_only=True,
        save_weights_only=False,
        mode='max' if 'accuracy' in monitor else 'min',
        verbose=1
    )
    
    # Penghentian awal
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    
    # Kurangi learning rate saat plateau
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=max(3, patience // 2),  # Setengah dari patience penghentian awal, min 3
        min_lr=1e-6,
        verbose=1
    )
    
    # Logging TensorBoard
    log_dir = os.path.join('logs', model_name)
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True
    )
    
    # Logger CSV
    csv_path = os.path.join(output_dir, f'{model_name}_training.csv')
    csv_logger = keras.callbacks.CSVLogger(csv_path)
    
    return [checkpoint, early_stop, reduce_lr, tensorboard, csv_logger]


def compile_model(
    model: keras.Model,
    learning_rate: float = 0.001,
    num_classes: int = 1
) -> keras.Model:
    """
    Kompilasi model dengan loss dan metrik yang sesuai
    
    Args:
        model: Model Keras untuk dikompilasi
        learning_rate: Learning rate untuk optimizer
        num_classes: Jumlah kelas (1 untuk biner, >1 untuk multi-kelas)
        
    Returns:
        Model yang dikompilasi
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    if num_classes == 1:
        # Klasifikasi biner
        loss = 'binary_crossentropy'
        metrics = ['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    else:
        # Klasifikasi multi-kelas
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model


def plot_training_history(
    history: keras.callbacks.History,
    save_path: Optional[str] = None
):
    """
    Plot riwayat pelatihan
    
    Args:
        history: Riwayat pelatihan dari model.fit()
        save_path: Path untuk menyimpan plot (opsional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot akurasi
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot loss
    axes[0, 1].plot(history.history['loss'], label='Train Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot presisi (jika tersedia)
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Train Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Plot recall (jika tersedia)
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Train Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training plot saved: {save_path}")
    
    plt.show()


def save_model_info(
    model: keras.Model,
    model_name: str,
    history: keras.callbacks.History,
    output_dir: str = 'trained_models'
):
    """
    Simpan informasi model dan riwayat pelatihan
    
    Args:
        model: Model terlatih
        model_name: Nama model
        history: Riwayat pelatihan
        output_dir: Direktori output
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Simpan arsitektur model sebagai JSON
    model_json = model.to_json()
    json_path = os.path.join(output_dir, f'{model_name}_architecture.json')
    with open(json_path, 'w') as f:
        f.write(model_json)
    
    # Simpan riwayat pelatihan
    history_dict = {
        'history': {k: [float(v) for v in vals] for k, vals in history.history.items()},
        'epochs': len(history.history['loss']),
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1])
    }
    
    history_path = os.path.join(output_dir, f'{model_name}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    # Simpan ringkasan model
    summary_path = os.path.join(output_dir, f'{model_name}_summary.txt')
    with open(summary_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    print(f"\n✅ Model info saved:")
    print(f"   - Architecture: {json_path}")
    print(f"   - History: {history_path}")
    print(f"   - Summary: {summary_path}")


def load_trained_model(
    model_path: str
) -> keras.Model:
    """
    Muat model terlatih dari file
    
    Args:
        model_path: Path ke model tersimpan (.keras atau .h5)
        
    Returns:
        Model yang dimuat
    """
    model = keras.models.load_model(model_path)
    print(f"✅ Model loaded from: {model_path}")
    return model


def print_model_summary(model: keras.Model):
    """Cetak ringkasan model secara detail"""
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*60)
    model.summary()
    print("="*60 + "\n")


def calculate_model_size(model: keras.Model) -> Dict:
    """
    Hitung ukuran model dan parameter
    
    Args:
        model: Model Keras
        
    Returns:
        Dictionary dengan info ukuran
    """
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) 
                            for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    # Perkirakan ukuran dalam MB (perkiraan kasar)
    size_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per param (float32)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'estimated_size_mb': size_mb
    }


def print_training_config(config: Dict):
    """Cetak konfigurasi pelatihan"""
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"\n{key.upper()}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    print("="*60 + "\n")


# Contoh penggunaan
if __name__ == '__main__':
    print("🧪 Testing Model Utils...\n")
    
    # Dapatkan config
    config = get_model_config()
    print("✅ Config loaded:")
    print(f"   Word input shape: {config['word']['input_shape']}")
    print(f"   Alphabet input shape: {config['alphabet']['input_shape']}")
    
    # Tes pembuatan callback
    print("\n✅ Creating callbacks...")
    cbs = create_callbacks('test_model')
    print(f"   Created {len(cbs)} callbacks")
    
    # Tes perhitungan ukuran model
    print("\n✅ Testing model size calculation...")
    dummy_model = keras.Sequential([
        keras.layers.Dense(64, input_shape=(126,)),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    size_info = calculate_model_size(dummy_model)
    print(f"   Total params: {size_info['total_params']:,}")
    print(f"   Estimated size: {size_info['estimated_size_mb']:.2f} MB")
    
    print("\n✅ All tests completed!")
