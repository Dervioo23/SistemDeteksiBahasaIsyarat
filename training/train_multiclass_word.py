import numpy as np
import os
import json
import pickle
from datetime import datetime

from models.multiclass_models import build_multiclass_word_model
from models.model_utils import create_callbacks, plot_training_history
from training.callbacks import create_anti_overfitting_callbacks
from training.utils import set_random_seeds
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def train_multiclass_word_model(
    model_name: str = 'multiclass_word_model',
    model_type: str = 'simple',      # Diubah dari 'default' ke 'simple' untuk anti-overfitting
    data_dir: str = 'preprocessed_data/words', 
    output_dir: str = 'trained_models',
    epochs: int = 50,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    anti_overfit: bool = True,       # Parameter baru untuk mode anti-overfitting
    seed: int = 42,
):
    """
    Latih model pengenalan gestur kata multi-kelas
    
    Args:
        model_name: Nama untuk menyimpan model
        model_type: 'simple', 'default', atau 'deep'
        data_dir: Direktori berisi data yang telah diproses
        output_dir: Direktori untuk menyimpan model yang telah dilatih
        epochs: Jumlah epoch pelatihan
        batch_size: Ukuran batch untuk pelatihan
        learning_rate: Laju pembelajaran untuk optimizer
    """
    
    print("\n" + "="*60)
    print("MULTI-CLASS WORD GESTURE MODEL TRAINING")
    print("="*60)

    # Atur seed acak untuk reproduktifitas
    set_random_seeds(seed)

    # Buat direktori output
    os.makedirs(output_dir, exist_ok=True)
    
    # Muat data
    print("\n" + "="*60)
    print("LOADING PREPROCESSED DATA")
    print("="*60)
    
    try:
        X_train = np.load(os.path.join(data_dir, 'train_X.npy'))
        y_train = np.load(os.path.join(data_dir, 'train_y.npy'))
        X_val = np.load(os.path.join(data_dir, 'val_X.npy'))
        y_val = np.load(os.path.join(data_dir, 'val_y.npy'))
        X_test = np.load(os.path.join(data_dir, 'test_X.npy'))
        y_test = np.load(os.path.join(data_dir, 'test_y.npy'))
        
        # Muat metadata
        with open(os.path.join(data_dir, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        class_names = metadata['class_names']
        num_classes = len(class_names)
        
        print("\n✅ Data loaded successfully!")
        print(f"\n📊 Dataset shapes:")
        print(f"   Train: X={X_train.shape}, y={y_train.shape}")
        print(f"   Val:   X={X_val.shape}, y={y_val.shape}")
        print(f"   Test:  X={X_test.shape}, y={y_test.shape}")
        print(f"\n📝 Classes ({num_classes}): {class_names}")
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] Error loading data: {e}")
        print("\n[INFO] Please run preprocessing first:")
        print("   python run_preprocessing.py")
        return None, None
    
    # Bangun model
    print("\n" + "="*60)
    print("BUILDING MODEL")
    print("="*60)
    
    input_shape = X_train.shape[1:]
    print(f"\nInput shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Model type: {model_type}")
    
    model = build_multiclass_word_model(
        input_shape=input_shape,
        num_classes=num_classes,
        model_type=model_type
    )
    
    print(f"\n✅ Multi-class word model built!")
    
    # Kompilasi model
    print(f"\n📦 Compiling model...")
    model.compile(
        loss='sparse_categorical_crossentropy',  # untuk label integer
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )
    
    # Cetak ringkasan
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*60)
    model.summary()
    
    # Statistik model
    total_params = model.count_params()
    model_size_mb = total_params * 4 / (1024 * 1024)
    print(f"\n📊 Model Statistics:")
    print(f"   Total params: {total_params:,}")
    print(f"   Estimated size: {model_size_mb:.2f} MB")
    
    # Buat callback (anti-overfitting atau standar)
    print(f"\n📋 Creating callbacks...")
    if anti_overfit:
        print("🛡️  Using ANTI-OVERFITTING callbacks for small dataset")
        callbacks = create_anti_overfitting_callbacks(
            model_name=model_name,
            output_dir=output_dir,
            monitor='val_accuracy',
            early_patience=3,    # Penghentian awal yang sangat agresif
            lr_patience=2        # Pengurangan LR cepat
        )
        # Peringatkan tentang jumlah parameter vs ukuran dataset
        train_samples = X_train.shape[0] 
        param_ratio = total_params / train_samples
        print(f"⚠️  Model complexity check:")
        print(f"   Parameters: {total_params:,}")
        print(f"   Training samples: {train_samples}")
        print(f"   Ratio: {param_ratio:.1f} params per sample")
        if param_ratio > 50:
            print(f"🚨 HIGH OVERFITTING RISK! Ratio > 50")
        elif param_ratio > 20:
            print(f"⚠️  Moderate overfitting risk. Ratio > 20")
        else:
            print(f"✅ Good ratio for generalization")
    else:
        callbacks = create_callbacks(
            model_name=model_name,
            output_dir=output_dir,
            monitor='val_accuracy',
            patience=10
        )
    
    # Pelatihan
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    print(f"   Model: {model_name}")
    print(f"   Type: {model_type}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print("="*60 + "\n")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    
    # Plot riwayat pelatihan
    print(f"\n📊 Plotting training history...")
    history_plot_path = os.path.join(output_dir, f'{model_name}_history.png')
    plot_training_history(history, save_path=history_plot_path)
    print(f"✅ Training history plot saved: {history_plot_path}")
    
    # Evaluasi
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n📊 Test Results:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Prediksi
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Laporan klasifikasi
    print(f"\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion matrix
    print(f"\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = os.path.join(output_dir, f'{model_name}_confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"✅ Confusion matrix saved: {cm_path}")
    
    # Simpan model akhir
    final_model_path = os.path.join(output_dir, f'{model_name}_final.keras')
    model.save(final_model_path)
    print(f"\n✅ Final model saved: {final_model_path}")
    
    # Simpan metadata (termasuk metode normalisasi yang digunakan selama preprocessing)
    normalization_method = metadata.get('config', {}).get('normalization_method', 'full') if isinstance(metadata, dict) else 'full'
    model_metadata = {
        'model_name': model_name,
        'model_type': model_type,
        'num_classes': num_classes,
        'class_names': class_names,
        'input_shape': list(input_shape),
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_params': int(total_params),
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'normalization_method': normalization_method,
    }
    
    metadata_path = os.path.join(output_dir, f'{model_name}_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=2)
    print(f"✅ Model metadata saved: {metadata_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\n📦 Trained model: {final_model_path}")
    print(f"📊 Test accuracy: {test_acc*100:.2f}%")
    print(f"📝 Classes: {', '.join(class_names)}")
    
    return model, history


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train multi-class word model')
    parser.add_argument('--model_name', type=str, default='multiclass_word_model',
                       help='Model name for saving')
    parser.add_argument('--model_type', type=str, default='default',
                       choices=['simple', 'default', 'deep', 'baseline'],
                       help='Model architecture type')
    parser.add_argument('--data_dir', type=str, default='preprocessed_data/words',
                       help='Directory containing preprocessed data')
    parser.add_argument('--output_dir', type=str, default='trained_models',
                       help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    train_multiclass_word_model(
        model_name=args.model_name,
        model_type=args.model_type,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
