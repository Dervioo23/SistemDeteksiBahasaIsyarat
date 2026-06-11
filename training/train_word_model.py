import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras

from models import (
    build_cnn_bilstm_word_model,
    build_word_model_simple,
    compile_model,
    print_model_summary,
    calculate_model_size
)
from training.callbacks import create_standard_callbacks
from training.utils import set_random_seeds


# Definisi konstanta
MASK_VALUE = -10.0

def load_preprocessed_data(data_dir: str = 'preprocessed_data/words'):
    """
    Muat data yang telah diproses
    
    Args:
        data_dir: Direktori berisi data yang telah diproses
        
    Returns:
        Tuple dari (X_train, y_train, X_val, y_val, X_test, y_test, metadata)
    """
    print(f"\n{'='*60}")
    print("LOADING PREPROCESSED DATA")
    print(f"{'='*60}\n")
    
    # Muat data pelatihan
    X_train = np.load(os.path.join(data_dir, 'train_X.npy'))
    y_train = np.load(os.path.join(data_dir, 'train_y.npy'))
    
    # Muat data validasi
    X_val = np.load(os.path.join(data_dir, 'val_X.npy'))
    y_val = np.load(os.path.join(data_dir, 'val_y.npy'))
    
    # Muat data uji
    X_test = np.load(os.path.join(data_dir, 'test_X.npy'))
    y_test = np.load(os.path.join(data_dir, 'test_y.npy'))
    
    # Muat metadata
    with open(os.path.join(data_dir, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"✅ Data loaded successfully!")
    print(f"\n📊 Dataset shapes:")
    print(f"   Train: X={X_train.shape}, y={y_train.shape}")
    print(f"   Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"   Test:  X={X_test.shape}, y={y_test.shape}")
    
    print(f"\n📝 Metadata:")
    print(f"   Class names: {metadata['class_names']}")
    print(f"   Num classes: {metadata['num_classes']}")
    print(f"   Num features: {metadata['num_features']}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, metadata


def train_word_model(
    model_name: str = 'word_halo_model',
    model_type: str = 'default',
    data_dir: str = 'preprocessed_data/words',
    output_dir: str = 'trained_models',
    epochs: int = 50,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    seed: int = 42,
):
    """
    Latih model pengenalan gestur kata
    
    Args:
        model_name: Nama untuk model
        model_type: 'simple', 'default', atau 'deep'
        data_dir: Direktori dengan data yang telah diproses
        output_dir: Direktori untuk menyimpan model yang telah dilatih
        epochs: Jumlah epoch pelatihan
        batch_size: Ukuran batch untuk pelatihan
        learning_rate: Laju pembelajaran (learning rate)
    """
    
    print("\n" + "="*60)
    print("WORD GESTURE MODEL TRAINING")
    print("="*60)

    # Atur seed acak untuk reproduktifitas
    set_random_seeds(seed)

    # Muat data
    X_train, y_train, X_val, y_val, X_test, y_test, metadata = load_preprocessed_data(data_dir)
    
    # Tentukan bentuk input
    input_shape = (X_train.shape[1], X_train.shape[2])  # (frame, fitur)
    num_classes = 1  # Klasifikasi biner untuk POC (halo vs bukan-halo)

    # Peringatkan jika metadata menyarankan data multi-kelas
    if isinstance(metadata, dict) and metadata.get('num_classes', 1) != 1:
        print("\n⚠️  WARNING: This script trains a BINARY model (halo vs not-halo).")
        print("   Metadata num_classes = {0}, class_names = {1}".format(
            metadata.get('num_classes'), metadata.get('class_names')
        ))
        print("   For multi-class word models, use training/train_multiclass_word.py.")
    
    print(f"\n{'='*60}")
    print("BUILDING MODEL")
    print(f"{'='*60}\n")
    
    # Bangun model berdasarkan tipe
    if model_type == 'simple':
        print("Building SIMPLE model...")
        model = build_word_model_simple(
            input_shape=input_shape,
            num_classes=num_classes
        )
    elif model_type == 'default':
        print("Building DEFAULT model...")
        model = build_cnn_bilstm_word_model(
            input_shape=input_shape,
            num_classes=num_classes,
            mask_value=MASK_VALUE,
            cnn_filters=[32, 64],
            lstm_units=[64, 32],
            dropout_rate=0.3,
            dense_units=[32]
        )
    else:
        print("Unknown model type! Using default...")
        model = build_cnn_bilstm_word_model(
            input_shape=input_shape,
            num_classes=num_classes
        )
    
    # Kompilasi model
    print("\n📦 Compiling model...")
    model = compile_model(
        model=model,
        learning_rate=learning_rate,
        num_classes=num_classes
    )
    
    # Cetak ringkasan
    print_model_summary(model)
    
    # Ukuran model
    size_info = calculate_model_size(model)
    print(f"📊 Model Statistics:")
    print(f"   Total params: {size_info['total_params']:,}")
    print(f"   Trainable params: {size_info['trainable_params']:,}")
    print(f"   Estimated size: {size_info['estimated_size_mb']:.2f} MB")
    
    # Buat callback
    print(f"\n📋 Creating callbacks...")
    callbacks = create_standard_callbacks(
        model_name=model_name,
        output_dir=output_dir,
        monitor='val_accuracy',
        patience=10
    )
    print(f"   Created {len(callbacks)} callbacks")
    
    # Pelatihan
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}")
    print(f"   Model: {model_name}")
    print(f"   Type: {model_type}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"{'='*60}\n")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0  # Callback menangani pencetakan
    )
    
    # Evaluasi pada set uji
    print(f"\n{'='*60}")
    print("FINAL EVALUATION ON TEST SET")
    print(f"{'='*60}\n")
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)[:2]
    
    print(f"📊 Test Results:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Prediksi pada set uji
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int)
    
    # Metrik
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Dapatkan kelas unik
    unique_classes = sorted(list(set(y_test.flatten().tolist() + y_pred.flatten().tolist())))
    
    print(f"\n📊 Classification Report:")
    try:
        if len(unique_classes) == 1:
            # Kelas tunggal - laporan sederhana
            print(f"   All samples belong to class: {unique_classes[0]}")
            print(f"   Accuracy: {test_acc:.4f}")
        else:
            # Laporan multi-kelas
            target_names = ['Not Halo', 'Halo'] if len(unique_classes) == 2 else None
            print(classification_report(y_test, y_pred, target_names=target_names))
    except Exception as e:
        print(f"   Accuracy: {test_acc:.4f}")
        print(f"   (Classification report skipped: {e})")
    
    print(f"\n📊 Confusion Matrix:")
    try:
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape[0] == 1:
            print(f"   Single class detected: {unique_classes[0]}")
            print(f"   Correct predictions: {cm[0][0]}")
        else:
            print(f"   Predicted →")
            print(f"   {'True ↓':<12} Not Halo   Halo")
            print(f"   {'Not Halo':<12} {cm[0][0]:<10} {cm[0][1]:<10}")
            print(f"   {'Halo':<12} {cm[1][0]:<10} {cm[1][1]:<10}")
    except Exception as e:
        print(f"   (Confusion matrix skipped: {e})")
    
    # Simpan model akhir secara eksplisit
    final_model_path = os.path.join(output_dir, f'{model_name}_final.keras')
    model.save(final_model_path)
    print(f"\n✅ Final model saved: {final_model_path}")
    
    # Simpan ringkasan pelatihan
    summary_path = os.path.join(output_dir, f'{model_name}_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Type: {model_type}\n")
        f.write(f"Input shape: {input_shape}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Validation samples: {len(X_val)}\n")
        f.write(f"Test samples: {len(X_test)}\n")
        f.write(f"\nFinal Results:\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"\nParameters: {size_info['total_params']:,}\n")
        f.write(f"Model size: {size_info['estimated_size_mb']:.2f} MB\n")
    
    print(f"✅ Training summary saved: {summary_path}")
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}\n")
    
    return model, history


def main():
    """Fungsi utama"""

    import argparse

    parser = argparse.ArgumentParser(
        description="Train binary word model (halo vs not-halo).",
    )
    parser.add_argument("--model_name", type=str, default="word_halo_model")
    parser.add_argument(
        "--model_type",
        type=str,
        default="default",
        choices=["simple", "default", "deep"],
    )
    parser.add_argument("--data_dir", type=str, default="preprocessed_data/words")
    parser.add_argument("--output_dir", type=str, default="trained_models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    config = {
        "model_name": args.model_name,
        "model_type": args.model_type,
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
    }

    # Periksa apakah data ada
    if not os.path.exists(config["data_dir"]):
        print(f"❌ Error: Data directory not found: {config['data_dir']}")
        print("   Please run preprocessing first:")
        print("   python -m run_preprocessing")
        return

    # Latih model
    model, history = train_word_model(**config)

    print("\n🎉 Training pipeline completed!")
    print(f"   Model saved in: {config['output_dir']}/")
    print(f"   Logs saved in: logs/{config['model_name']}/")
    print("\n💡 To view training logs:")
    print("   tensorboard --logdir=logs")


if __name__ == '__main__':
    main()
