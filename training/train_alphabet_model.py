import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras

from models import (
    build_alphabet_model_simple,
    build_alphabet_model_cnn_only,
    compile_model,
    print_model_summary,
    calculate_model_size
)
from training.callbacks import create_standard_callbacks
from training.utils import set_random_seeds


def load_preprocessed_data(data_dir: str = 'preprocessed_data/alphabet'):
    """Muat data alfabet yang telah diproses"""
    print(f"\n{'='*60}")
    print("LOADING PREPROCESSED DATA")
    print(f"{'='*60}\n")
    
    # Muat data
    X_train = np.load(os.path.join(data_dir, 'train_X.npy'))
    y_train = np.load(os.path.join(data_dir, 'train_y.npy'))
    X_val = np.load(os.path.join(data_dir, 'val_X.npy'))
    y_val = np.load(os.path.join(data_dir, 'val_y.npy'))
    X_test = np.load(os.path.join(data_dir, 'test_X.npy'))
    y_test = np.load(os.path.join(data_dir, 'test_y.npy'))
    
    with open(os.path.join(data_dir, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"✅ Data loaded successfully!")
    print(f"\n📊 Dataset shapes:")
    print(f"   Train: X={X_train.shape}, y={y_train.shape}")
    print(f"   Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"   Test:  X={X_test.shape}, y={y_test.shape}")
    
    print(f"\n📝 Metadata:")
    print(f"   Class names: {metadata['class_names']}")
    print(f"   Note: Static pose (1 frame per sample)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, metadata


def train_alphabet_model(
    model_name: str = 'alphabet_C_model',
    model_type: str = 'simple',
    data_dir: str = 'preprocessed_data/alphabet',
    output_dir: str = 'trained_models',
    epochs: int = 50,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    seed: int = 42,
):
    """
    Latih model pengenalan gestur alfabet
    
    Args:
        model_name: Nama untuk model
        model_type: 'simple' (Dense NN) atau 'cnn' (CNN-only)
        data_dir: Direktori dengan data yang telah diproses
        output_dir: Direktori untuk menyimpan model yang telah dilatih
        epochs: Jumlah epoch pelatihan
        batch_size: Ukuran batch untuk pelatihan
        learning_rate: Laju pembelajaran (learning rate)
    """
    
    print("\n" + "="*60)
    print("ALPHABET GESTURE MODEL TRAINING")
    print("="*60)

    # Atur seed acak untuk reproduktifitas
    set_random_seeds(seed)

    # Muat data
    X_train, y_train, X_val, y_val, X_test, y_test, metadata = load_preprocessed_data(data_dir)
    
    # Bentuk input
    input_shape = (X_train.shape[1], X_train.shape[2])  # (1, 126)
    num_classes = 1  # Klasifikasi biner (C vs bukan-C)

    # Peringatkan jika metadata menyarankan data multi-kelas
    if isinstance(metadata, dict) and metadata.get('num_classes', 1) != 1:
        print("\n⚠️  WARNING: This script trains a BINARY model (C vs not-C).")
        print("   Metadata num_classes = {0}, class_names = {1}".format(
            metadata.get('num_classes'), metadata.get('class_names')
        ))
        print("   For multi-class alphabet models, use training/train_multiclass_alphabet.py.")
    
    print(f"\n{'='*60}")
    print("BUILDING MODEL")
    print(f"{'='*60}\n")
    
    # Bangun model
    if model_type == 'simple':
        print("Building SIMPLE model (Dense NN - Recommended for static poses)...")
        model = build_alphabet_model_simple(
            input_shape=input_shape,
            num_classes=num_classes
        )
    elif model_type == 'cnn':
        print("Building CNN-only model...")
        model = build_alphabet_model_cnn_only(
            input_shape=input_shape,
            num_classes=num_classes
        )
    else:
        print("Unknown model type! Using simple...")
        model = build_alphabet_model_simple(
            input_shape=input_shape,
            num_classes=num_classes
        )
    
    # Kompilasi
    print("\n📦 Compiling model...")
    model = compile_model(
        model=model,
        learning_rate=learning_rate,
        num_classes=num_classes
    )
    
    # Ringkasan
    print_model_summary(model)
    
    size_info = calculate_model_size(model)
    print(f"📊 Model Statistics:")
    print(f"   Total params: {size_info['total_params']:,}")
    print(f"   Trainable params: {size_info['trainable_params']:,}")
    print(f"   Estimated size: {size_info['estimated_size_mb']:.2f} MB")
    
    # Callback
    print(f"\n📋 Creating callbacks...")
    callbacks = create_standard_callbacks(
        model_name=model_name,
        output_dir=output_dir,
        monitor='val_accuracy',
        patience=10
    )
    
    # Pelatihan
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}")
    print(f"   Model: {model_name}")
    print(f"   Type: {model_type}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"{'='*60}\n")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )
    
    # Evaluasi
    print(f"\n{'='*60}")
    print("FINAL EVALUATION ON TEST SET")
    print(f"{'='*60}\n")
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)[:2]
    
    print(f"📊 Test Results:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Prediksi
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int)
    
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Dapatkan kelas unik
    unique_classes = sorted(list(set(y_test.flatten().tolist() + y_pred.flatten().tolist())))
    
    print(f"\n📊 Classification Report:")
    try:
        if len(unique_classes) == 1:
            print(f"   All samples belong to class: {unique_classes[0]}")
            print(f"   Accuracy: {test_acc:.4f}")
        else:
            target_names = ['Not C', 'C'] if len(unique_classes) == 2 else None
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
            print(f"   {'True ↓':<12} Not C      C")
            print(f"   {'Not C':<12} {cm[0][0]:<10} {cm[0][1]:<10}")
            print(f"   {'C':<12} {cm[1][0]:<10} {cm[1][1]:<10}")
    except Exception as e:
        print(f"   (Confusion matrix skipped: {e})")
    
    # Simpan
    final_model_path = os.path.join(output_dir, f'{model_name}_final.keras')
    model.save(final_model_path)
    print(f"\n✅ Final model saved: {final_model_path}")
    
    # Ringkasan
    summary_path = os.path.join(output_dir, f'{model_name}_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Type: {model_type}\n")
        f.write(f"Input shape: {input_shape}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Parameters: {size_info['total_params']:,}\n")
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED!")
    print(f"{'='*60}\n")
    
    return model, history


def main():
    """Fungsi utama"""

    import argparse

    parser = argparse.ArgumentParser(
        description="Train binary alphabet model (C vs not-C).",
    )
    parser.add_argument("--model_name", type=str, default="alphabet_C_model")
    parser.add_argument(
        "--model_type",
        type=str,
        default="simple",
        choices=["simple", "cnn"],
    )
    parser.add_argument("--data_dir", type=str, default="preprocessed_data/alphabet")
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

    if not os.path.exists(config["data_dir"]):
        print(f"❌ Error: Data not found: {config['data_dir']}")
        print("   Run preprocessing first!")
        return

    model, history = train_alphabet_model(**config)

    print("\n🎉 Training completed!")
    print(f"   Model: {config['output_dir']}/")
    print(f"   Logs: logs/{config['model_name']}/")


if __name__ == '__main__':
    main()
