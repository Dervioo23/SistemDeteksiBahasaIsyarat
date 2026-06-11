import tensorflow as tf
from tensorflow import keras
from typing import Tuple, Optional
import numpy as np
layers = keras.layers

def build_cnn_bilstm_alphabet_model(
    input_shape: Tuple[int, int] = (1, 126),
    num_classes: int = 1,
    cnn_filters: list = [16, 32],
    cnn_kernel_size: int = 3,
    lstm_units: list = [16],
    dropout_rate: float = 0.3,
    dense_units: list = [16, 8],
    name: str = 'cnn_bilstm_alphabet'
) -> keras.Model:
    """
    Bangun model CNN-BiLSTM untuk pengenalan gestur alfabet
    
    Catatan: Gestur alfabet adalah pose statis (1 frame)
    Arsitektur lebih sederhana daripada model kata tetapi menjaga konsistensi
    
    Arsitektur:
    1. Input: (batch, 1 frame, 126 features)
    2. Layer CNN: Ekstrak fitur spasial
    3. Layer BiLSTM: Pemrosesan tambahan (lebih ringan dari model kata)
    4. Layer Dense: Klasifikasi
    5. Output: Probabilitas kelas gestur
    
    Args:
        input_shape: (frames, features) - misal, (1, 126)
        num_classes: Jumlah kelas (1 untuk biner, >1 untuk multi-kelas)
        cnn_filters: Daftar ukuran filter untuk layer CNN
        cnn_kernel_size: Ukuran kernel untuk CNN
        lstm_units: Daftar unit untuk layer LSTM
        dropout_rate: Tingkat dropout untuk regularisasi
        dense_units: Daftar unit untuk layer dense
        name: Nama model
        
    Returns:
        Model Keras yang telah dikompilasi
    """
    
    # Layer Input
    inputs = keras.layers.Input(shape=input_shape, name='input_static_pose')
    x = inputs
    
    print(f"Building {name}...")
    print(f"  Input shape: {input_shape}")
    print(f"  Note: Static pose model (1 frame)")
    
    # ========================================
    # 1. LAYER CNN (Ekstraksi Fitur Spasial)
    # ========================================
    
    for i, filters in enumerate(cnn_filters):
        x = keras.layers.Conv1D(
            filters=filters,
            kernel_size=cnn_kernel_size,
            padding='same',
            activation='relu',
            name=f'conv1d_{i+1}'
        )(x)
        
        x = keras.layers.BatchNormalization(name=f'bn_conv_{i+1}')(x)
        
        # Catatan: Lewati max pooling untuk pose statis dengan hanya 1 frame
        # untuk menjaga informasi
        
        x = keras.layers.Dropout(dropout_rate, name=f'dropout_conv_{i+1}')(x)
        
        print(f"  CNN Layer {i+1}: {filters} filters")
    
    # ========================================
    # 2. BIDIRECTIONAL LSTM (Pemrosesan Opsional)
    # ========================================
    # Meskipun statis, LSTM masih dapat mempelajari hubungan fitur
    
    for i, units in enumerate(lstm_units):
        return_sequences = (i < len(lstm_units) - 1)
        
        x = keras.layers.Bidirectional(
            keras.layers.LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=dropout_rate,
                recurrent_dropout=0.1,
                name=f'lstm_{i+1}'
            ),
            name=f'bidirectional_lstm_{i+1}'
        )(x)
        
        x = keras.layers.BatchNormalization(name=f'bn_lstm_{i+1}')(x)
        
        print(f"  BiLSTM Layer {i+1}: {units} units (bidirectional → {units*2} output)")
    
    # ========================================
    # 3. LAYER DENSE (Klasifikasi)
    # ========================================
    
    for i, units in enumerate(dense_units):
        x = keras.layers.Dense(
            units=units,
            activation='relu',
            name=f'dense_{i+1}'
        )(x)
        
        x = keras.layers.BatchNormalization(name=f'bn_dense_{i+1}')(x)
        x = keras.layers.Dropout(dropout_rate, name=f'dropout_dense_{i+1}')(x)
        
        print(f"  Dense Layer {i+1}: {units} units")
    
    # ========================================
    # 4. LAYER OUTPUT
    # ========================================
    
    if num_classes == 1:
        outputs = keras.layers.Dense(
            1,
            activation='sigmoid',
            name='output_binary'
        )(x)
        print(f"  Output: Binary classification (sigmoid)")
    else:
        outputs = layers.Dense(
            num_classes,
            activation='softmax',
            name='output_multiclass'
        )(x)
        print(f"  Output: Multi-class classification ({num_classes} classes, softmax)")
    
    # Buat model
    model = keras.models.Model(inputs=inputs, outputs=outputs, name=name)
    
    print(f"\n✅ Model '{name}' built successfully!")
    
    return model


def build_alphabet_model_simple(
    input_shape: Tuple[int, int] = (1, 126),
    num_classes: int = 1
) -> keras.Model:
    """
    Jaringan Dense Sederhana untuk pose alfabet statis
    Lebih ringan dan cepat untuk gestur statis
    
    Args:
        input_shape: (frames, features) - (1, 126)
        num_classes: Jumlah kelas
        
    Returns:
        Model Keras
    """
    inputs = layers.Input(shape=input_shape, name='input_static_pose')
    
    # Ratakan karena hanya 1 frame
    x = keras.layers.Flatten()(inputs)
    
    # Layer Dense
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    
    # Output
    if num_classes == 1:
        outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    else:
        outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.models.Model(inputs=inputs, outputs=outputs, name='alphabet_model_simple')
    
    print("✅ Simple alphabet model (Dense NN) built!")
    
    return model


def build_alphabet_model_cnn_only(
    input_shape: Tuple[int, int] = (1, 126),
    num_classes: int = 1
) -> keras.Model:
    """
    Model hanya CNN untuk alfabet
    Keseimbangan yang baik antara sederhana dan kompleks
    
    Args:
        input_shape: (frames, features)
        num_classes: Jumlah kelas
        
    Returns:
        Model Keras
    """
    inputs = layers.Input(shape=input_shape, name='input_static_pose')
    
    # Layer CNN
    x = keras.layers.Conv1D(64, 3, padding='same', activation='relu')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    
    x = keras.layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    
    # Ratakan
    x = keras.layers.Flatten()(x)
    
    # Layer Dense
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output
    if num_classes == 1:
        outputs = layers.Dense(1, activation='sigmoid')(x)
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.models.Model(inputs=inputs, outputs=outputs, name='alphabet_model_cnn')
    
    print("✅ CNN-only alphabet model built!")
    
    return model


# Contoh penggunaan
if __name__ == '__main__':
    print("\n" + "="*60)
    print("CNN-BiLSTM ALPHABET MODEL BUILDER")
    print("="*60 + "\n")
    
    # Tes model default
    print("🧪 Testing Default Model (CNN-BiLSTM)...\n")
    model = build_cnn_bilstm_alphabet_model(
        input_shape=(1, 126),
        num_classes=1,
        cnn_filters=[16, 32],
        lstm_units=[16],
        dense_units=[16, 8]
    )
    
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    model.summary()
    
    # Hitung ukuran model
    total_params = model.count_params()
    print(f"\n📊 Total Parameters: {total_params:,}")
    print(f"📏 Estimated Size: {(total_params * 4) / (1024 * 1024):.2f} MB")
    
    # Tes dengan data dummy
    print("\n🧪 Testing with dummy data...")
    dummy_input = np.random.rand(8, 1, 126).astype(np.float32)
    output = model.predict(dummy_input, verbose=0)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Tes model sederhana (Dense NN)
    print("\n🧪 Testing Simple Model (Dense NN)...\n")
    simple_model = build_alphabet_model_simple()
    simple_params = simple_model.count_params()
    print(f"📊 Simple Model Parameters: {simple_params:,}")
    
    # Tes model hanya CNN
    print("\n🧪 Testing CNN-only Model...\n")
    cnn_model = build_alphabet_model_cnn_only()
    cnn_params = cnn_model.count_params()
    print(f"📊 CNN Model Parameters: {cnn_params:,}")
    
    # Bandingkan ukuran model
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(f"CNN-BiLSTM:  {total_params:,} params")
    print(f"Dense NN:    {simple_params:,} params")
    print(f"CNN-only:    {cnn_params:,} params")
    print(f"\nRecommendation for static poses: Dense NN or CNN-only")
    print("="*60)
    
    print("\n✅ All model tests completed!")
