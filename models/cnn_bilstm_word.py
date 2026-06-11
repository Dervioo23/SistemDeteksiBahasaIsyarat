import tensorflow as tf
from tensorflow import keras
from typing import Tuple, Optional
import numpy as np

layers = keras.layers
models = keras.models

def build_cnn_bilstm_word_model(
    input_shape: Tuple[int, int] = (45, 126),
    num_classes: int = 1,
    mask_value: float = 0.0,
    cnn_filters: list = [32, 64],
    cnn_kernel_size: int = 3,
    lstm_units: list = [64, 32],
    dropout_rate: float = 0.3,
    dense_units: list = [32],
    name: str = 'cnn_bilstm_word'
) -> keras.Model:
    """
    Bangun model CNN-BiLSTM untuk pengenalan gestur kata
    
    Arsitektur:
    1. Input: (batch, 45 frames, 126 features)
    2. Layer CNN: Ekstrak fitur spasial dari landmarks
    3. Layer BiLSTM: Tangkap pola temporal
    4. Layer Dense: Klasifikasi
    5. Output: Probabilitas kelas gestur
    
    Args:
        input_shape: (frames, features) - misal, (45, 126)
        num_classes: Jumlah kelas (1 untuk biner, >1 untuk multi-kelas)
        mask_value: Nilai yang digunakan untuk padding (untuk diabaikan)
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
    inputs = keras.layers.Input(shape=input_shape, name='input_sequence')
    x = inputs
    mask_tensor = keras.layers.Lambda(
        lambda v: tf.cast(
            tf.math.reduce_any(tf.not_equal(v, mask_value), axis=-1, keepdims=True),
            tf.float32
        ),
        name='sequence_mask'
    )(inputs)
    
    # ========================================
    # 1. LAYER CNN (Ekstraksi Fitur Spasial)
    # ========================================
    # Proses setiap frame secara independen untuk mengekstrak fitur spasial
    
    print(f"Building {name}...")
    print(f"  Input shape: {input_shape}")
    
    for i, filters in enumerate(cnn_filters):
        # Konvolusi 1D melintasi fitur (dalam setiap frame)
        x = keras.layers.Conv1D(
            filters=filters,
            kernel_size=cnn_kernel_size,
            padding='same',
            activation='relu',
            name=f'conv1d_{i+1}'
        )(x)
        
        # Normalisasi batch untuk stabilitas pelatihan
        x = keras.layers.BatchNormalization(name=f'bn_conv_{i+1}')(x)
        
        # Max pooling untuk mengurangi dimensi
        x = keras.layers.MaxPooling1D(
            pool_size=2,
            padding='same',
            name=f'maxpool_{i+1}'
        )(x)
        mask_tensor = keras.layers.MaxPooling1D(
            pool_size=2,
            padding='same',
            name=f'mask_maxpool_{i+1}'
        )(mask_tensor)
        
        # Dropout untuk regularisasi
        x = keras.layers.Dropout(dropout_rate, name=f'dropout_conv_{i+1}')(x)
        
        print(f"  CNN Layer {i+1}: {filters} filters")
    
    # ========================================
    # 2. LAYER BIDIRECTIONAL LSTM (Penangkapan Pola Temporal)
    # ========================================
    # Tangkap ketergantungan temporal di kedua arah maju dan mundur
    
    mask_lstm = keras.layers.Lambda(
        lambda m: tf.squeeze(tf.cast(m > 0.5, tf.bool), axis=-1),
        name='lstm_mask'
    )(mask_tensor)

    for i, units in enumerate(lstm_units):
        x = keras.layers.Bidirectional(
            keras.layers.LSTM(
                units=units,
                return_sequences=True,
                dropout=dropout_rate,
                recurrent_dropout=0.1,
                name=f'lstm_{i+1}'
            ),
            name=f'bidirectional_lstm_{i+1}'
        )(x, mask=mask_lstm)
        
        # Normalisasi batch
        x = keras.layers.BatchNormalization(name=f'bn_lstm_{i+1}')(x)
        
        print(f"  BiLSTM Layer {i+1}: {units} units (bidirectional → {units*2} output)")

    # Mekanisme atensi di atas output BiLSTM
    attn_scores = keras.layers.Dense(
        1,
        activation='tanh',
        name='attention_scores'
    )(x)
    attn_scores = keras.layers.Lambda(
        lambda s: tf.squeeze(s, axis=-1),
        name='attention_scores_squeezed'
    )(attn_scores)
    attn_scores = keras.layers.Lambda(
        lambda inputs: tf.where(
            inputs[1],
            inputs[0],
            tf.ones_like(inputs[0]) * (-1e9)
        ),
        name='attention_scores_masked'
    )([attn_scores, mask_lstm])
    attn_weights = keras.layers.Activation(
        'softmax',
        name='attention_weights'
    )(attn_scores)
    x = keras.layers.Lambda(
        lambda inputs: tf.reduce_sum(
            inputs[0] * tf.expand_dims(inputs[1], axis=-1),
            axis=1
        ),
        name='attention_weighted_sum'
    )([x, attn_weights])
    
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
        # Klasifikasi biner
        outputs = keras.layers.Dense(
            1,
            activation='sigmoid',
            name='output_binary'
        )(x)
        print(f"  Output: Binary classification (sigmoid)")
    else:
        # Klasifikasi multi-kelas
        outputs = keras.layers.Dense(
            num_classes,
            activation='softmax',
            name='output_multiclass'
        )(x)
        print(f"  Output: Multi-class classification ({num_classes} classes, softmax)")
    
    # Buat model
    model = keras.models.Model(inputs=inputs, outputs=outputs, name=name)
    
    print(f"\n✅ Model '{name}' built successfully!")
    
    return model


def build_word_model_simple(
    input_shape: Tuple[int, int] = (45, 126),
    num_classes: int = 1
) -> keras.Model:
    """
    Versi sederhana untuk dataset kecil - ANTI-OVERFITTING
    Model yang jauh lebih ringan dengan regularisasi berat untuk mencegah overfitting
    
    Target: <3000 parameter (vs 21K+ di model kompleks)
    
    Args:
        input_shape: (frames, features)
        num_classes: Jumlah kelas
        
    Returns:
        Model Keras yang dioptimalkan untuk dataset kecil
    """
    inputs = keras.layers.Input(shape=input_shape, name='input_sequence')
    mask_tensor = keras.layers.Lambda(
        lambda v: tf.cast(
            tf.math.reduce_any(tf.not_equal(v, -10.0), axis=-1, keepdims=True),
            tf.float32
        ),
        name='simple_sequence_mask'
    )(inputs)
    
    # Satu layer CNN ringan
    x = keras.layers.Conv1D(16, 3, padding='same', activation='relu')(inputs)  # Dikurangi dari 32 ke 16
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2, padding='same')(x)
    mask_tensor = keras.layers.MaxPooling1D(2, padding='same', name='simple_mask_maxpool')(mask_tensor)
    x = keras.layers.Dropout(0.6)(x)  # Ditingkatkan dari 0.3 ke 0.6 untuk regularisasi berat
    
    # Satu BiLSTM kecil (jauh lebih kecil dari sebelumnya)
    mask_lstm = keras.layers.Lambda(
        lambda m: tf.squeeze(tf.cast(m > 0.5, tf.bool), axis=-1),
        name='simple_lstm_mask'
    )(mask_tensor)
    x = keras.layers.Bidirectional(keras.layers.LSTM(16, dropout=0.5, recurrent_dropout=0.3))(x, mask=mask_lstm)  # Dikurangi dari 32 ke 16
    x = keras.layers.Dropout(0.7)(x)  # Dropout berat setelah LSTM
    
    # Satu layer dense kecil dengan regularisasi kuat
    x = keras.layers.Dense(16, activation='relu', 
                    kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)  # Ditambahkan regularisasi L2
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.6)(x)  # Dropout berat
    
    # Output dengan regularisasi L2
    if num_classes == 1:
        outputs = keras.layers.Dense(1, activation='sigmoid',
                              kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    else:
        outputs = keras.layers.Dense(num_classes, activation='softmax',
                              kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    
    model = keras.models.Model(inputs=inputs, outputs=outputs, name='word_model_anti_overfit')
    
    # Hitung parameter untuk memverifikasi cukup kecil
    param_count = model.count_params()
    print(f"✅ Anti-overfitting word model built!")
    print(f"📊 Parameters: {param_count:,} (Target: <3,000)")
    if param_count > 3000:
        print(f"⚠️  WARNING: Still too complex for small dataset!")
    else:
        print(f"✅ Good parameter count for small dataset")
    
    return model


def build_word_model_deep(
    input_shape: Tuple[int, int] = (45, 126),
    num_classes: int = 1
) -> keras.Model:
    """
    Model lebih dalam untuk akurasi lebih baik (membutuhkan lebih banyak data)
    
    Args:
        input_shape: (frames, features)
        num_classes: Jumlah kelas
        
    Returns:
        Model Keras
    """
    inputs = keras.layers.Input(shape=input_shape, name='input_sequence')
    mask_tensor = keras.layers.Lambda(
        lambda v: tf.cast(
            tf.math.reduce_any(tf.not_equal(v, -10.0), axis=-1, keepdims=True),
            tf.float32
        ),
        name='deep_sequence_mask'
    )(inputs)
    
    # Deep CNN
    x = keras.layers.Conv1D(64, 3, padding='same', activation='relu')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2, padding='same')(x)
    mask_tensor = keras.layers.MaxPooling1D(2, padding='same', name='deep_mask_maxpool_1')(mask_tensor)
    x = keras.layers.Dropout(0.2)(x)
    
    x = keras.layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2, padding='same')(x)
    mask_tensor = keras.layers.MaxPooling1D(2, padding='same', name='deep_mask_maxpool_2')(mask_tensor)
    x = keras.layers.Dropout(0.2)(x)
    
    x = keras.layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    
    # Stacked BiLSTM
    mask_lstm = keras.layers.Lambda(
        lambda m: tf.squeeze(tf.cast(m > 0.5, tf.bool), axis=-1),
        name='deep_lstm_mask'
    )(mask_tensor)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(64, return_sequences=True, dropout=0.3),
        name='deep_bidirectional_lstm_1'
    )(x, mask=mask_lstm)
    x = keras.layers.BatchNormalization(name='deep_bn_lstm_1')(x)
    
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(32, return_sequences=True, dropout=0.3),
        name='deep_bidirectional_lstm_2'
    )(x, mask=mask_lstm)
    x = keras.layers.BatchNormalization(name='deep_bn_lstm_2')(x)

    # Atensi di atas output BiLSTM
    deep_attn_scores = keras.layers.Dense(
        1,
        activation='tanh',
        name='deep_attention_scores'
    )(x)
    deep_attn_scores = keras.layers.Lambda(
        lambda s: tf.squeeze(s, axis=-1),
        name='deep_attention_scores_squeezed'
    )(deep_attn_scores)
    deep_attn_scores = keras.layers.Lambda(
        lambda inputs: tf.where(
            inputs[1],
            inputs[0],
            tf.ones_like(inputs[0]) * (-1e9)
        ),
        name='deep_attention_scores_masked'
    )([deep_attn_scores, mask_lstm])
    deep_attn_weights = keras.layers.Activation(
        'softmax',
        name='deep_attention_weights'
    )(deep_attn_scores)
    x = keras.layers.Lambda(
        lambda inputs: tf.reduce_sum(
            inputs[0] * tf.expand_dims(inputs[1], axis=-1),
            axis=1
        ),
        name='deep_attention_weighted_sum'
    )([x, deep_attn_weights])
    
    # Layer Dense
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)
    
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dropout(0.4)(x)
    
    # Output
    if num_classes == 1:
        outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    else:
        outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.models.Model(inputs=inputs, outputs=outputs, name='word_model_deep')
    
    print("✅ Deep word model built!")
    
    return model


# Contoh penggunaan
if __name__ == '__main__':
    print("\n" + "="*60)
    print("CNN-BiLSTM WORD MODEL BUILDER")
    print("="*60 + "\n")
    
    # Tes model default
    print("🧪 Testing Default Model...\n")
    model = build_cnn_bilstm_word_model(
        input_shape=(45, 126),
        num_classes=1,
        cnn_filters=[32, 64],
        lstm_units=[64, 32],
        dense_units=[32]
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
    dummy_input = np.random.rand(8, 45, 126).astype(np.float32)
    output = model.predict(dummy_input, verbose=0)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Tes model sederhana
    print("\n🧪 Testing Simple Model...\n")
    simple_model = build_word_model_simple()
    simple_params = simple_model.count_params()
    print(f"📊 Simple Model Parameters: {simple_params:,}")
    
    # Tes model dalam
    print("\n🧪 Testing Deep Model...\n")
    deep_model = build_word_model_deep()
    deep_params = deep_model.count_params()
    print(f"📊 Deep Model Parameters: {deep_params:,}")
    
    print("\n✅ All model tests completed!")
