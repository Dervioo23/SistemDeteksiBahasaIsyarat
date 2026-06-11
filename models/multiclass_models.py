import tensorflow as tf
from tensorflow import keras


def build_multiclass_word_model(
    input_shape: tuple,
    num_classes: int,
    model_type: str = 'default'
) -> keras.Model:
    """
    Bangun model kata multi-kelas
    
    Args:
        input_shape: (sequence_length, features) misal, (45, 126)
        num_classes: Jumlah kelas kata
        model_type: 'simple', 'default', atau 'deep'
    
    Returns:
        Model Keras dengan output SOFTMAX
    """
    
    if model_type == 'simple':
        return _build_simple_word(input_shape, num_classes)
    elif model_type == 'default':
        return _build_default_word(input_shape, num_classes)
    elif model_type == 'deep':
        return _build_deep_word(input_shape, num_classes)
    elif model_type == 'baseline':
        return _build_baseline_lstm_word(input_shape, num_classes)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _build_simple_word(input_shape: tuple, num_classes: int) -> keras.Model:
    """ANTI-OVERFITTING Model kata multi-kelas sederhana untuk dataset kecil"""
    
    inputs = keras.Input(shape=input_shape)
    
    # CNN ultra-ringan (32→8 filter)
    x = keras.layers.Conv1D(8, 3, activation='relu', padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Dropout(0.7)(x)  # Dropout berat 0.3→0.7
    
    # BiLSTM kecil (32→8 unit)
    x = keras.layers.Bidirectional(keras.layers.LSTM(8, dropout=0.5, recurrent_dropout=0.3))(x)
    x = keras.layers.Dropout(0.8)(x)  # Dropout ekstrem
    
    # Dense kecil dengan regularisasi L2 (64→16 unit)
    x = keras.layers.Dense(16, activation='relu', 
                    kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.7)(x)
    
    # SOFTMAX untuk multi-kelas dengan L2!
    outputs = keras.layers.Dense(num_classes, activation='softmax',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    
    return keras.Model(inputs=inputs, outputs=outputs, name='multiclass_word_anti_overfit')


def _build_default_word(input_shape: tuple, num_classes: int) -> keras.Model:
    """ANTI-OVERFITTING Model default - Disederhanakan secara drastis untuk dataset kecil"""
    
    inputs = keras.Input(shape=input_shape)
    
    # Satu CNN kecil (16→8 filter)
    x = keras.layers.Conv1D(8, 3, activation='relu', padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Dropout(0.6)(x)
    
    # Satu BiLSTM kecil (16→12 unit)
    x = keras.layers.Bidirectional(keras.layers.LSTM(12, dropout=0.5, recurrent_dropout=0.3))(x)
    x = keras.layers.Dropout(0.7)(x)
    
    # Dense kecil dengan regularisasi berat (32→16)
    x = keras.layers.Dense(16, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.6)(x)
    
    # SOFTMAX dengan L2!
    outputs = keras.layers.Dense(num_classes, activation='softmax',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    
    return keras.Model(inputs=inputs, outputs=outputs, name='multiclass_word_anti_overfit_default')


def _build_deep_word(input_shape: tuple, num_classes: int) -> keras.Model:
    """Model kata multi-kelas dalam (untuk 20+ kelas)"""
    
    inputs = keras.Input(shape=input_shape)
    
    x = keras.layers.Conv1D(64, 5, activation='relu', padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(64, 5, activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Dropout(0.3)(x)
    
    x = keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Dropout(0.3)(x)
    
    x = keras.layers.Conv1D(256, 3, activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Dropout(0.4)(x)
    
    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True))(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(32))(x)
    x = keras.layers.Dropout(0.3)(x)
    
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.5)(x)
    
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)
    
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    
    # SOFTMAX untuk multi-kelas!
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs=inputs, outputs=outputs, name='multiclass_word_deep')


def _build_baseline_lstm_word(input_shape: tuple, num_classes: int) -> keras.Model:
    """Baseline Model: Standard LSTM without CNN for comparison tasks"""
    
    inputs = keras.Input(shape=input_shape)
    
    # Standard LSTM (No CNN, No Bidirectional)
    x = keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2)(inputs)
    
    # Dense Layer
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    
    # SOFTMAX for multi-class!
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs=inputs, outputs=outputs, name='baseline_lstm_word')


def build_multiclass_alphabet_model(
    input_shape: tuple,
    num_classes: int,
    model_type: str = 'default'
) -> keras.Model:
    """
    Bangun model alfabet multi-kelas
    
    Args:
        input_shape: (1, 126) untuk pose statis
        num_classes: Jumlah kelas alfabet (26 untuk A-Z)
        model_type: 'simple' atau 'default'
    
    Returns:
        Model Keras dengan output SOFTMAX
    """
    
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Flatten()(inputs)
    
    if model_type == 'simple':
        # Sangat sederhana untuk pengujian (parameter minimal)
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.4)(x)
        
        x = keras.layers.Dense(32, activation='relu')(x)
        x = keras.layers.Dropout(0.3)(x)
    
    elif model_type == 'baseline':
        # Baseline: Single Dense Layer (Linear Classifier)
        x = keras.layers.Dense(32, activation='relu')(x)
        # No BatchNormalization, No Dropout for raw baseline comparison
    
    else:  # default - INCREASED CAPACITY (32->64, 16->32) and REDUCED DROPOUT (0.6->0.3)
        x = keras.layers.Dense(64, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.3)(x)  # Dropout dikurangi agar bisa belajar lebih baik
        
        x = keras.layers.Dense(32, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.3)(x)
    
    # SOFTMAX untuk multi-kelas!
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs=inputs, outputs=outputs, name='multiclass_alphabet')
