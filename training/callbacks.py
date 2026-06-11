import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
from typing import Dict, List
import os


class TrainingMonitor(keras.callbacks.Callback):
    """Monitor kemajuan pelatihan dengan metrik terperinci"""
    
    def __init__(self):
        super().__init__()
        self.epoch_times = []
        self.epoch_start = None
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}")
        print(f"{'='*60}")
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)
        
        logs = logs or {}
        
        print(f"\n📊 Epoch {epoch + 1} Summary:")
        print(f"   Time: {epoch_time:.2f}s")
        print(f"   Train Loss: {logs.get('loss', 0):.4f}")
        print(f"   Train Acc:  {logs.get('accuracy', 0):.4f}")
        print(f"   Val Loss:   {logs.get('val_loss', 0):.4f}")
        print(f"   Val Acc:    {logs.get('val_accuracy', 0):.4f}")
        
        if 'precision' in logs:
            print(f"   Precision:  {logs.get('precision', 0):.4f}")
        if 'recall' in logs:
            print(f"   Recall:     {logs.get('recall', 0):.4f}")
        
        # Learning rate (jika tersedia)
        if hasattr(self.model.optimizer, 'lr'):
            lr = float(keras.backend.get_value(self.model.optimizer.lr))
            print(f"   Learning Rate: {lr:.6f}")
        
        print(f"{'='*60}")
    
    def on_train_end(self, logs=None):
        avg_time = np.mean(self.epoch_times)
        total_time = np.sum(self.epoch_times)
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"Total Time: {total_time:.2f}s ({total_time/60:.1f} min)")
        print(f"Avg Epoch Time: {avg_time:.2f}s")
        print(f"{'='*60}\n")


class MetricsLogger(keras.callbacks.Callback):
    """Log metrik ke file untuk analisis"""
    
    def __init__(self, log_file: str):
        super().__init__()
        self.log_file = log_file
        self.metrics_history = []
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        log_entry = {'epoch': epoch + 1}
        log_entry.update(logs)
        self.metrics_history.append(log_entry)
    
    def on_train_end(self, logs=None):
        # Simpan ke file
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        with open(self.log_file, 'w') as f:
            # Header
            if self.metrics_history:
                keys = self.metrics_history[0].keys()
                f.write(','.join(keys) + '\n')
                
                # Data
                for entry in self.metrics_history:
                    values = [str(entry[k]) for k in keys]
                    f.write(','.join(values) + '\n')
        
        print(f"✅ Metrics saved to: {self.log_file}")


class BestModelSaver(keras.callbacks.Callback):
    """Simpan model saat mencapai akurasi validasi terbaik"""
    
    def __init__(self, save_path: str, monitor='val_accuracy', mode='max'):
        super().__init__()
        self.save_path = save_path
        self.monitor = monitor
        self.mode = mode
        self.best = -np.Inf if mode == 'max' else np.Inf
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            return
        
        if self.mode == 'max':
            is_better = current > self.best
        else:
            is_better = current < self.best
        
        if is_better:
            self.best = current
            self.model.save(self.save_path)
            print(f"\n💾 Best model saved! {self.monitor}={current:.4f}")


class ConfusionMatrixCallback(keras.callbacks.Callback):
    """Hitung confusion matrix pada set validasi"""
    
    def __init__(self, X_val, y_val, class_names=None, freq=5):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.class_names = class_names or ['Class 0', 'Class 1']
        self.freq = freq  # Hitung setiap N epoch
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.freq != 0:
            return
        
        # Prediksi
        y_pred_probs = self.model.predict(self.X_val, verbose=0)
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
        y_true = self.y_val.flatten()
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        
        print(f"\n📊 Confusion Matrix (Epoch {epoch + 1}):")
        print(f"   Predicted →")
        print(f"   {'True ↓':<12} {self.class_names[0]:<10} {self.class_names[1]:<10}")
        for i, row in enumerate(cm):
            print(f"   {self.class_names[i]:<12} {row[0]:<10} {row[1]:<10}")


def create_standard_callbacks(
    model_name: str,
    output_dir: str = 'trained_models',
    monitor: str = 'val_accuracy',
    patience: int = 10
) -> List[keras.callbacks.Callback]:
    """
    Buat set callback standar untuk pelatihan
    
    Args:
        model_name: Nama model
        output_dir: Direktori untuk menyimpan model
        monitor: Metrik untuk dipantau
        patience: Kesabaran untuk penghentian awal (early stopping)
        
    Returns:
        Daftar callback
    """
    os.makedirs(output_dir, exist_ok=True)
    
    callbacks_list = []
    
    # Monitor pelatihan
    callbacks_list.append(TrainingMonitor())
    
    # Checkpoint model
    checkpoint_path = os.path.join(output_dir, f'{model_name}_best.keras')
    callbacks_list.append(
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=monitor,
            save_best_only=True,
            mode='max',
            verbose=1
        )
    )
    
    # Penghentian awal (Early stopping)
    callbacks_list.append(
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
    )
    
    # Kurangi learning rate
    callbacks_list.append(
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    )
    
    # TensorBoard
    log_dir = os.path.join('logs', model_name)
    callbacks_list.append(
        keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
    )
    
    # CSV Logger
    csv_path = os.path.join(output_dir, f'{model_name}_history.csv')
    callbacks_list.append(
        keras.callbacks.CSVLogger(csv_path)
    )
    
    # Logger metrik
    metrics_path = os.path.join(output_dir, f'{model_name}_metrics.txt')
    callbacks_list.append(
        MetricsLogger(metrics_path)
    )
    
    return callbacks_list


def create_anti_overfitting_callbacks(
    model_name: str,
    output_dir: str = 'trained_models',
    monitor: str = 'val_accuracy',
    early_patience: int = 3,  # Penghentian awal yang sangat agresif
    lr_patience: int = 2      # Pengurangan learning rate cepat
) -> List[keras.callbacks.Callback]:
    """
    Buat callback ANTI-OVERFITTING untuk dataset kecil
    Penghentian awal dan regularisasi yang jauh lebih agresif
    
    Args:
        model_name: Nama model
        output_dir: Direktori untuk menyimpan model
        monitor: Metrik untuk dipantau (default val_accuracy)
        early_patience: Kesabaran untuk penghentian awal (jauh lebih kecil!)
        lr_patience: Kesabaran untuk pengurangan LR
        
    Returns:
        Daftar callback agresif untuk mencegah overfitting
    """
    os.makedirs(output_dir, exist_ok=True)
    
    callbacks_list = []
    
    # Monitor pelatihan
    callbacks_list.append(TrainingMonitor())
    
    # Checkpoint model (simpan model terbaik)
    checkpoint_path = os.path.join(output_dir, f'{model_name}_best.keras')
    callbacks_list.append(
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=monitor,
            save_best_only=True,
            mode='max',
            verbose=1
        )
    )
    
    # PENGHENTIAN AWAL AGRESIF (patience=3 alih-alih 10)
    callbacks_list.append(
        keras.callbacks.EarlyStopping(
            monitor='val_loss',           # Pantau loss validasi
            patience=early_patience,      # Berhenti setelah 3 epoch tanpa perbaikan
            restore_best_weights=True,    # Pulihkan bobot terbaik
            verbose=1,
            min_delta=0.001              # Perubahan minimum untuk memenuhi syarat sebagai perbaikan
        )
    )
    
    # PENGURANGAN LEARNING RATE AGRESIF (patience=2 alih-alih 5)
    callbacks_list.append(
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,                  # Kurangi LR sebesar 5x (alih-alih 2x)
            patience=lr_patience,        # Kurangi setelah 2 epoch (alih-alih 5)
            min_lr=1e-7,                # LR minimum lebih rendah
            verbose=1
        )
    )
    
    # CSV Logger untuk tracking
    csv_path = os.path.join(output_dir, f'{model_name}_training.csv')
    callbacks_list.append(
        keras.callbacks.CSVLogger(csv_path)
    )
    
    print(f"🛡️  Anti-overfitting callbacks created:")
    print(f"   - Early stopping patience: {early_patience}")
    print(f"   - LR reduction patience: {lr_patience}")
    print(f"   - Monitoring: {monitor}")
    
    return callbacks_list


# Contoh penggunaan
if __name__ == '__main__':
    print("🧪 Testing Callbacks...\n")
    
    # Buat data dummy
    X_train = np.random.rand(100, 45, 126).astype(np.float32)
    y_train = np.random.randint(0, 2, (100, 1)).astype(np.float32)
    X_val = np.random.rand(20, 45, 126).astype(np.float32)
    y_val = np.random.randint(0, 2, (20, 1)).astype(np.float32)
    
    # Buat model sederhana
    model = keras.Sequential([
        keras.layers.Input(shape=(45, 126)),
        keras.layers.LSTM(32),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Buat callback
    callbacks = create_standard_callbacks('test_model')
    
    print(f"✅ Created {len(callbacks)} callbacks")
    
    # Tes pelatihan cepat
    print("\n🧪 Testing with dummy training (3 epochs)...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=3,
        batch_size=16,
        callbacks=callbacks,
        verbose=0
    )
    
    print("\n✅ Callback tests completed!")
