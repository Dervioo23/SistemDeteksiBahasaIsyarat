import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Optional, Tuple, List
import time
from collections import deque
import logging


logger = logging.getLogger(__name__)


class WordDetector:
    """Detektor untuk gestur kata berurutan"""
    
    def __init__(
        self,
        model_path: str,
        sequence_length: int = 45,
        confidence_threshold: float = 0.7,
        cooldown_seconds: float = 2.0
    ):
        """
        Inisialisasi detektor kata
        
        Args:
            model_path: Path ke model yang telah dilatih
            sequence_length: Panjang urutan yang diharapkan
            confidence_threshold: Kepercayaan minimum untuk deteksi
            cooldown_seconds: Detik untuk menunggu setelah deteksi
        """
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.cooldown_seconds = cooldown_seconds
        
        # Muat model
        logger.info("Loading word model from: %s", model_path)
        self.model = keras.models.load_model(model_path)
        logger.info("Word model loaded successfully")
        
        # Muat metadata untuk nama kelas
        self.class_names = []
        self.positive_label = "halo" # Default untuk biner
        
        import os, json
        # Coba temukan file metadata
        meta_path = model_path.replace("_final.keras", "_metadata.json").replace("_best.keras", "_metadata.json")
        if not os.path.exists(meta_path):
             # Coba hapus ekstensi dan tambahkan _metadata.json
             base_path = os.path.splitext(model_path)[0]
             meta_path = base_path + "_metadata.json"
             
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                
                # Dapatkan nama kelas
                if "class_names" in meta:
                    self.class_names = meta["class_names"]
                    logger.info(f"Loaded {len(self.class_names)} class names from metadata")
                    
                # Untuk kompatibilitas biner
                if isinstance(self.class_names, list) and len(self.class_names) >= 1:
                    self.positive_label = self.class_names[0]
                    
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        else:
            logger.warning(f"Metadata file not found: {meta_path}")
        
        # Penyangga frame
        self.frame_buffer = deque(maxlen=sequence_length)
        self.is_recording = False
        
        # Status deteksi
        self.last_detection_time = 0
        self.current_prediction = None
        self.current_confidence = 0.0
        
        # Statistik
        self.total_detections = 0
        self.successful_detections = 0
        
    def reset_buffer(self):
        """Reset penyangga frame"""
        self.frame_buffer.clear()
        self.is_recording = False
    
    def add_frame(self, landmarks: np.ndarray) -> bool:
        """
        Tambahkan frame ke penyangga
        
        Args:
            landmarks: Array landmark (126 fitur)
            
        Returns:
            True jika penyangga penuh
        """
        self.frame_buffer.append(landmarks)
        return len(self.frame_buffer) >= self.sequence_length
    
    def is_in_cooldown(self) -> bool:
        """Periksa apakah detektor dalam periode cooldown"""
        current_time = time.time()
        return (current_time - self.last_detection_time) < self.cooldown_seconds
    
    def detect(self, landmarks: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Deteksi gestur kata dari satu frame
        
        Args:
            landmarks: Landmark frame saat ini (126 fitur)
            
        Returns:
            Tuple dari (label, confidence) atau (None, 0.0)
        """
        # Periksa cooldown
        if self.is_in_cooldown():
            return None, 0.0
        
        # Tambahkan frame ke penyangga
        buffer_full = self.add_frame(landmarks)
        
        # Debug: log status penyangga setiap 10 frame atau saat penuh
        buffer_size = len(self.frame_buffer)
        if buffer_size % 10 == 0 or buffer_full:
            logger.debug("Word buffer: %d/%d frames", buffer_size, self.sequence_length)
        
        if not buffer_full:
            return None, 0.0
        
        # Siapkan urutan untuk prediksi
        sequence = np.array(list(self.frame_buffer))
        sequence = np.expand_dims(sequence, axis=0)  # Tambahkan dimensi batch
        
        # Prediksi
        predictions = self.model.predict(sequence, verbose=0)[0]
        
        # Tangani multi-kelas vs biner
        if len(predictions) > 1:
            # Multi-kelas
            class_idx = np.argmax(predictions)
            confidence = float(predictions[class_idx])
            
            if self.class_names and class_idx < len(self.class_names):
                label = self.class_names[class_idx]
            else:
                label = str(class_idx) # Fallback
        else:
            # Biner
            confidence = float(predictions[0])
            if confidence >= self.confidence_threshold:
                label = self.positive_label
            else:
                label = f"not_{self.positive_label}"
                confidence = 1.0 - confidence
        
        # Debug: log detail prediksi
        logger.debug("Word prediction: %s (conf=%.3f, threshold=%.3f)", label, confidence, self.confidence_threshold)
        
        # Periksa kepercayaan
        if confidence >= self.confidence_threshold:
            self.last_detection_time = time.time()
            self.successful_detections += 1
            self.reset_buffer()  # Reset untuk deteksi berikutnya
            logger.info("Word detected: %s (conf=%.3f)", label, confidence)
            return label, confidence
        
        self.total_detections += 1
        return None, confidence
    
    def detect_from_sequence(self, sequence: np.ndarray) -> Tuple[str, float]:
        """
        Deteksi dari urutan lengkap
        
        Args:
            sequence: Urutan penuh (sequence_length, 126)
            
        Returns:
            Tuple dari (label, confidence)
        """
        # Pastikan bentuk yang benar
        if len(sequence) != self.sequence_length:
            raise ValueError(f"Expected {self.sequence_length} frames, got {len(sequence)}")
        
        # Prediksi
        sequence = np.expand_dims(sequence, axis=0)
        predictions = self.model.predict(sequence, verbose=0)[0]
        
        # Tangani multi-kelas vs biner
        if len(predictions) > 1:
            # Multi-kelas
            class_idx = np.argmax(predictions)
            confidence = float(predictions[class_idx])
            
            if self.class_names and class_idx < len(self.class_names):
                label = self.class_names[class_idx]
            else:
                label = str(class_idx)
        else:
            # Biner
            confidence = float(predictions[0])
            if confidence >= self.confidence_threshold:
                label = self.positive_label
            else:
                label = f"not_{self.positive_label}"
                confidence = 1.0 - confidence
                
        return label, confidence
    
    def get_buffer_status(self) -> dict:
        """Dapatkan status penyangga saat ini"""
        return {
            'buffer_size': len(self.frame_buffer),
            'buffer_full': len(self.frame_buffer) >= self.sequence_length,
            'in_cooldown': self.is_in_cooldown(),
            'recording': self.is_recording
        }
    
    def get_statistics(self) -> dict:
        """Dapatkan statistik deteksi"""
        success_rate = 0.0
        if self.total_detections > 0:
            success_rate = self.successful_detections / self.total_detections
        
        return {
            'total_detections': self.total_detections,
            'successful_detections': self.successful_detections,
            'success_rate': success_rate
        }

    def _infer_positive_label(self, model_path: str) -> str:
        import os, re, json
        base = os.path.basename(model_path)
        m = re.search(r"word_([^_]+)_model", base)
        if m:
            return m.group(1)
        meta_path = model_path.replace("_final.keras", "_metadata.json").replace("_best.keras", "_metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                names = meta.get("class_names")
                if isinstance(names, list) and len(names) >= 1:
                    return names[0]
            except Exception:
                pass
        return "halo"


# Contoh penggunaan
if __name__ == '__main__':
    print("\n" + "="*60)
    print("WORD DETECTOR TEST")
    print("="*60 + "\n")
    
    # Periksa jika model ada
    model_path = 'trained_models/word_halo_model_best.keras'
    
    import os
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("   Please train the model first!")
    else:
        # Inisialisasi detektor
        detector = WordDetector(
            model_path=model_path,
            sequence_length=45,
            confidence_threshold=0.7
        )
        
        print("✅ Detector initialized")
        print(f"   Sequence length: {detector.sequence_length}")
        print(f"   Confidence threshold: {detector.confidence_threshold}")
        
        # Uji dengan data dummy
        print("\n🧪 Testing with dummy frames...")
        
        for i in range(50):
            dummy_frame = np.random.rand(126).astype(np.float32)
            label, confidence = detector.detect(dummy_frame)
            
            if label:
                print(f"\n✅ Detection at frame {i+1}!")
                print(f"   Label: {label}")
                print(f"   Confidence: {confidence:.4f}")
                break
            
            if (i + 1) % 10 == 0:
                status = detector.get_buffer_status()
                print(f"   Frame {i+1}: Buffer {status['buffer_size']}/{detector.sequence_length}")
        
        # Statistik
        stats = detector.get_statistics()
        print(f"\n📊 Statistics:")
        print(f"   Total: {stats['total_detections']}")
        print(f"   Successful: {stats['successful_detections']}")
        
        print("\n✅ Word detector test completed!")
