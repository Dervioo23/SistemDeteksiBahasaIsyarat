import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Optional, Tuple
import time
import logging


logger = logging.getLogger(__name__)


class AlphabetDetector:
    """Detektor untuk gestur alfabet statis"""
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.7,
        stability_frames: int = 5,
        cooldown_seconds: float = 1.5
    ):
        """
        Inisialisasi detektor alfabet
        
        Args:
            model_path: Path ke model yang telah dilatih
            confidence_threshold: Kepercayaan minimum untuk deteksi
            stability_frames: Frame untuk mengonfirmasi deteksi stabil
            cooldown_seconds: Detik untuk menunggu setelah deteksi
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.stability_frames = stability_frames
        self.cooldown_seconds = cooldown_seconds
        
        # Muat model
        logger.info("Loading alphabet model from: %s", model_path)
        self.model = keras.models.load_model(model_path)
        logger.info("Alphabet model loaded successfully")
        
        # Muat metadata untuk nama kelas
        self.class_names = []
        self.positive_label = "C" # Default untuk biner
        
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
        
        # Status deteksi
        self.last_detection_time = 0
        self.stable_count = 0
        self.last_label = None
        self.last_confidence = 0.0
        
        # Statistik
        self.total_detections = 0
        self.successful_detections = 0
    
    def is_in_cooldown(self) -> bool:
        """Periksa apakah dalam periode cooldown"""
        current_time = time.time()
        return (current_time - self.last_detection_time) < self.cooldown_seconds
    
    def detect(self, landmarks: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Deteksi alfabet dari satu frame
        
        Args:
            landmarks: Array landmark (126 fitur)
            
        Returns:
            Tuple dari (label, confidence) atau (None, 0.0)
        """
        # Periksa cooldown
        if self.is_in_cooldown():
            return None, 0.0
        
        # Siapkan input (tambahkan dimensi frame)
        input_data = landmarks.reshape(1, 1, 126)
        
        # Prediksi
        predictions = self.model.predict(input_data, verbose=0)[0]
        
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
        
        # Periksa ambang batas kepercayaan
        if confidence < self.confidence_threshold:
            return None, confidence

        # Periksa stabilitas
        if label == self.last_label:
            self.stable_count += 1
        else:
            self.stable_count = 0
            self.last_label = label
        
        # Konfirmasi deteksi jika stabil
        if self.stable_count >= self.stability_frames:
            self.last_detection_time = time.time()
            self.stable_count = 0
            self.successful_detections += 1
            return label, confidence
        
        self.total_detections += 1
        return None, confidence
    
    def detect_immediate(self, landmarks: np.ndarray) -> Tuple[str, float]:
        """
        Deteksi segera tanpa pemeriksaan stabilitas
        
        Args:
            landmarks: Array landmark (126 fitur)
            
        Returns:
            Tuple dari (label, confidence)
        """
        # Siapkan input
        input_data = landmarks.reshape(1, 1, 126)
        
        # Prediksi
        predictions = self.model.predict(input_data, verbose=0)[0]
        
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
    
    def reset_stability(self):
        """Reset penghitung stabilitas"""
        self.stable_count = 0
        self.last_label = None
    
    def get_statistics(self) -> dict:
        """Dapatkan statistik deteksi"""
        success_rate = 0.0
        if self.total_detections > 0:
            success_rate = self.successful_detections / self.total_detections
        
        return {
            'total_detections': self.total_detections,
            'successful_detections': self.successful_detections,
            'success_rate': success_rate,
            'stable_count': self.stable_count
        }


# Contoh penggunaan
if __name__ == '__main__':
    print("\n" + "="*60)
    print("ALPHABET DETECTOR TEST")
    print("="*60 + "\n")
    
    # Periksa jika model ada
    model_path = 'trained_models/alphabet_C_model_best.keras'
    
    import os
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("   Please train the model first!")
    else:
        # Inisialisasi detektor
        detector = AlphabetDetector(
            model_path=model_path,
            confidence_threshold=0.7,
            stability_frames=5
        )
        
        print("✅ Detector initialized")
        print(f"   Confidence threshold: {detector.confidence_threshold}")
        print(f"   Stability frames: {detector.stability_frames}")
        
        # Uji dengan data dummy
        print("\n🧪 Testing with dummy frames...")
        
        for i in range(20):
            dummy_frame = np.random.rand(126).astype(np.float32)
            label, confidence = detector.detect(dummy_frame)
            
            stats = detector.get_statistics()
            
            if label:
                print(f"\n✅ Stable detection at frame {i+1}!")
                print(f"   Label: {label}")
                print(f"   Confidence: {confidence:.4f}")
                break
            
            if (i + 1) % 5 == 0:
                print(f"   Frame {i+1}: Stable count = {stats['stable_count']}")
        
        # Uji deteksi segera
        print("\n🧪 Testing immediate detection...")
        dummy_frame = np.random.rand(126).astype(np.float32)
        label, confidence = detector.detect_immediate(dummy_frame)
        print(f"   Label: {label}")
        print(f"   Confidence: {confidence:.4f}")
        
        # Statistik
        stats = detector.get_statistics()
        print(f"\n📊 Statistics:")
        print(f"   Total: {stats['total_detections']}")
        print(f"   Successful: {stats['successful_detections']}")
        
        print("\n✅ Alphabet detector test completed!")
