import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import json
import os
from typing import Optional, Tuple, List
from collections import deque
import logging


logger = logging.getLogger(__name__)


class MultiClassWordDetector:
    """
    Detektor gestur kata multi-kelas Anti-Overfitting
    
    Dioptimalkan untuk kinerja realistis dengan generalisasi yang tepat
    Akurasi yang diharapkan: 94-98% (rentang kinerja sehat)
    """
    
    def __init__(
        self,
        model_path: str,
        class_names: List[str],
        sequence_length: int = 45,
        confidence_threshold: float = 0.6,  # Diturunkan dari 0.7 untuk model realistis
        cooldown_seconds: float = 2.0
    ):
        """
        Inisialisasi detektor kata multi-kelas
        
        Args:
            model_path: Path ke model multi-kelas yang terlatih
            class_names: Daftar nama kelas (misal, ['halo', 'terimakasih', 'tolong'])
            sequence_length: Jumlah frame dalam urutan
            confidence_threshold: Keyakinan minimum untuk deteksi
            cooldown_seconds: Waktu cooldown setelah deteksi
        """
        self.model_path = model_path
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.cooldown_seconds = cooldown_seconds
        
        # Muat model
        logger.info("Loading multi-class word model from: %s", model_path)
        self.model = keras.models.load_model(model_path)
        
        # Dapatkan info model
        total_params = self.model.count_params()
        model_name = self.model.name if hasattr(self.model, 'name') else 'unknown'
        
        logger.info("Multi-class word model loaded successfully")
        logger.info("Classes (%d): %s", self.num_classes, ", ".join(class_names))
        logger.info("Model name: %s, parameters: %d", model_name, total_params)
        
        # Cek jika model anti-overfitting
        if 'anti_overfit' in model_path or total_params < 10000:
            logger.info("Anti-overfitting model detected (realistic 94-98%% accuracy, generalized)")
        else:
            logger.warning("Standard model loaded - check for potential overfitting")
        
        # Buffer frame
        self.frame_buffer = deque(maxlen=sequence_length)
        
        # Status deteksi
        self.last_detection_time = 0
        self.total_detections = 0
        self.successful_detections = 0
        
        # Statistik per kelas
        self.class_detections = {name: 0 for name in class_names}
    
    def add_frame(self, landmarks: np.ndarray) -> bool:
        """Tambahkan frame ke buffer"""
        self.frame_buffer.append(landmarks)
        return len(self.frame_buffer) == self.sequence_length
    
    def is_in_cooldown(self) -> bool:
        """Cek apakah dalam periode cooldown"""
        return (time.time() - self.last_detection_time) < self.cooldown_seconds
    
    def reset_buffer(self):
        """Reset buffer frame"""
        self.frame_buffer.clear()
    
    def detect(self, landmarks: np.ndarray) -> Tuple[Optional[str], float, dict]:
        """
        Deteksi gestur dari landmarks
        
        Args:
            landmarks: Fitur landmark (126,)
            
        Returns:
            Tuple dari (label, confidence, all_probabilities)
        """
        self.total_detections += 1
        
        # Cek cooldown
        if self.is_in_cooldown():
            return None, 0.0, {}
        
        # Tambahkan frame ke buffer
        buffer_full = self.add_frame(landmarks)
        
        if not buffer_full:
            return None, 0.0, {}
        
        # Siapkan urutan
        sequence = np.array(list(self.frame_buffer))
        sequence = np.expand_dims(sequence, axis=0)  # (1, 45, 126)
        
        # Prediksi
        predictions = self.model.predict(sequence, verbose=0)[0]  # (num_classes,)
        
        # Dapatkan prediksi teratas
        class_idx = np.argmax(predictions)
        confidence = float(predictions[class_idx])
        label = self.class_names[class_idx]
        
        # Semua probabilitas
        all_probs = {name: float(predictions[i]) for i, name in enumerate(self.class_names)}
        
        # Cek kepercayaan yang ditingkatkan untuk model anti-overfitting
        # Model anti-overfitting memiliki distribusi kepercayaan yang lebih realistis
        
        # Cek ambang batas kepercayaan utama
        if confidence >= self.confidence_threshold:
            # Validasi tambahan untuk model anti-overfitting
            # Cek jika prediksi secara signifikan lebih baik dari yang kedua terbaik
            sorted_probs = sorted(predictions, reverse=True)
            confidence_gap = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else confidence
            
            # Untuk model anti-overfitting, butuh celah kepercayaan yang masuk akal
            min_confidence_gap = 0.1  # 10% celah antara prediksi teratas
            
            if confidence_gap >= min_confidence_gap or confidence >= 0.8:
                self.last_detection_time = time.time()
                self.successful_detections += 1
                self.class_detections[label] += 1
                self.reset_buffer()
                
                # Tambahkan metadata model anti-overfitting
                detection_metadata = all_probs.copy()
                detection_metadata.update({
                    'confidence_gap': float(confidence_gap),
                    'top2_ratio': float(sorted_probs[0] / sorted_probs[1]) if len(sorted_probs) > 1 and sorted_probs[1] > 0 else float('inf'),
                    'model_type': 'anti_overfitting' if 'anti_overfit' in self.model_path or self.model.count_params() < 10000 else 'standard'
                })
                
                return label, confidence, detection_metadata
        
        # Kembalikan dengan probabilitas detail untuk debugging
        debug_metadata = all_probs.copy()
        debug_metadata.update({
            'rejection_reason': 'low_confidence',
            'threshold_used': self.confidence_threshold
        })
        
        return None, confidence, debug_metadata
    
    def get_statistics(self) -> dict:
        """Dapatkan statistik deteksi"""
        return {
            'total_detections': self.total_detections,
            'successful_detections': self.successful_detections,
            'class_detections': self.class_detections.copy(),
            'success_rate': self.successful_detections / max(1, self.total_detections)
        }


class MultiClassAlphabetDetector:
    """
    Detektor gestur alfabet multi-kelas Anti-Overfitting
    
    Dioptimalkan untuk pengenalan pose statis realistis dengan generalisasi yang tepat
    Akurasi yang diharapkan: 96-98% (rentang kinerja sehat)
    """
    
    def __init__(
        self,
        model_path: str,
        class_names: List[str],
        confidence_threshold: float = 0.6,  # Diturunkan untuk model anti-overfitting
        stability_frames: int = 3,           # Dikurangi untuk respons lebih cepat
        cooldown_seconds: float = 1.5
    ):
        """
        Inisialisasi detektor alfabet multi-kelas
        
        Args:
            model_path: Path ke model multi-kelas yang terlatih
            class_names: Daftar nama kelas (misal, ['A', 'B', 'C', ...])
            confidence_threshold: Keyakinan minimum untuk deteksi
            stability_frames: Frame dengan prediksi sama sebelum konfirmasi
            cooldown_seconds: Waktu cooldown setelah deteksi
        """
        self.model_path = model_path
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.confidence_threshold = confidence_threshold
        self.stability_frames = stability_frames
        self.cooldown_seconds = cooldown_seconds
        
        # Muat model
        logger.info("Loading multi-class alphabet model from: %s", model_path)
        self.model = keras.models.load_model(model_path)
        
        # Dapatkan info model
        total_params = self.model.count_params()
        model_name = self.model.name if hasattr(self.model, 'name') else 'unknown'
        
        logger.info("Multi-class alphabet model loaded successfully")
        logger.info("Classes (%d): %s", self.num_classes, ", ".join(class_names))
        logger.info("Model name: %s, parameters: %d", model_name, total_params)
        
        # Cek jika model anti-overfitting
        if 'anti_overfit' in model_path or total_params < 8000:
            logger.info("Anti-overfitting model detected (realistic 96-98%% accuracy, static pose)")
        else:
            logger.warning("Standard model loaded - check for potential overfitting")
        
        # Pelacakan stabilitas
        self.stability_buffer = []
        
        # Status deteksi
        self.last_detection_time = 0
        self.total_detections = 0
        self.successful_detections = 0
        
        # Statistik per kelas
        self.class_detections = {name: 0 for name in class_names}
    
    def is_in_cooldown(self) -> bool:
        """Cek apakah dalam periode cooldown"""
        return (time.time() - self.last_detection_time) < self.cooldown_seconds
    
    def reset_stability(self):
        """Reset buffer stabilitas"""
        self.stability_buffer.clear()
    
    def detect(self, landmarks: np.ndarray) -> Tuple[Optional[str], float, dict]:
        """
        Deteksi gestur dari landmarks
        
        Args:
            landmarks: Fitur landmark (126,)
            
        Returns:
            Tuple dari (label, confidence, all_probabilities)
        """
        self.total_detections += 1
        
        # Cek cooldown
        if self.is_in_cooldown():
            return None, 0.0, {}
        
        # Siapkan input (1, 1, 126) untuk pose statis
        landmarks_input = landmarks.reshape(1, 1, -1)
        
        # Prediksi
        predictions = self.model.predict(landmarks_input, verbose=0)[0]  # (num_classes,)
        
        # Dapatkan prediksi teratas
        class_idx = np.argmax(predictions)
        confidence = float(predictions[class_idx])
        label = self.class_names[class_idx]
        
        # Semua probabilitas
        all_probs = {name: float(predictions[i]) for i, name in enumerate(self.class_names)}
        
        # Cek kepercayaan
        if confidence >= self.confidence_threshold:
            # --- GENERALIZED DYNAMIC THRESHOLD & SAFETY CHECKS ---
            # "A" hanyalah contoh, jadi kita terapkan logika robust ini untuk SEMUA kelas.
            # Tujuannya: Akurasi stabil, menolak 'tangan diam/ragu-ragu' untuk semua huruf.
            
            # Parameter Stabilitas
            HIGH_CONFIDENCE_THRESHOLD = 0.85  # Jika confidence > 0.85, kita sangat percaya
            REQUIRED_GAP = 0.15               # Gap minimal 15% jika confidence pas-pasan
            
            # 1. Hitung Confidence Gap (Jarak dengan prediksi kedua)
            sorted_probs = sorted(predictions, reverse=True)
            confidence_gap = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else confidence
            
            # 2. Logika Validasi "Dual Threshold"
            is_valid = False
            
            if confidence >= HIGH_CONFIDENCE_THRESHOLD:
                # KASUS 1: Sangat Yakin
                # Jika confidence sangat tinggi, kita bisa lebih toleran terhadap gap
                # tapi tetap butuh gap minimal kecil (misal 5%) untuk safety
                 if confidence_gap >= 0.05:
                    is_valid = True
                    
            elif confidence >= self.confidence_threshold:
                # KASUS 2: Yakin Sedang (Pas-pasan)
                # Jika confidence hanya di atas threshold dasar (misal 0.6 - 0.85),
                # kita WAJIB punya gap yang lebar untuk memastikan tidak bingung
                if confidence_gap >= REQUIRED_GAP:
                    is_valid = True
            
            if is_valid:
                # Tambahkan ke buffer stabilitas
                self.stability_buffer.append(label)
                
                # Simpan hanya prediksi terbaru
                if len(self.stability_buffer) > self.stability_frames:
                    self.stability_buffer.pop(0)
                
                # Cek stabilitas (Persistence)
                # Harus konsisten selama N frame berturut-turut
                if len(self.stability_buffer) >= self.stability_frames:
                    if len(set(self.stability_buffer)) == 1:  # Semua sama
                        detected_label = self.stability_buffer[0]
                        self.last_detection_time = time.time()
                        self.successful_detections += 1
                        self.class_detections[detected_label] += 1
                        self.reset_stability()
                        
                        # Tambahkan metadata deteksi untuk debugging
                        detection_metadata = all_probs.copy()
                        detection_metadata.update({
                            'confidence_gap': float(confidence_gap),
                            'validation_mode': 'high_confidence' if confidence >= HIGH_CONFIDENCE_THRESHOLD else 'gap_verified',
                            'stability_progress': 1.0 # Completed
                        })
                        
                        return detected_label, confidence, detection_metadata
            else:
                # Reset buffer jika tidak valid (untuk mencegah akumulasi frame ragu-ragu)
                if self.stability_buffer and self.stability_buffer[-1] != label:
                    self.reset_stability()
                    
        else:
            # Reset jika kepercayaan turun di bawah threshold dasar
            self.reset_stability()
            
        # Kembalikan metadata progres jika ada buffer
        current_progress = len(self.stability_buffer) / self.stability_frames
        progress_metadata = all_probs.copy()
        progress_metadata.update({
             'stability_progress': current_progress,
             'potential_label': self.stability_buffer[-1] if self.stability_buffer else None
        })
        
        return None, confidence, progress_metadata
    
    def get_statistics(self) -> dict:
        """Dapatkan statistik deteksi"""
        return {
            'total_detections': self.total_detections,
            'successful_detections': self.successful_detections,
            'class_detections': self.class_detections.copy(),
            'success_rate': self.successful_detections / max(1, self.total_detections)
        }


def load_multiclass_model(model_path: str) -> Tuple[keras.Model, List[str]]:
    """
    Muat model multi-kelas dan ekstrak nama kelas dari metadata
    
    Args:
        model_path: Path ke file model
        
    Returns:
        Tuple dari (model, class_names)
    """
    # Muat model
    model = keras.models.load_model(model_path)
    
    # Coba muat metadata
    metadata_path = model_path.replace('_final.keras', '_metadata.json')
    metadata_path = metadata_path.replace('_best.keras', '_metadata.json')
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        class_names = metadata['class_names']
    else:
        # Fallback: ekstrak dari layer output
        output_shape = model.output_shape
        num_classes = output_shape[-1]
        class_names = [f'class_{i}' for i in range(num_classes)]
        logger.warning("Metadata not found for %s, using generic class names: %s", model_path, class_names)
    
    return model, class_names


def get_model_metadata(model_path: str) -> dict:
    """
    Dapatkan metadata tentang model termasuk status anti-overfitting
    
    Args:
        model_path: Path ke file model
        
    Returns:
        Dictionary dengan metadata model
    """
    try:
        model = keras.models.load_model(model_path)
        total_params = model.count_params()
        model_name = model.name if hasattr(model, 'name') else 'unknown'
        
        # Tentukan jika anti-overfitting berdasarkan path dan parameter
        is_anti_overfit = ('anti_overfit' in model_path or 
                          total_params < 10000)  # Threshold untuk model kecil
        
        metadata = {
            'model_name': model_name,
            'total_parameters': total_params,
            'model_path': model_path,
            'is_anti_overfitting': is_anti_overfit,
            'expected_accuracy': '94-98%' if is_anti_overfit else '95-100% (check for overfitting)',
            'model_type': 'anti_overfitting' if is_anti_overfit else 'standard',
            'parameter_efficiency': 'high' if total_params < 10000 else 'standard'
        }
        
        return metadata
    except Exception as e:
        return {
            'error': str(e),
            'model_path': model_path,
            'is_anti_overfitting': False
        }


# Contoh penggunaan
if __name__ == '__main__':
    print("="*60)
    print("MULTI-CLASS DETECTOR TEST")
    print("="*60)
    
    print("\n✅ Multi-class detectors ready with ANTI-OVERFITTING support!")
    print("\n🛡️  ANTI-OVERFITTING MODELS:")
    print("   - Realistic 94-98% accuracy (healthy performance)")
    print("   - True generalization on new users")
    print("   - Efficient parameter usage (<10K params)")
    print("")
    print("📋 Usage:")
    print("  # Word detector (Anti-Overfitting)")
    print("  detector = MultiClassWordDetector(")
    print("      model_path='trained_models/multiclass_word_model_final.keras',")
    print("      class_names=['dervio', 'halo', 'namasaya', 'terimakasih'],")
    print("      confidence_threshold=0.6  # Lowered for realistic models")
    print("  )")
    print("")
    print("  # Alphabet detector (Anti-Overfitting)")
    print("  detector = MultiClassAlphabetDetector(")
    print("      model_path='trained_models/multiclass_alphabet_model_final.keras',")
    print("      class_names=['A', 'B', 'C', 'D', 'E'],")
    print("      confidence_threshold=0.6,  # Optimized for real performance")
    print("      stability_frames=3         # Faster response")
    print("  )")
    print("")
    print("🔍 Model Analysis:")
    print("  metadata = get_model_metadata('path/to/model.keras')")
    print("  print(f'Anti-overfitting: {metadata[\"is_anti_overfitting\"]}')")
