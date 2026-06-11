"""
Z-Coordinate Experiment Test
Uji coba performa model dengan dan tanpa koordinat Z (kedalaman)
"""
import os
import numpy as np
"""
Uji Coba Eksperimen Koordinat Z
Uji coba performa model dengan dan tanpa koordinat Z (kedalaman)
"""
import os
import numpy as np
import time
import cv2
from typing import Dict, List
import pytest

if __name__ != "__main__":
    pytest.skip("Debug script that expects a real dataset; run directly with python, not via pytest.", allow_module_level=True)

from data_collection.landmark_extractor import LandmarkExtractor
from data_collection.quality_validator import QualityValidator
from data_collection.utils import load_config


class ZCoordinateExperiment:
    """Class untuk menguji performa dengan dan tanpa koordinat Z"""
    
    def __init__(self, test_duration: int = 30):
        """
        Argumen:
            test_duration: Durasi pengujian dalam detik
        """
        self.test_duration = test_duration
        self.results = {
            'with_z': {'samples': [], 'processing_times': [], 'z_stability': []},
            'without_z': {'samples': [], 'processing_times': []}
        }
        
    def run_experiment(self):
        """Jalankan eksperimen perbandingan koordinat Z"""
        print("\n" + "="*70)
        print("🧪 Z-COORDINATE EXPERIMENT")
        print("="*70)
        print(f"📊 Testing duration: {self.test_duration} seconds each mode")
        print(f"🎯 Objective: Compare performance with vs without Z coordinate")
        print("="*70)
        
        # Inisialisasi kamera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("❌ Error: Cannot open camera!")
            return
        
        # Tes DENGAN koordinat Z
        print(f"\n🧪 PHASE 1: Testing WITH Z-coordinate (126 features)")
        print(f"⏱️  Duration: {self.test_duration}s")
        print("👋 Place your hands in front of camera...")
        time.sleep(3)  # Hitung mundur
        
        self._test_mode(cap, use_z=True, mode_name="WITH Z")
        
        # Tes TANPA koordinat Z
        print(f"\n🧪 PHASE 2: Testing WITHOUT Z-coordinate (84 features)")
        print(f"⏱️  Duration: {self.test_duration}s")
        print("👋 Keep your hands in the same position...")
        time.sleep(3)  # Hitung mundur
        
        self._test_mode(cap, use_z=False, mode_name="WITHOUT Z")
        
        # Analisis hasil
        self._analyze_results()
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _test_mode(self, cap: cv2.VideoCapture, use_z: bool, mode_name: str):
        """Uji mode tertentu (dengan atau tanpa Z)"""
        
        # Inisialisasi ekstraktor untuk mode ini menggunakan ambang batas yang didorong konfigurasi
        config = load_config()
        coll_cfg = config.get("collection", {})
        det_conf = coll_cfg.get("min_detection_confidence", 0.7)
        track_conf = coll_cfg.get("min_tracking_confidence", 0.5)

        extractor = LandmarkExtractor(
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
            use_z_coordinate=use_z,
            experimental_mode=True
        )
        
        # Inisialisasi validator kualitas
        validator = QualityValidator(
            min_confidence=0.7,
            min_stability=0.8,
            dynamic_gesture=False  # Pengujian pose statis
        )
        
        samples = []
        processing_times = []
        z_stability_data = []
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < self.test_duration:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            
            # Ukur waktu pemrosesan
            process_start = time.time()
            
            # Ekstrak landmark
            hands_dict, annotated_frame, num_hands = extractor.extract_both_hands(frame)
            
            process_time = time.time() - process_start
            processing_times.append(process_time)
            
            # Kumpulkan sampel jika tangan terdeteksi
            if num_hands > 0:
                hands_data = extractor.get_both_hands_with_fallback(hands_dict)
                flattened = extractor.flatten_both_hands(hands_data)
                samples.append(flattened)
                
                # Validasi kualitas (opsional)
                if hands_dict['right']:
                    right_lm = np.array([[lm['x'], lm['y']] + ([lm['z']] if use_z and 'z' in lm else []) 
                                       for lm in hands_dict['right']], dtype=np.float32)
                    quality_passed, quality_scores, feedback = validator.validate_sample(right_lm, "test", frame)
            
            # Tampilkan info
            h, w, _ = annotated_frame.shape
            cv2.rectangle(annotated_frame, (10, 10), (w-10, 120), (0, 0, 0), -1)
            cv2.rectangle(annotated_frame, (10, 10), (w-10, 120), (255, 255, 255), 2)
            
            cv2.putText(annotated_frame, f"Mode: {mode_name}", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Features: {extractor.total_features} per frame", 
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            remaining = self.test_duration - (time.time() - start_time)
            cv2.putText(annotated_frame, f"Time: {remaining:.1f}s", 
                       (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow("Z-Coordinate Experiment", annotated_frame)
            
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Simpan hasil
        mode_key = 'with_z' if use_z else 'without_z'
        self.results[mode_key] = {
            'samples': samples,
            'processing_times': processing_times,
            'frame_count': frame_count,
            'total_features': extractor.total_features,
            'features_per_hand': extractor.features_per_hand,
            'features_per_landmark': extractor.features_per_landmark
        }
        
        # Dapatkan analisis stabilitas Z (hanya untuk mode with_z)
        if use_z:
            z_analysis = extractor.get_z_stability_analysis()
            self.results[mode_key]['z_analysis'] = z_analysis
        
        extractor.close()
        print(f"✅ {mode_name} test completed: {len(samples)} samples collected")
    
    def _analyze_results(self):
        """Analisis dan bandingkan hasil"""
        print(f"\n📊 EXPERIMENT RESULTS")
        print("="*70)
        
        with_z = self.results['with_z']
        without_z = self.results['without_z']
        
        # Statistik dasar
        print(f"\n📈 BASIC STATISTICS:")
        print(f"{'Metric':<25} {'With Z':<15} {'Without Z':<15} {'Difference':<15}")
        print(f"{'-'*25} {'-'*15} {'-'*15} {'-'*15}")
        
        # Dimensi fitur
        print(f"{'Features per frame':<25} {with_z['total_features']:<15} {without_z['total_features']:<15} {with_z['total_features'] - without_z['total_features']:<15}")
        
        # Sampel yang dikumpulkan
        with_z_samples = len(with_z['samples'])
        without_z_samples = len(without_z['samples'])
        print(f"{'Samples collected':<25} {with_z_samples:<15} {without_z_samples:<15} {with_z_samples - without_z_samples:<15}")
        
        # Waktu pemrosesan
        if with_z['processing_times'] and without_z['processing_times']:
            with_z_avg_time = np.mean(with_z['processing_times']) * 1000  # Konversi ke ms
            without_z_avg_time = np.mean(without_z['processing_times']) * 1000
            time_diff = with_z_avg_time - without_z_avg_time
            
            print(f"{'Avg processing (ms)':<25} {with_z_avg_time:<15.2f} {without_z_avg_time:<15.2f} {time_diff:<+15.2f}")
            
            # Peningkatan kinerja
            if with_z_avg_time > 0:
                perf_improvement = (time_diff / with_z_avg_time) * 100
                print(f"{'Performance gain (%)':<25} {'-':<15} {perf_improvement:<+15.1f} {'-':<15}")
        
        # Estimasi penggunaan memori
        if with_z_samples > 0 and without_z_samples > 0:
            with_z_memory = with_z_samples * with_z['total_features'] * 4  # float32 = 4 bytes
            without_z_memory = without_z_samples * without_z['total_features'] * 4
            memory_saving = with_z_memory - without_z_memory
            
            print(f"{'Est. memory (KB)':<25} {with_z_memory/1024:<15.1f} {without_z_memory/1024:<15.1f} {memory_saving/1024:<+15.1f}")
        
        # Analisis stabilitas koordinat Z
        if 'z_analysis' in with_z and with_z['z_analysis']:
            z_stats = with_z['z_analysis']
            print(f"\n📊 Z-COORDINATE STABILITY ANALYSIS:")
            print(f"  - Stability Score: {z_stats.get('stability_score', 0):.3f}")
            print(f"  - Standard Deviation: {z_stats.get('std_z', 0):.4f}")
            print(f"  - Coefficient of Variation: {z_stats.get('coefficient_of_variation', 0):.4f}")
            
            stability_score = z_stats.get('stability_score', 0)
            if stability_score > 0.8:
                stability_verdict = "✅ STABIL - Pertahankan koordinat Z"
            elif stability_score > 0.6:
                stability_verdict = "⚠️  MODERAT - Pertimbangkan persyaratan aplikasi"
            else:
                stability_verdict = "❌ TIDAK STABIL - Rekomendasikan penghapusan koordinat Z"
            
            print(f"  - Verdict: {stability_verdict}")
        
        # Rekomendasi
        print(f"\n💡 RECOMMENDATIONS:")
        print("="*70)
        
        # Manfaat pengurangan fitur
        feature_reduction = with_z['total_features'] - without_z['total_features']
        reduction_percent = (feature_reduction / with_z['total_features']) * 100
        
        print(f"📊 Feature Reduction: {feature_reduction} features ({reduction_percent:.1f}% reduction)")
        
        # Analisis kinerja
        if with_z['processing_times'] and without_z['processing_times']:
            with_z_avg_time = np.mean(with_z['processing_times']) * 1000
            without_z_avg_time = np.mean(without_z['processing_times']) * 1000
            
            if without_z_avg_time < with_z_avg_time:
                print(f"⚡ Processing Speed: {((with_z_avg_time - without_z_avg_time) / with_z_avg_time * 100):.1f}% faster without Z")
            
        # Konsistensi sampel
        sample_diff = abs(with_z_samples - without_z_samples)
        if sample_diff <= with_z_samples * 0.05:  # Dalam 5%
            print(f"✅ Detection Consistency: Similar detection rates ({sample_diff} sample difference)")
        else:
            print(f"⚠️  Detection Consistency: Significant difference ({sample_diff} samples)")
        
        # Rekomendasi akhir
        print(f"\n🎯 FINAL RECOMMENDATION:")
        
        reasons_to_remove_z = []
        reasons_to_keep_z = []
        
        # Periksa stabilitas Z
        if 'z_analysis' in with_z and with_z['z_analysis']:
            stability_score = with_z['z_analysis'].get('stability_score', 0)
            if stability_score < 0.7:
                reasons_to_remove_z.append(f"Koordinat Z tidak stabil (skor: {stability_score:.3f})")
            else:
                reasons_to_keep_z.append(f"Koordinat Z stabil (skor: {stability_score:.3f})")
        
        # Periksa peningkatan kinerja
        if with_z['processing_times'] and without_z['processing_times']:
            with_z_avg_time = np.mean(with_z['processing_times'])
            without_z_avg_time = np.mean(without_z['processing_times'])
            if without_z_avg_time < with_z_avg_time * 0.95:  # Peningkatan 5%
                reasons_to_remove_z.append("Peningkatan kinerja yang signifikan")
        
        # Periksa penghematan memori
        if reduction_percent >= 25:  # Pengurangan signifikan
            reasons_to_remove_z.append(f"Pengurangan memori yang signifikan ({reduction_percent:.1f}%)")
        
        # Periksa konsistensi deteksi
        if sample_diff <= with_z_samples * 0.1:  # Konsistensi yang baik
            reasons_to_remove_z.append("Tidak ada dampak negatif pada deteksi")
        else:
            reasons_to_keep_z.append("Potensi dampak negatif pada deteksi")
        
        if len(reasons_to_remove_z) > len(reasons_to_keep_z):
            print(f"✅ REKOMENDASI: Hapus koordinat Z")
            print(f"   Alasan:")
            for reason in reasons_to_remove_z:
                print(f"   - {reason}")
        else:
            print(f"⚠️  REKOMENDASI: Pertahankan koordinat Z")
            print(f"   Alasan:")
            for reason in reasons_to_keep_z:
                print(f"   - {reason}")
        
        print(f"\n💾 Langkah selanjutnya:")
        print(f"   1. Jika menghapus Z: Perbarui konfigurasi ke use_z_coordinate=False")
        print(f"   2. Perbarui bentuk input model: 126 → 84 fitur")
        print(f"   3. Latih ulang model dengan dimensi fitur baru")
        print(f"   4. Bandingkan akurasi model antar versi")
        
        print("="*70)


def main():
    """Jalankan eksperimen koordinat Z"""
    experiment = ZCoordinateExperiment(test_duration=15)  # 15 detik setiap mode
    experiment.run_experiment()


if __name__ == "__main__":
    main()
