import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import json
import os


class QualityValidator:
    """Validasi kualitas sampel saat pengumpulan data"""
    
    def __init__(
        self,
        min_confidence: float = 0.7,
        min_stability: float = 0.85,
        baseline_dir: str = 'reference_gestures',
        dynamic_gesture: bool = False,
        enable_blur_check: bool = True,
        blur_min_variance: float = 30.0,
        blur_max_variance: float = 120.0
    ):
        """
        Args:
            min_confidence: Keyakinan minimum untuk menerima sampel
            min_stability: Skor stabilitas minimum (0-1)
            baseline_dir: Direktori untuk gestur referensi
            dynamic_gesture: True untuk gestur yang bergerak (kata), bobot stabilitas dikurangi
        """
        self.min_confidence = min_confidence
        self.min_stability = min_stability
        self.baseline_dir = baseline_dir
        self.dynamic_gesture = dynamic_gesture
        self.enable_blur_check = enable_blur_check
        self.blur_min_variance = blur_min_variance
        self.blur_max_variance = blur_max_variance
        
        # Muat referensi baseline jika ada
        self.baselines = self._load_baselines()
        
        # Pelacakan untuk konsistensi
        self.previous_landmarks = []
        self.sample_history = {}  # gestur -> daftar landmark
        
    def _load_baselines(self) -> Dict:
        """Muat gestur referensi untuk perbandingan"""
        baselines = {}
        if os.path.exists(self.baseline_dir):
            for gesture_file in os.listdir(self.baseline_dir):
                if gesture_file.endswith('.json'):
                    gesture_name = gesture_file.replace('.json', '')
                    with open(os.path.join(self.baseline_dir, gesture_file), 'r') as f:
                        baselines[gesture_name] = json.load(f)
        return baselines
    
    def validate_sample(
        self,
        landmarks: np.ndarray,
        gesture: str,
        frame: np.ndarray = None
    ) -> Tuple[bool, Dict[str, float], str]:
        """
        Validasi kualitas sampel
        
        Args:
            landmarks: Data landmark (21, 3)
            gesture: Nama gestur
            frame: Frame gambar (opsional)
        
        Returns:
            (is_valid, scores, feedback_message)
        """
        scores = {}
        feedback_parts = []
        weights = {}
        
        # 1. Cek kelengkapan (semua landmark terdeteksi)
        completeness_score = self._check_completeness(landmarks)
        scores['completeness'] = completeness_score
        weights['completeness'] = 1.0  # Selalu penting
        
        if completeness_score < 1.0:
            feedback_parts.append(f"⚠️ Landmarks incomplete ({completeness_score*100:.0f}%)")
        
        # 2. Cek stabilitas (tidak goyang)
        stability_score = self._check_stability(landmarks)
        scores['stability'] = stability_score
        
        # Gestur dinamis: kurangi bobot stabilitas (gerakan diharapkan!)
        if self.dynamic_gesture:
            weights['stability'] = 0.3  # Bobot rendah untuk gestur bergerak
            # Umpan balik lebih longgar untuk gestur dinamis
            if stability_score < 0.5:  # Hanya peringatkan jika SANGAT tidak stabil
                feedback_parts.append(f"⚠️ Too much variation ({stability_score*100:.0f}%)")
        else:
            weights['stability'] = 1.0  # Bobot penuh untuk gestur statis
            if stability_score < self.min_stability:
                feedback_parts.append(f"⚠️ Hand unstable ({stability_score*100:.0f}%)")
        
        # 3. Cek posisi (dalam frame yang baik)
        position_score = self._check_position(landmarks)
        scores['position'] = position_score
        weights['position'] = 0.8  # Agak penting
        
        if position_score < 0.7:
            feedback_parts.append(f"⚠️ Position suboptimal ({position_score*100:.0f}%)")

        # Cek blur opsional jika frame tersedia
        if frame is not None and self.enable_blur_check:
            blur_score = self._check_blur(frame, landmarks)
            scores['blur'] = blur_score
            weights['blur'] = 1.0
            if blur_score < 0.7:
                feedback_parts.append(f"⚠️ Image blurry ({blur_score*100:.0f}%)")
        
        # 4. Cek konsistensi dengan baseline (jika ada)
        if gesture in self.baselines:
            consistency_score = self._check_consistency(landmarks, gesture)
            scores['consistency'] = consistency_score
            weights['consistency'] = 0.5  # Cek opsional
            
            if consistency_score < 0.6:
                feedback_parts.append(f"⚠️ Different from reference ({consistency_score*100:.0f}%)")
        
        # 5. Skor kualitas keseluruhan (rata-rata tertimbang)
        if weights:
            overall_score = sum(scores[k] * weights.get(k, 1.0) for k in scores) / sum(weights.get(k, 1.0) for k in scores)
        else:
            overall_score = np.mean(list(scores.values()))
        scores['overall'] = overall_score
        
        # Keputusan
        if self.dynamic_gesture:
            # Untuk gestur dinamis: abaikan ambang batas stabilitas, fokus pada kualitas keseluruhan
            is_valid = (
                completeness_score >= 1.0 and
                position_score >= 0.5 and
                overall_score >= self.min_confidence
            )
        else:
            # Untuk gestur statis: butuh stabilitas
            is_valid = (
                completeness_score >= 1.0 and
                stability_score >= self.min_stability and
                position_score >= 0.5 and
                overall_score >= self.min_confidence
            )
        
        # Pesan umpan balik
        if is_valid:
            feedback = f"✅ EXCELLENT! Quality: {overall_score*100:.0f}%"
        elif overall_score >= 0.6:
            feedback = f"⚠️ MODERATE Quality: {overall_score*100:.0f}% | " + " | ".join(feedback_parts)
        else:
            feedback = f"❌ POOR Quality: {overall_score*100:.0f}% | " + " | ".join(feedback_parts)
        
        return is_valid, scores, feedback
    
    def _check_completeness(self, landmarks: np.ndarray) -> float:
        """Cek apakah semua landmark terdeteksi dengan baik"""
        if landmarks is None:
            return 0.0
        
        # Pastikan landmarks adalah array numpy dengan tipe float
        if not isinstance(landmarks, np.ndarray):
            try:
                landmarks = np.array(landmarks, dtype=np.float32)
            except:
                return 0.0
        
        # Konversi ke float jika belum
        if landmarks.dtype == object or not np.issubdtype(landmarks.dtype, np.number):
            try:
                landmarks = landmarks.astype(np.float32)
            except:
                return 0.0
        
        # Cek untuk NaN atau nilai tidak valid
        try:
            valid_points = np.isfinite(landmarks).all(axis=1).sum()
            total_points = landmarks.shape[0]
            return valid_points / total_points
        except:
            # Fallback: hanya cek jika semua titik ada
            return 1.0 if landmarks.shape[0] >= 21 else 0.0
    
    def _check_stability(self, landmarks: np.ndarray) -> float:
        """Cek stabilitas tangan (tidak goyang)"""
        # Pastikan array numpy
        if not isinstance(landmarks, np.ndarray):
            try:
                landmarks = np.array(landmarks, dtype=np.float32)
            except:
                return 1.0
        
        if landmarks.dtype == object or not np.issubdtype(landmarks.dtype, np.number):
            try:
                landmarks = landmarks.astype(np.float32)
            except:
                return 1.0
        
        if len(self.previous_landmarks) == 0:
            self.previous_landmarks.append(landmarks.copy())
            return 1.0
        
        # Bandingkan dengan 3 frame sebelumnya
        recent_frames = self.previous_landmarks[-3:]
        
        if len(recent_frames) == 0:
            return 1.0
        
        # Hitung rata-rata gerakan PER LANDMARK (bukan total)
        movements = []
        for prev_lm in recent_frames:
            try:
                if prev_lm.shape == landmarks.shape:
                    # Hitung jarak per-landmark dan rata-ratakan
                    per_landmark_distances = np.linalg.norm(landmarks - prev_lm, axis=1)
                    avg_per_landmark = np.mean(per_landmark_distances)
                    movements.append(avg_per_landmark)
            except:
                continue
        
        if len(movements) == 0:
            return 1.0
        
        avg_movement = np.mean(movements)
        
        # Normalisasi: gerakan < 0.02 = sangat stabil, > 0.05 = tidak stabil
        # Menggunakan ambang batas yang lebih lembut untuk memperhitungkan gerakan mikro tangan alami
        if avg_movement < 0.015:
            stability_score = 1.0  # Stabilitas sempurna
        elif avg_movement < 0.03:
            stability_score = 0.95 - ((avg_movement - 0.015) / 0.015) * 0.1  # 95-85%
        elif avg_movement < 0.05:
            stability_score = 0.85 - ((avg_movement - 0.03) / 0.02) * 0.15  # 85-70%
        else:
            # Penurunan linear setelah 0.05
            stability_score = max(0.0, 0.7 - ((avg_movement - 0.05) / 0.05) * 0.7)  # 70-0%
        
        # Perbarui riwayat
        self.previous_landmarks.append(landmarks.copy())
        if len(self.previous_landmarks) > 5:
            self.previous_landmarks.pop(0)
        
        return stability_score
    
    def _check_position(self, landmarks: np.ndarray) -> float:
        """Cek apakah posisi tangan dalam area yang baik"""
        if landmarks is None:
            return 0.0
        
        # Pastikan array numpy
        if not isinstance(landmarks, np.ndarray):
            try:
                landmarks = np.array(landmarks, dtype=np.float32)
            except:
                return 0.5  # Skor netral default
        
        if landmarks.dtype == object or not np.issubdtype(landmarks.dtype, np.number):
            try:
                landmarks = landmarks.astype(np.float32)
            except:
                return 0.5
        
        try:
            # Dapatkan bounding box
            x_coords = landmarks[:, 0]
            y_coords = landmarks[:, 1]
            
            min_x, max_x = np.min(x_coords), np.max(x_coords)
            min_y, max_y = np.min(y_coords), np.max(y_coords)
            
            # Cek jika tangan di tengah (rentang 0.2 - 0.8 bagus)
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            
            # Ideal: tengah sekitar 0.5
            x_score = 1.0 - abs(center_x - 0.5) * 2  # Penalti untuk di luar tengah
            y_score = 1.0 - abs(center_y - 0.5) * 2
            
            # Cek ukuran (tidak terlalu kecil, tidak terlalu besar)
            hand_width = max_x - min_x
            hand_height = max_y - min_y
            
            # Ukuran ideal: 0.3 - 0.6 dari frame
            size = max(hand_width, hand_height)
            if 0.3 <= size <= 0.6:
                size_score = 1.0
            elif size < 0.3:
                size_score = size / 0.3
            else:
                size_score = max(0.0, 1.0 - (size - 0.6) / 0.4)
            
            # Skor gabungan
            position_score = (x_score * 0.3 + y_score * 0.3 + size_score * 0.4)
            
            return max(0.0, min(1.0, position_score))
        except:
            return 0.5  # Skor netral default saat error

    def _check_blur(self, frame: np.ndarray, landmarks: np.ndarray) -> float:
        try:
            if frame is None or landmarks is None:
                return 1.0

            # Pastikan array numpy
            if not isinstance(landmarks, np.ndarray):
                try:
                    landmarks = np.array(landmarks, dtype=np.float32)
                except:
                    return 1.0

            if landmarks.dtype == object or not np.issubdtype(landmarks.dtype, np.number):
                try:
                    landmarks = landmarks.astype(np.float32)
                except:
                    return 1.0

            if landmarks.shape[0] == 0 or frame.ndim < 2:
                return 1.0

            h, w = frame.shape[:2]

            # Gunakan koordinat x, y ternormalisasi (0-1) dan klip ke rentang valid
            x_coords = np.clip(landmarks[:, 0], 0.0, 1.0)
            y_coords = np.clip(landmarks[:, 1], 0.0, 1.0)

            min_x = int(max(0, np.min(x_coords) * w))
            max_x = int(min(w - 1, np.max(x_coords) * w))
            min_y = int(max(0, np.min(y_coords) * h))
            max_y = int(min(h - 1, np.max(y_coords) * h))

            # Perluas bounding box sedikit untuk menyertakan konteks
            margin_x = int((max_x - min_x) * 0.25)
            margin_y = int((max_y - min_y) * 0.25)
            min_x = max(0, min_x - margin_x)
            max_x = min(w - 1, max_x + margin_x)
            min_y = max(0, min_y - margin_y)
            max_y = min(h - 1, max_y + margin_y)

            if max_x - min_x <= 1 or max_y - min_y <= 1:
                return 1.0

            hand_crop = frame[min_y:max_y, min_x:max_x]
            if hand_crop.size == 0:
                return 1.0

            gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
            variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            # Jaga terhadap konfigurasi tidak valid
            if self.blur_max_variance <= self.blur_min_variance:
                return 1.0

            if variance >= self.blur_max_variance:
                blur_score = 1.0
            elif variance <= self.blur_min_variance:
                blur_score = 0.0
            else:
                blur_score = (variance - self.blur_min_variance) / (self.blur_max_variance - self.blur_min_variance)

            return float(max(0.0, min(1.0, blur_score)))
        except:
            return 1.0
    
    def _check_consistency(self, landmarks: np.ndarray, gesture: str) -> float:
        """Cek konsistensi dengan baseline reference"""
        if gesture not in self.baselines:
            return 1.0  # Tidak ada baseline, asumsikan OK
        
        try:
            baseline = np.array(self.baselines[gesture]['landmarks'], dtype=np.float32)
            
            # Pastikan landmarks adalah array numpy
            if not isinstance(landmarks, np.ndarray):
                landmarks = np.array(landmarks, dtype=np.float32)
            
            if landmarks.dtype == object or not np.issubdtype(landmarks.dtype, np.number):
                landmarks = landmarks.astype(np.float32)
            
            # Normalisasi keduanya (hapus perbedaan posisi/skala)
            landmarks_norm = self._normalize_landmarks(landmarks)
            baseline_norm = self._normalize_landmarks(baseline)
            
            # Hitung kemiripan (jarak lebih rendah = kemiripan lebih tinggi)
            distance = np.linalg.norm(landmarks_norm - baseline_norm)
            
            # Konversi ke skor kemiripan (0-1)
            # Jarak < 0.5 = sangat mirip
            similarity = max(0.0, 1.0 - (distance / 1.0))
            
            return similarity
        except:
            return 1.0  # Jika error, asumsikan OK
    
    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Normalisasi landmarks untuk perbandingan yang adil"""
        try:
            # Pastikan array numpy
            if not isinstance(landmarks, np.ndarray):
                landmarks = np.array(landmarks, dtype=np.float32)
            
            if landmarks.dtype == object or not np.issubdtype(landmarks.dtype, np.number):
                landmarks = landmarks.astype(np.float32)
            
            # Pusatkan di titik asal
            center = np.mean(landmarks, axis=0)
            normalized = landmarks - center
            
            # Skalakan ke ukuran unit
            scale = np.max(np.abs(normalized))
            if scale > 0:
                normalized = normalized / scale
            
            return normalized
        except:
            return landmarks  # Kembalikan apa adanya jika error
    
    def add_to_history(self, gesture: str, landmarks: np.ndarray):
        """Tambahkan sampel ke riwayat untuk pelacakan konsistensi"""
        # Pastikan array numpy
        if not isinstance(landmarks, np.ndarray):
            try:
                landmarks = np.array(landmarks, dtype=np.float32)
            except:
                return
        
        if landmarks.dtype == object or not np.issubdtype(landmarks.dtype, np.number):
            try:
                landmarks = landmarks.astype(np.float32)
            except:
                return
        
        if gesture not in self.sample_history:
            self.sample_history[gesture] = []
        
        self.sample_history[gesture].append(landmarks.copy())
        
        # Simpan hanya 50 sampel terakhir
        if len(self.sample_history[gesture]) > 50:
            self.sample_history[gesture].pop(0)
    
    def get_gesture_statistics(self, gesture: str) -> Dict:
        """Dapatkan statistik untuk gestur tertentu"""
        if gesture not in self.sample_history or len(self.sample_history[gesture]) < 2:
            return {
                'count': 0,
                'mean_distance': 0.0,
                'std_distance': 0.0,
                'consistency': 1.0
            }
        
        samples = self.sample_history[gesture]
        
        # Hitung jarak berpasangan
        distances = []
        for i in range(len(samples) - 1):
            norm_i = self._normalize_landmarks(samples[i])
            norm_j = self._normalize_landmarks(samples[i + 1])
            dist = np.linalg.norm(norm_i - norm_j)
            distances.append(dist)
        
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        # Skor konsistensi (jarak lebih rendah = lebih konsisten)
        consistency = max(0.0, 1.0 - (mean_dist / 0.5))
        
        return {
            'count': len(samples),
            'mean_distance': mean_dist,
            'std_distance': std_dist,
            'consistency': consistency
        }
    
    def reset(self):
        """Reset status validator"""
        self.previous_landmarks = []
    
    def create_baseline(self, gesture: str, landmarks: np.ndarray):
        """Buat referensi baseline untuk gestur"""
        # Pastikan array numpy
        if not isinstance(landmarks, np.ndarray):
            try:
                landmarks = np.array(landmarks, dtype=np.float32)
            except:
                print(f"❌ Error: Invalid landmarks format for baseline")
                return
        
        if landmarks.dtype == object or not np.issubdtype(landmarks.dtype, np.number):
            try:
                landmarks = landmarks.astype(np.float32)
            except:
                print(f"❌ Error: Cannot convert landmarks to numeric format")
                return
        
        os.makedirs(self.baseline_dir, exist_ok=True)
        
        baseline_data = {
            'gesture': gesture,
            'landmarks': landmarks.tolist(),
            'timestamp': str(np.datetime64('now'))
        }
        
        baseline_path = os.path.join(self.baseline_dir, f'{gesture}.json')
        with open(baseline_path, 'w') as f:
            json.dump(baseline_data, f, indent=2)
        
        # Perbarui baseline internal
        self.baselines[gesture] = baseline_data
        
        print(f"✅ Baseline created for '{gesture}': {baseline_path}")


class VisualFeedback:
    """Umpan balik visual untuk validasi kualitas"""
    
    @staticmethod
    def draw_quality_indicator(
        frame: np.ndarray,
        scores: Dict[str, float],
        feedback: str,
        position: Tuple[int, int] = (10, 30)
    ) -> np.ndarray:
        """Gambar indikator kualitas pada frame"""
        frame = frame.copy()
        x, y = position
        
        # Bar kualitas keseluruhan
        overall = scores.get('overall', 0.0)
        bar_width = 200
        bar_height = 20
        
        # Latar belakang
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (50, 50, 50), -1)
        
        # Bar kualitas (warna berdasarkan skor)
        if overall >= 0.8:
            color = (0, 255, 0)  # Hijau
        elif overall >= 0.6:
            color = (0, 255, 255)  # Kuning
        else:
            color = (0, 0, 255)  # Merah
        
        fill_width = int(bar_width * overall)
        cv2.rectangle(frame, (x, y), (x + fill_width, y + bar_height), color, -1)
        
        # Batas
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (255, 255, 255), 2)
        
        # Teks
        text = f"Quality: {overall*100:.0f}%"
        cv2.putText(frame, text, (x + bar_width + 10, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Skor individu
        y_offset = y + bar_height + 10
        for metric, score in scores.items():
            if metric != 'overall':
                if score >= 0.8:
                    icon = "✓"
                    color = (0, 255, 0)
                elif score >= 0.6:
                    icon = "~"
                    color = (0, 255, 255)
                else:
                    icon = "✗"
                    color = (0, 0, 255)
                
                label = "Sharpness" if metric == "blur" else metric.capitalize()
                text = f"{icon} {label}: {score*100:.0f}%"
                cv2.putText(frame, text, (x, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                y_offset += 20
        
        # Pesan umpan balik
        y_offset += 10
        cv2.putText(frame, feedback, (x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    @staticmethod
    def draw_enhanced_landmarks(
        frame: np.ndarray,
        landmarks: List[Dict],
        hand_type: str,
        quality_scores: Dict[str, float] = None
    ) -> np.ndarray:
        """Gambar landmark dengan kode warna berdasarkan kualitas"""
        if not landmarks:
            return frame
            
        frame = frame.copy()
        h, w, _ = frame.shape
        
        # Dapatkan skor kualitas
        overall_quality = quality_scores.get('overall', 1.0) if quality_scores else 1.0
        stability_score = quality_scores.get('stability', 1.0) if quality_scores else 1.0
        position_score = quality_scores.get('position', 1.0) if quality_scores else 1.0
        
        # Definisi koneksi landmark
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # Ibu jari
            (0, 5), (5, 6), (6, 7), (7, 8),      # Telunjuk
            (0, 9), (9, 10), (10, 11), (11, 12), # Jari tengah
            (0, 13), (13, 14), (14, 15), (15, 16), # Jari manis
            (0, 17), (17, 18), (18, 19), (19, 20), # Kelingking
        ]
        
        # Kode warna berdasarkan kualitas
        def get_landmark_color(landmark_idx, landmark):
            """Dapatkan warna untuk landmark tertentu berdasarkan metrik kualitas"""
            # Warna dasar tergantung kualitas keseluruhan
            if overall_quality >= 0.8:
                base_color = (0, 255, 0)  # Hijau - sangat baik
            elif overall_quality >= 0.6:
                base_color = (0, 255, 255)  # Kuning - baik
            else:
                base_color = (0, 165, 255)  # Oranye - perlu perbaikan
            
            # Pewarnaan khusus untuk masalah tertentu
            x, y = landmark['x'] * w, landmark['y'] * h
            
            # Cek jika landmark di tepi (masalah posisi)
            edge_threshold = 0.05
            if (landmark['x'] < edge_threshold or landmark['x'] > 1-edge_threshold or
                landmark['y'] < edge_threshold or landmark['y'] > 1-edge_threshold):
                return (0, 0, 255)  # Merah untuk landmark tepi
            
            # Cek keyakinan (jika tersedia)
            if 'visibility' in landmark and landmark['visibility'] < 0.5:
                return (128, 128, 128)  # Abu-abu untuk keyakinan rendah
            
            # Landmark ujung jari - lebih penting
            fingertips = [4, 8, 12, 16, 20]
            if landmark_idx in fingertips:
                if stability_score < 0.5:
                    return (255, 0, 255)  # Magenta untuk ujung jari tidak stabil
                else:
                    # Versi lebih terang dari warna dasar
                    return tuple(min(255, int(c * 1.2)) for c in base_color)
            
            return base_color
        
        # Gambar koneksi terlebih dahulu
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_lm = landmarks[start_idx]
                end_lm = landmarks[end_idx]
                
                start_point = (int(start_lm['x'] * w), int(start_lm['y'] * h))
                end_point = (int(end_lm['x'] * w), int(end_lm['y'] * h))
                
                # Warna garis berdasarkan kualitas rata-rata landmark yang terhubung
                start_color = get_landmark_color(start_idx, start_lm)
                end_color = get_landmark_color(end_idx, end_lm)
                line_color = tuple((s + e) // 2 for s, e in zip(start_color, end_color))
                
                # Ketebalan garis berdasarkan stabilitas
                thickness = 3 if stability_score >= 0.8 else (2 if stability_score >= 0.6 else 1)
                cv2.line(frame, start_point, end_point, line_color, thickness)
        
        # Gambar landmark (titik)
        for i, landmark in enumerate(landmarks):
            x, y = int(landmark['x'] * w), int(landmark['y'] * h)
            color = get_landmark_color(i, landmark)
            
            # Ukuran titik berdasarkan kepentingan dan kualitas
            fingertips = [4, 8, 12, 16, 20]
            if i in fingertips:
                radius = 6  # Lebih besar untuk ujung jari
            elif i == 0:
                radius = 8  # Terbesar untuk pergelangan tangan
            else:
                radius = 4  # Normal untuk sendi lain
            
            # Sesuaikan radius berdasarkan kualitas
            if overall_quality < 0.5:
                radius = max(2, radius - 2)
            
            cv2.circle(frame, (x, y), radius, color, -1)
            cv2.circle(frame, (x, y), radius + 1, (255, 255, 255), 1)  # Batas putih
            
            # Tambahkan indeks landmark untuk debugging (opsional)
            if overall_quality < 0.4:  # Hanya tampilkan angka saat kualitas sangat rendah
                cv2.putText(frame, str(i), (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Tambahkan legenda kualitas
        # Pindahkan ke kanan bawah untuk menghindari tumpang tindih dengan statistik kualitas di kiri
        # Tombol REC ada di kanan bawah (h-60), jadi kita tempatkan di atasnya
        legend_width = 200
        legend_height = 120
        legend_x = w - legend_width - 10
        legend_y = h - legend_height - 70  # Padding 10px di atas tombol REC (yang mulai di h-60)
        
        cv2.rectangle(frame, (legend_x, legend_y), (legend_x + legend_width, legend_y + legend_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (legend_x, legend_y), (legend_x + legend_width, legend_y + legend_height), (255, 255, 255), 1)
        
        cv2.putText(frame, "Landmark Colors:", (legend_x + 5, legend_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "Green: Excellent", (legend_x + 5, legend_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        cv2.putText(frame, "Yellow: Good", (legend_x + 5, legend_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        cv2.putText(frame, "Orange: Fair", (legend_x + 5, legend_y + 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1)
        cv2.putText(frame, "Red: Edge/Issue", (legend_x + 5, legend_y + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        cv2.putText(frame, "Gray: Low Confidence", (legend_x + 5, legend_y + 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (128, 128, 128), 1)
        cv2.putText(frame, "Magenta: Unstable Tips", (legend_x + 5, legend_y + 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)
        
        return frame
    
    @staticmethod
    def draw_reference_comparison(
        frame: np.ndarray,
        current_landmarks: np.ndarray,
        reference_landmarks: np.ndarray,
        position: Tuple[int, int] = (500, 30)
    ) -> np.ndarray:
        """Gambar perbandingan berdampingan dengan referensi"""
        frame = frame.copy()
        x, y = position
        
        # Gambar kerangka mini untuk referensi
        size = 150
        
        # Latar belakang
        cv2.rectangle(frame, (x, y), (x + size, y + size), (30, 30, 30), -1)
        cv2.putText(frame, "Reference", (x + 5, y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Gambar kerangka referensi
        if reference_landmarks is not None:
            VisualFeedback._draw_mini_skeleton(
                frame, reference_landmarks, (x + size//2, y + size//2), size//2, (0, 255, 0)
            )
        
        return frame
    
    @staticmethod
    def _draw_mini_skeleton(
        frame: np.ndarray,
        landmarks: np.ndarray,
        center: Tuple[int, int],
        scale: int,
        color: Tuple[int, int, int]
    ):
        """Gambar kerangka tangan mini"""
        # Normalisasi landmark
        lm_norm = landmarks - np.mean(landmarks, axis=0)
        lm_max = np.max(np.abs(lm_norm))
        if lm_max > 0:
            lm_norm = lm_norm / lm_max
        
        # Skala dan terjemahkan
        lm_scaled = lm_norm[:, :2] * scale * 0.8
        lm_scaled[:, 0] += center[0]
        lm_scaled[:, 1] += center[1]
        
        # Gambar koneksi (disederhanakan)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Ibu jari
            (0, 5), (5, 6), (6, 7), (7, 8),  # Telunjuk
            (0, 9), (9, 10), (10, 11), (11, 12),  # Jari tengah
            (0, 13), (13, 14), (14, 15), (15, 16),  # Jari manis
            (0, 17), (17, 18), (18, 19), (19, 20),  # Kelingking
        ]
        
        for start, end in connections:
            pt1 = tuple(lm_scaled[start].astype(int))
            pt2 = tuple(lm_scaled[end].astype(int))
            cv2.line(frame, pt1, pt2, color, 1)
        
        # Gambar titik
        for point in lm_scaled:
            pt = tuple(point.astype(int))
            cv2.circle(frame, pt, 2, color, -1)
