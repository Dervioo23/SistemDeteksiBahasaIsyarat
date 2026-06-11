import cv2
import mediapipe as mp
import numpy as np
from typing import List, Optional, Tuple, Dict


class LandmarkExtractor:
    """Kelas untuk ekstraksi landmark tangan menggunakan MediaPipe"""
    
    def __init__(self, 
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5,
                 max_num_hands: int = 2,
                 use_z_coordinate: bool = True,
                 experimental_mode: bool = False):
        """
        Inisialisasi MediaPipe Hands dengan dukungan 2 tangan
        
        Args:
            min_detection_confidence: Keyakinan minimum untuk deteksi tangan
            min_tracking_confidence: Keyakinan minimum untuk pelacakan
            max_num_hands: Jumlah maksimum tangan yang dideteksi
            use_z_coordinate: Apakah akan menyertakan koordinat Z (kedalaman)
            experimental_mode: Aktifkan fitur eksperimental (logging, perbandingan)
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1, # FORCE: Complex Model (Better Accuracy)
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.num_landmarks = 21
        self.max_num_hands = max_num_hands
        
        # BARU: Kontrol koordinat-Z
        self.use_z_coordinate = use_z_coordinate
        self.experimental_mode = experimental_mode
        
        # Dimensi fitur berdasarkan penggunaan koordinat Z
        self.features_per_landmark = 3 if use_z_coordinate else 2
        self.features_per_hand = self.num_landmarks * self.features_per_landmark  # 63 atau 42
        self.total_features = self.features_per_hand * 2  # 126 atau 84
        
        # Pelacakan eksperimental
        if experimental_mode:
            self.z_stability_log = []
            self.comparison_log = []
            print(f"🧪 EXPERIMENTAL MODE: Z-coordinate {'ENABLED' if use_z_coordinate else 'DISABLED'}")
            print(f"📏 Feature dimensions: {self.total_features} features per frame")
            print(f"   - Per landmark: {self.features_per_landmark} coordinates")
            print(f"   - Per hand: {self.features_per_hand} features")
            print(f"   - Both hands: {self.total_features} features")
    
    def extract_landmarks(self, frame: np.ndarray) -> Tuple[Optional[List[Dict]], np.ndarray]:
        """Ekstrak landmarks dari frame (kompatibilitas mundur)"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        annotated_frame = frame.copy()
        
        if results.multi_hand_landmarks:
            landmarks_list = []
            
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    if self.use_z_coordinate:
                        landmarks.append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z
                        })
                    else:
                        landmarks.append({
                            'x': landmark.x,
                            'y': landmark.y
                        })
                    
                    # Eksperimental: Lacak stabilitas Z
                    if self.experimental_mode and hasattr(self, 'z_stability_log'):
                        self.z_stability_log.append(landmark.z)
                
                landmarks_list.append(landmarks)
            
            return landmarks_list, annotated_frame
        
        return None, annotated_frame
    
    def extract_both_hands(self, frame: np.ndarray) -> Tuple[Dict[str, Optional[List[Dict]]], np.ndarray, int]:
        """
        Ekstrak landmarks untuk kedua tangan dengan deteksi otomatis dan pelabelan
        
        Returns:
            Tuple dari (hands_dict, annotated_frame, num_hands_detected)
            hands_dict: {'right': landmarks atau None, 'left': landmarks atau None}
            annotated_frame: Frame dengan gambar
            num_hands_detected: Jumlah tangan yang terdeteksi (0, 1, atau 2)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        annotated_frame = frame.copy()
        
        # Inisialisasi dengan None
        hands_dict = {
            'right': None,
            'left': None
        }
        
        num_hands_detected = 0
        
        if results.multi_hand_landmarks and results.multi_handedness:
            num_hands_detected = len(results.multi_hand_landmarks)
            
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Gambar landmarks
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Ekstrak koordinat
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    if self.use_z_coordinate:
                        landmarks.append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z
                        })
                    else:
                        landmarks.append({
                            'x': landmark.x,
                            'y': landmark.y
                        })
                    
                    # Eksperimental: Lacak stabilitas Z untuk tangan ini
                    if self.experimental_mode and hasattr(self, 'z_stability_log'):
                        self.z_stability_log.append(landmark.z)
                
                # Tentukan tangan kiri atau kanan
                # MediaPipe mengembalikan "Left" atau "Right" berdasarkan perspektif orang
                hand_label = handedness.classification[0].label
                
                if hand_label == 'Right':
                    hands_dict['right'] = landmarks
                else:
                    hands_dict['left'] = landmarks
        
        return hands_dict, annotated_frame, num_hands_detected
    
    def extract_single_hand(self, frame: np.ndarray) -> Tuple[Optional[List[Dict]], np.ndarray]:
        """Ekstrak landmarks untuk satu tangan saja"""
        landmarks_list, annotated_frame = self.extract_landmarks(frame)
        
        if landmarks_list and len(landmarks_list) > 0:
            return landmarks_list[0], annotated_frame
        
        return None, annotated_frame
    
    def landmarks_to_array(self, landmarks: List[Dict]) -> np.ndarray:
        """Konversi dict landmarks ke array numpy"""
        if self.use_z_coordinate:
            arr = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks])
        else:
            arr = np.array([[lm['x'], lm['y']] for lm in landmarks])
        return arr
    
    def normalize_landmarks(self, landmarks: List[Dict]) -> List[Dict]:
        """Normalisasi landmarks relatif terhadap pergelangan tangan"""
        if not landmarks or len(landmarks) == 0:
            return landmarks
        
        wrist = landmarks[0]
        wrist_x, wrist_y = wrist['x'], wrist['y']
        
        normalized = []
        for lm in landmarks:
            if self.use_z_coordinate:
                wrist_z = wrist['z']
                normalized.append({
                    'x': lm['x'] - wrist_x,
                    'y': lm['y'] - wrist_y,
                    'z': lm['z'] - wrist_z
                })
            else:
                normalized.append({
                    'x': lm['x'] - wrist_x,
                    'y': lm['y'] - wrist_y
                })
        
        return normalized
    
    def is_hand_detected(self, frame: np.ndarray) -> bool:
        """Cek apakah ada tangan terdeteksi"""
        landmarks, _ = self.extract_single_hand(frame)
        return landmarks is not None
    
    def flatten_landmarks(self, landmarks: List[Dict]) -> List[float]:
        """Ratakan landmarks ke array 1D [x1,y1,z1,x2,y2,z2,...] atau [x1,y1,x2,y2,...]"""
        flattened = []
        for lm in landmarks:
            if self.use_z_coordinate:
                flattened.extend([lm['x'], lm['y'], lm['z']])
            else:
                flattened.extend([lm['x'], lm['y']])
        return flattened
    
    def get_zero_landmarks(self) -> List[Dict]:
        """Hasilkan landmarks berisi nol untuk tangan yang tidak terdeteksi"""
        if self.use_z_coordinate:
            return [{'x': 0.0, 'y': 0.0, 'z': 0.0} for _ in range(21)]
        else:
            return [{'x': 0.0, 'y': 0.0} for _ in range(21)]
    
    def get_both_hands_with_fallback(self, hands_dict: Dict[str, Optional[List[Dict]]]) -> Dict[str, List[Dict]]:
        """
        Dapatkan data kedua tangan dengan fallback ke nol jika tidak terdeteksi
        
        Args:
            hands_dict: Dict dari extract_both_hands
            
        Returns:
            Dict dengan right dan left selalu berisi data (nol jika None)
        """
        return {
            'right': hands_dict['right'] if hands_dict['right'] is not None else self.get_zero_landmarks(),
            'left': hands_dict['left'] if hands_dict['left'] is not None else self.get_zero_landmarks()
        }
    
    def flatten_both_hands(self, hands_dict: Dict[str, List[Dict]]) -> List[float]:
        """
        Ratakan landmarks kedua tangan ke array 1D
        Format: [fitur_tangan_kanan, fitur_tangan_kiri]
        - Dengan Z: total 126 fitur (63 per tangan)
        - Tanpa Z: total 84 fitur (42 per tangan)
        
        Args:
            hands_dict: Dict dengan kunci 'right' dan 'left'
            
        Returns:
            Array rata dengan total fitur sesuai mode
        """
        flattened = []
        
        # Tangan kanan dulu
        for lm in hands_dict['right']:
            if self.use_z_coordinate:
                flattened.extend([lm['x'], lm['y'], lm['z']])
            else:
                flattened.extend([lm['x'], lm['y']])
        
        # Tangan kiri kedua
        for lm in hands_dict['left']:
            if self.use_z_coordinate:
                flattened.extend([lm['x'], lm['y'], lm['z']])
            else:
                flattened.extend([lm['x'], lm['y']])
        
        return flattened  # Total fitur: self.total_features
    
    def is_any_hand_detected(self, frame: np.ndarray) -> Tuple[bool, int]:
        """
        Cek apakah ada tangan terdeteksi dan berapa banyak
        
        Returns:
            Tuple dari (terdeteksi, jumlah_tangan)
        """
        hands_dict, _, num_hands = self.extract_both_hands(frame)
        return num_hands > 0, num_hands
    
    def get_z_stability_analysis(self) -> Dict:
        """
        Analisis stabilitas koordinat Z (hanya mode eksperimental)
        
        Returns:
            Dict dengan statistik stabilitas koordinat Z
        """
        if not self.experimental_mode or not hasattr(self, 'z_stability_log'):
            return {}
        
        if len(self.z_stability_log) < 10:
            return {'error': 'Insufficient data for analysis'}
        
        z_values = np.array(self.z_stability_log)
        
        return {
            'total_samples': len(z_values),
            'mean_z': float(np.mean(z_values)),
            'std_z': float(np.std(z_values)),
            'min_z': float(np.min(z_values)),
            'max_z': float(np.max(z_values)),
            'range_z': float(np.max(z_values) - np.min(z_values)),
            'coefficient_of_variation': float(np.std(z_values) / np.mean(z_values)) if np.mean(z_values) != 0 else float('inf'),
            'stability_score': 1.0 - min(1.0, np.std(z_values) / 0.1)  # Higher std = lower stability
        }
    
    def compare_with_without_z(self, frame: np.ndarray) -> Dict:
        """
        Bandingkan ekstraksi dengan dan tanpa koordinat Z (eksperimental)
        
        Returns:
            Data perbandingan untuk analisis
        """
        if not self.experimental_mode:
            return {}
        
        # Ekstrak dengan Z
        original_use_z = self.use_z_coordinate
        
        # Ekstrak dengan Z
        self.use_z_coordinate = True
        hands_with_z, _, num_hands_z = self.extract_both_hands(frame)
        
        # Ekstrak tanpa Z  
        self.use_z_coordinate = False
        hands_without_z, _, num_hands_no_z = self.extract_both_hands(frame)
        
        # Kembalikan pengaturan asli
        self.use_z_coordinate = original_use_z
        
        # Bandingkan jumlah fitur
        comparison = {
            'num_hands_detected_with_z': num_hands_z,
            'num_hands_detected_without_z': num_hands_no_z,
            'detection_consistency': num_hands_z == num_hands_no_z
        }
        
        # Bandingkan dimensi fitur jika tangan terdeteksi
        if hands_with_z['right'] and hands_without_z['right']:
            with_z_features = len(self.flatten_both_hands({'right': hands_with_z['right'], 'left': hands_with_z['left'] or self.get_zero_landmarks()}))
            
            # Sementara set ke False untuk mendapatkan hitungan yang benar
            temp_use_z = self.use_z_coordinate
            self.use_z_coordinate = False
            without_z_features = len(self.flatten_both_hands({'right': hands_without_z['right'], 'left': hands_without_z['left'] or self.get_zero_landmarks()}))
            self.use_z_coordinate = temp_use_z
            
            comparison.update({
                'features_with_z': with_z_features,
                'features_without_z': without_z_features,
                'feature_reduction': with_z_features - without_z_features,
                'feature_reduction_percent': ((with_z_features - without_z_features) / with_z_features) * 100
            })
        
        return comparison
    
    def print_experimental_summary(self):
        """Cetak ringkasan hasil eksperimental"""
        if not self.experimental_mode:
            return
        
        print(f"\n🧪 EXPERIMENTAL SUMMARY")
        print(f"="*50)
        print(f"Mode: {'WITH Z-coordinate' if self.use_z_coordinate else 'WITHOUT Z-coordinate'}")
        print(f"Feature dimensions: {self.total_features} per frame")
        print(f"  - Per landmark: {self.features_per_landmark} coordinates")
        print(f"  - Per hand: {self.features_per_hand} features")
        
        # Analisis stabilitas Z
        z_analysis = self.get_z_stability_analysis()
        if z_analysis and 'error' not in z_analysis:
            print(f"\n📊 Z-Coordinate Stability Analysis:")
            print(f"  - Samples analyzed: {z_analysis['total_samples']}")
            print(f"  - Mean Z: {z_analysis['mean_z']:.4f}")
            print(f"  - Std Z: {z_analysis['std_z']:.4f}")
            print(f"  - Range: {z_analysis['range_z']:.4f}")
            print(f"  - Coefficient of Variation: {z_analysis['coefficient_of_variation']:.4f}")
            print(f"  - Stability Score: {z_analysis['stability_score']:.4f}")
            
            # Interpretasi
            if z_analysis['stability_score'] > 0.8:
                print(f"  ✅ Z-coordinate is STABLE - consider keeping it")
            elif z_analysis['stability_score'] > 0.6:
                print(f"  ⚠️  Z-coordinate is MODERATELY stable")
            else:
                print(f"  ❌ Z-coordinate is UNSTABLE - consider removing it")
    
    def close(self):
        """Lepaskan sumber daya MediaPipe"""
        if self.experimental_mode:
            self.print_experimental_summary()
        self.hands.close()
