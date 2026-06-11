import os
import json
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import argparse
import sys
# Tambahkan direktori root proyek ke path Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_collection.landmark_extractor import LandmarkExtractor


def load_sample(filepath: str) -> Optional[Dict]:
    """Muat sampel dari file JSON"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except:
        return None


def find_gesture_samples(gesture: str, participant_id: Optional[str] = None, 
                        category: str = 'alphabet') -> List[Tuple[str, Dict]]:
    """Temukan semua sampel untuk gesture tertentu"""
    base_dir = f'dataset/{category}'
    gesture_dir = os.path.join(base_dir, gesture)
    
    if not os.path.exists(gesture_dir):
        return []
    
    samples = []
    for filename in os.listdir(gesture_dir):
        if not filename.endswith('.json'):
            continue
        
        # Filter berdasarkan peserta jika ditentukan
        if participant_id and not filename.startswith(participant_id):
            continue
        
        filepath = os.path.join(gesture_dir, filename)
        data = load_sample(filepath)
        
        if data:
            samples.append((filepath, data))
    
    return samples


def get_best_quality_sample(samples: List[Tuple[str, Dict]]) -> Optional[Tuple[str, Dict]]:
    """Dapatkan sampel dengan skor kualitas terbaik"""
    if not samples:
        return None
    
    best_sample = None
    best_quality = -1
    
    for filepath, data in samples:
        quality = data.get('metadata', {}).get('quality_score', 0)
        if quality > best_quality:
            best_quality = quality
            best_sample = (filepath, data)
    
    return best_sample


def landmarks_to_numpy(landmarks_list: List) -> np.ndarray:
    """Konversi landmark dari list ke numpy array"""
    if not landmarks_list:
        return np.zeros((21, 3), dtype=np.float32)
    
    # Ambil frame pertama
    frame = landmarks_list[0] if isinstance(landmarks_list, list) else landmarks_list
    
    # Periksa format elemen pertama
    if isinstance(frame, list):
        first_elem = frame[0] if frame else None
        
        if isinstance(first_elem, dict):
            # Format: list of dicts [{'x': ..., 'y': ..., 'z': ...}, ...]
            coords = []
            for lm in frame:
                if isinstance(lm, dict):
                    coords.extend([lm.get('x', 0), lm.get('y', 0), lm.get('z', 0)])
                else:
                    coords.extend([float(lm[0]), float(lm[1]), float(lm[2])])
            return np.array(coords, dtype=np.float32).reshape(-1, 3)
        
        elif isinstance(first_elem, (int, float)):
            # Format: flat list [x, y, z, x, y, z, ...]
            return np.array(frame, dtype=np.float32).reshape(-1, 3)
        
        elif isinstance(first_elem, list):
            # Format: nested list [[x, y, z], [x, y, z], ...]
            return np.array(frame, dtype=np.float32)
    
    elif isinstance(frame, (int, float)):
        # Single flat list
        return np.array(landmarks_list, dtype=np.float32).reshape(-1, 3)
    
    # Fallback
    return np.zeros((21, 3), dtype=np.float32)


def draw_hand_skeleton(frame: np.ndarray, landmarks: np.ndarray, 
                       hand_type: str = 'right') -> np.ndarray:
    """Gambar kerangka tangan pada frame"""
    # Validasi landmark
    if landmarks is None or len(landmarks) == 0:
        return frame
    
    # Periksa jika semua landmark nol (tidak valid)
    if np.allclose(landmarks, 0):
        return frame
    
    # Normalisasi dan skala landmark untuk mengisi kanvas dengan baik
    # Dapatkan bounding box
    min_coords = np.min(landmarks[:, :2], axis=0)  # Hanya x, y
    max_coords = np.max(landmarks[:, :2], axis=0)
    
    # Pusatkan dan skala
    center = (min_coords + max_coords) / 2
    size = max_coords - min_coords
    scale = 0.8  # Gunakan 80% dari kanvas
    
    # Koneksi tangan MediaPipe
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Ibu jari
        (0, 5), (5, 6), (6, 7), (7, 8),  # Telunjuk
        (0, 9), (9, 10), (10, 11), (11, 12),  # Tengah
        (0, 13), (13, 14), (14, 15), (15, 16),  # Manis
        (0, 17), (17, 18), (18, 19), (19, 20),  # Kelingking
        (5, 9), (9, 13), (13, 17)  # Telapak
    ]
    
    h, w = frame.shape[:2]
    
    # Normalisasi dan skala landmark ke kanvas
    landmarks_scaled = landmarks.copy()
    landmarks_scaled[:, :2] = (landmarks_scaled[:, :2] - center) / max(size[0], size[1]) * min(w, h) * scale
    landmarks_scaled[:, :2] += [w/2, h/2]  # Pusatkan di kanvas
    
    # Gambar koneksi
    for start_idx, end_idx in connections:
        if start_idx < len(landmarks_scaled) and end_idx < len(landmarks_scaled):
            start_point = landmarks_scaled[start_idx]
            end_point = landmarks_scaled[end_idx]
            
            # Gunakan koordinat berskala
            start_pos = (int(start_point[0]), int(start_point[1]))
            end_pos = (int(end_point[0]), int(end_point[1]))
            
            # Pastikan dalam batas
            if (0 <= start_pos[0] < w and 0 <= start_pos[1] < h and
                0 <= end_pos[0] < w and 0 <= end_pos[1] < h):
                cv2.line(frame, start_pos, end_pos, (0, 255, 0), 3)
    
    # Gambar landmark
    for i, landmark in enumerate(landmarks_scaled):
        pos = (int(landmark[0]), int(landmark[1]))
        
        # Periksa batas
        if not (0 <= pos[0] < w and 0 <= pos[1] < h):
            continue
        
        # Warna berbeda untuk bagian berbeda
        if i == 0:  # Pergelangan tangan
            color = (255, 0, 0)  # Biru
            radius = 10
        elif i in [4, 8, 12, 16, 20]:  # Ujung jari
            color = (0, 0, 255)  # Merah
            radius = 8
        else:
            color = (0, 255, 0)  # Hijau
            radius = 5
        
        cv2.circle(frame, pos, radius, color, -1)
        cv2.circle(frame, pos, radius+1, (255, 255, 255), 2)
    
    return frame


def create_reference_image(landmarks: np.ndarray, gesture: str, 
                          participant_id: str, quality: float) -> np.ndarray:
    """Buat gambar referensi dari landmark"""
    # Buat kanvas kosong
    img_size = 640
    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    # Pastikan landmark 2D
    if landmarks.ndim == 1:
        landmarks = landmarks.reshape(-1, 3)
    
    # Validasi landmark
    print(f"  🔍 Landmarks stats:")
    print(f"     Min values: {np.min(landmarks, axis=0)}")
    print(f"     Max values: {np.max(landmarks, axis=0)}")
    print(f"     All zero? {np.allclose(landmarks, 0)}")
    
    # Bagi landmark berdasarkan bentuk
    num_points = landmarks.shape[0]
    
    if num_points == 42:  # Kedua tangan (21+21)
        right_hand = landmarks[:21]
        left_hand = landmarks[21:]
    elif num_points == 21:  # Satu tangan
        right_hand = landmarks
        left_hand = None
    elif num_points > 21:  # Lebih dari 21 poin, asumsikan 21 pertama adalah tangan kanan
        right_hand = landmarks[:21]
        # Periksa jika ada tangan kiri (21 poin berikutnya)
        if num_points >= 42:
            left_hand = landmarks[21:42]
        else:
            left_hand = None
    else:  # Kurang dari 21 poin, gunakan apa yang ada
        right_hand = landmarks
        left_hand = None
    
    # Periksa apakah landmark valid
    landmarks_valid = not np.allclose(right_hand, 0)
    
    if landmarks_valid:
        # Gambar tangan kanan
        canvas = draw_hand_skeleton(canvas, right_hand, 'right')
        
        # Gambar tangan kiri jika tersedia
        if left_hand is not None and not np.allclose(left_hand, 0):
            canvas = draw_hand_skeleton(canvas, left_hand, 'left')
    else:
        # Tampilkan peringatan jika landmark tidak valid
        cv2.putText(canvas, "⚠️ INVALID LANDMARKS", 
                    (120, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        cv2.putText(canvas, "Sample data may be corrupted or empty", 
                    (100, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
        cv2.putText(canvas, "Solution: Re-collect this gesture", 
                    (120, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(canvas, "python run_collect_alphabet.py", 
                    (100, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Tambahkan info teks
    cv2.putText(canvas, f"Gesture: {gesture.upper()}", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(canvas, f"Participant: {participant_id}", 
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(canvas, f"Quality: {quality*100:.1f}%", 
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Tambahkan instruksi
    cv2.putText(canvas, "REFERENCE - Tiru gesture ini!", 
                (20, img_size - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(canvas, "Press any key to continue, Q to quit", 
                (20, img_size - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    
    return canvas


def save_reference_image(canvas: np.ndarray, gesture: str, participant_id: str):
    """Simpan gambar referensi ke disk"""
    ref_dir = 'reference_images'
    os.makedirs(ref_dir, exist_ok=True)
    
    filename = f"{participant_id}_{gesture}_reference.jpg"
    filepath = os.path.join(ref_dir, filename)
    
    cv2.imwrite(filepath, canvas)
    print(f"  💾 Reference image saved: {filepath}")


def view_gesture_reference(gesture: str, participant_id: Optional[str] = None,
                          category: str = 'alphabet', save: bool = False):
    """Lihat referensi untuk gesture tertentu"""
    print(f"\n  🔍 Loading reference for gesture '{gesture.upper()}'...")
    
    # Temukan sampel
    samples = find_gesture_samples(gesture, participant_id, category)
    
    if not samples:
        print(f"  ⚠️  No samples found for gesture '{gesture}'")
        if participant_id:
            print(f"      with participant '{participant_id}'")
        return
    
    print(f"  ✅ Found {len(samples)} samples")
    
    # Dapatkan sampel kualitas terbaik
    best_sample = get_best_quality_sample(samples)
    
    if not best_sample:
        print(f"  ⚠️  Could not load sample data")
        return
    
    filepath, data = best_sample
    
    # Ekstrak info
    metadata = data.get('metadata', {})
    landmarks_data = data.get('landmarks', [])
    
    participant = metadata.get('participant_id', 'unknown')
    quality = metadata.get('quality_score', 0)
    
    print(f"  📊 Best sample: {os.path.basename(filepath)}")
    print(f"  👤 Participant: {participant}")
    print(f"  ⭐ Quality: {quality*100:.1f}%")
    
    # Konversi landmark ke numpy
    try:
        landmarks = landmarks_to_numpy(landmarks_data)
        print(f"  📐 Landmarks shape: {landmarks.shape}")
    except Exception as e:
        print(f"  ❌ Error converting landmarks: {e}")
        print(f"  📋 Landmarks data type: {type(landmarks_data)}")
        if landmarks_data:
            print(f"  📋 First element type: {type(landmarks_data[0])}")
            if isinstance(landmarks_data[0], list) and landmarks_data[0]:
                print(f"  📋 First sub-element type: {type(landmarks_data[0][0])}")
        return False
    
    # Buat gambar referensi
    try:
        canvas = create_reference_image(landmarks, gesture, participant, quality)
    except Exception as e:
        print(f"  ❌ Error creating reference image: {e}")
        print(f"  📋 Landmarks shape: {landmarks.shape}")
        return False
    
    # Simpan jika diminta
    if save:
        save_reference_image(canvas, gesture, participant)
    
    # Tampilkan
    cv2.imshow(f'Reference: {gesture.upper()}', canvas)
    
    print(f"\n  👁️  Showing reference image...")
    print(f"  💡 TIRU GESTURE INI untuk consistency!")
    print(f"  Press any key to continue, Q to quit")
    
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return key != ord('q')


def interactive_reference_viewer():
    """Mode interaktif untuk melihat referensi"""
    print("\n" + "="*70)
    print("  👁️  GESTURE REFERENCE VIEWER")
    print("="*70)
    
    print("\n  Lihat bentuk gesture tangan dari samples yang sudah dikumpulkan")
    print("  Gunakan ini sebagai reference untuk consistency!\n")
    
    # Pilih kategori
    print("  Category:")
    print("  1. Alphabet (A-Z)")
    print("  2. Words")
    
    cat_choice = input("\n  Pilih category (1/2): ").strip()
    category = 'alphabet' if cat_choice == '1' else 'words'
    
    # Input gesture
    gesture = input(f"\n  Masukkan gesture (contoh: A atau halo): ").strip()
    
    if not gesture:
        print("  ⚠️  Gesture tidak boleh kosong!")
        return
    
    # Input peserta (opsional)
    participant = input(f"  Masukkan participant ID (optional, Enter to skip): ").strip()
    participant = participant if participant else None
    
    # Tanya untuk menyimpan
    save_choice = input(f"\n  Save reference image? (y/n): ").strip().lower()
    save = save_choice == 'y'
    
    # Lihat referensi
    view_gesture_reference(gesture, participant, category, save)


def batch_generate_references(category: str = 'alphabet', participant_id: Optional[str] = None):
    """Hasilkan gambar referensi untuk semua gesture"""
    print(f"\n  🎨 Generating reference images for {category}...")
    
    base_dir = f'dataset/{category}'
    
    if not os.path.exists(base_dir):
        print(f"  ⚠️  Directory not found: {base_dir}")
        return
    
    # Dapatkan semua folder gesture
    gestures = [d for d in os.listdir(base_dir) 
                if os.path.isdir(os.path.join(base_dir, d))]
    
    print(f"  Found {len(gestures)} gestures")
    
    generated = 0
    for gesture in gestures:
        samples = find_gesture_samples(gesture, participant_id, category)
        
        if not samples:
            continue
        
        best_sample = get_best_quality_sample(samples)
        
        if not best_sample:
            continue
        
        filepath, data = best_sample
        metadata = data.get('metadata', {})
        landmarks_data = data.get('landmarks', [])
        
        participant = metadata.get('participant_id', 'unknown')
        quality = metadata.get('quality_score', 0)
        
        landmarks = landmarks_to_numpy(landmarks_data)
        canvas = create_reference_image(landmarks, gesture, participant, quality)
        save_reference_image(canvas, gesture, participant)
        
        generated += 1
    
    print(f"\n  ✅ Generated {generated} reference images in 'reference_images/' folder")


def main():
    parser = argparse.ArgumentParser(description='View gesture references')
    parser.add_argument('--gesture', '-g', type=str, help='Gesture to view (e.g., A, halo)')
    parser.add_argument('--participant', '-p', type=str, help='Filter by participant ID')
    parser.add_argument('--category', '-c', type=str, choices=['alphabet', 'words'], 
                       default='alphabet', help='Category: alphabet or words')
    parser.add_argument('--save', '-s', action='store_true', help='Save reference image')
    parser.add_argument('--batch', '-b', action='store_true', help='Generate all references')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    if args.batch:
        # Mode batch
        batch_generate_references(args.category, args.participant)
    elif args.interactive or not args.gesture:
        # Mode interaktif
        interactive_reference_viewer()
    else:
        # Mode langsung
        view_gesture_reference(args.gesture, args.participant, args.category, args.save)


if __name__ == '__main__':
    main()
