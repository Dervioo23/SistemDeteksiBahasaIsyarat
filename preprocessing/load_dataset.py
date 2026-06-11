import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path


def load_json_file(file_path: str) -> Optional[Dict]:
    """
    Muat satu file JSON
    
    Args:
        file_path: Path ke file JSON
        
    Returns:
        Dictionary berisi landmarks dan metadata, atau None jika error
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def parse_landmarks_to_array(landmarks: List) -> np.ndarray:
    """
    Parse landmarks ke numpy array
    
    Mendukung dua format:
    1. Format daftar datar (dari pengumpulan data):
       [[val1, val2, ..., val126], [frame2...]]
    
    2. Format kamus (lama):
       [{'right': [...], 'left': [...]}, ...]
    
    Output: numpy array shape (frames, 126)
    - 126 = 21 landmarks × 3 coords × 2 tangan
    
    Args:
        landmarks: Daftar landmarks (format daftar datar atau kamus)
        
    Returns:
        numpy array shape (frames, 126)
    """
    # Periksa apakah frame pertama sudah berupa daftar datar (format baru)
    if landmarks and isinstance(landmarks[0], list):
        # Data sudah dalam format datar dari pengumpulan
        return np.array(landmarks, dtype=np.float32)
    
    # Jika tidak, parse dari format kamus (lama)
    frames = []
    
    for frame_data in landmarks:
        frame_features = []
        
        # Tangan kanan (21 landmarks × 3 coords = 63 fitur)
        if isinstance(frame_data, dict) and 'right' in frame_data:
            for landmark in frame_data['right']:
                frame_features.extend([
                    landmark['x'],
                    landmark['y'],
                    landmark['z']
                ])
        else:
            # Jika tidak ada tangan kanan, isi dengan nol
            frame_features.extend([0.0] * 63)
        
        # Tangan kiri (21 landmarks × 3 coords = 63 fitur)
        if isinstance(frame_data, dict) and 'left' in frame_data:
            for landmark in frame_data['left']:
                frame_features.extend([
                    landmark['x'],
                    landmark['y'],
                    landmark['z']
                ])
        else:
            # Jika tidak ada tangan kiri, isi dengan nol
            frame_features.extend([0.0] * 63)
        
        frames.append(frame_features)
    
    return np.array(frames, dtype=np.float32)


def load_gesture_data(
    data_dir: str,
    label: str,
    category: str = 'words'
) -> Tuple[List[np.ndarray], List[str], List[Dict]]:
    """
    Load semua samples untuk satu gesture
    
    Args:
        data_dir: Root directory dataset (misal, 'dataset')
        label: Label gesture (misal, 'halo', 'C')
        category: 'words' atau 'alphabet'
        
    Returns:
        Tuple dari:
        - Daftar array landmark (setiap shape: (frames, 126))
        - Daftar ID file
        - Daftar dictionary metadata
    """
    # Konstruksi path ke folder gesture
    gesture_dir = os.path.join(data_dir, category, label)
    
    if not os.path.exists(gesture_dir):
        print(f"Warning: Directory tidak ditemukan: {gesture_dir}")
        return [], [], []
    
    # Dapatkan semua file JSON
    json_files = sorted([f for f in os.listdir(gesture_dir) if f.endswith('.json')])
    
    if not json_files:
        print(f"Warning: Tidak ada file JSON di {gesture_dir}")
        return [], [], []
    
    samples = []
    file_ids = []
    metadatas = []
    
    for json_file in json_files:
        file_path = os.path.join(gesture_dir, json_file)
        data = load_json_file(file_path)
        
        if data is None:
            continue
        
        # Parse landmarks ke array
        landmarks_array = parse_landmarks_to_array(data['landmarks'])
        
        # Ekstrak ID file (tanpa ekstensi)
        file_id = os.path.splitext(json_file)[0]
        
        # Dapatkan metadata
        metadata = data.get('metadata', {})
        metadata['file_id'] = file_id
        metadata['label'] = label
        metadata['category'] = category
        
        samples.append(landmarks_array)
        file_ids.append(file_id)
        metadatas.append(metadata)
    
    print(f"Loaded {len(samples)} samples for '{label}' from {gesture_dir}")
    
    return samples, file_ids, metadatas


def load_multiple_gestures(
    data_dir: str,
    labels: List[str],
    category: str = 'words'
) -> Tuple[List[np.ndarray], List[int], List[str], List[Dict]]:
    """
    Load multiple gestures dan assign class labels
    
    Args:
        data_dir: Root directory dataset
        labels: Daftar label gesture
        category: 'words' atau 'alphabet'
        
    Returns:
        Tuple dari:
        - Daftar array landmark
        - Daftar indeks kelas (0, 1, 2, ...)
        - Daftar ID file
        - Daftar metadata
    """
    all_samples = []
    all_labels = []
    all_file_ids = []
    all_metadatas = []
    
    for class_idx, label in enumerate(labels):
        samples, file_ids, metadatas = load_gesture_data(
            data_dir=data_dir,
            label=label,
            category=category
        )
        
        # Tetapkan label kelas
        class_labels = [class_idx] * len(samples)
        
        all_samples.extend(samples)
        all_labels.extend(class_labels)
        all_file_ids.extend(file_ids)
        all_metadatas.extend(metadatas)
    
    print(f"\nTotal loaded: {len(all_samples)} samples across {len(labels)} classes")
    
    return all_samples, all_labels, all_file_ids, all_metadatas


def get_dataset_info(data_dir: str) -> Dict:
    """
    Dapatkan informasi tentang dataset yang tersedia
    
    Args:
        data_dir: Root directory dataset
        
    Returns:
        Dictionary berisi informasi dataset
    """
    info = {
        'words': {},
        'alphabet': {},
        'total_samples': 0
    }
    
    # Periksa kata
    words_dir = os.path.join(data_dir, 'words')
    if os.path.exists(words_dir):
        for word in os.listdir(words_dir):
            word_path = os.path.join(words_dir, word)
            if os.path.isdir(word_path):
                json_files = [f for f in os.listdir(word_path) if f.endswith('.json')]
                info['words'][word] = len(json_files)
                info['total_samples'] += len(json_files)
    
    # Periksa alfabet
    alphabet_dir = os.path.join(data_dir, 'alphabet')
    if os.path.exists(alphabet_dir):
        for letter in os.listdir(alphabet_dir):
            letter_path = os.path.join(alphabet_dir, letter)
            if os.path.isdir(letter_path):
                json_files = [f for f in os.listdir(letter_path) if f.endswith('.json')]
                info['alphabet'][letter] = len(json_files)
                info['total_samples'] += len(json_files)
    
    return info


def print_dataset_summary(data_dir: str):
    """
    Cetak ringkasan dataset
    
    Args:
        data_dir: Root directory dataset
    """
    info = get_dataset_info(data_dir)
    
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    
    print(f"\n📝 Words ({len(info['words'])} gestures):")
    for word, count in sorted(info['words'].items()):
        print(f"   - {word}: {count} samples")
    
    print(f"\n🔤 Alphabet ({len(info['alphabet'])} letters):")
    for letter, count in sorted(info['alphabet'].items()):
        print(f"   - {letter}: {count} samples")
    
    print(f"\n📊 Total: {info['total_samples']} samples")
    print("="*60 + "\n")


# Contoh penggunaan
if __name__ == '__main__':
    # Tes pemuatan
    data_dir = 'dataset'
    
    # Cetak ringkasan dataset
    print_dataset_summary(data_dir)
    
    # Muat gesture spesifik
    print("\n🧪 Testing load_gesture_data()...")
    samples, file_ids, metadatas = load_gesture_data(
        data_dir=data_dir,
        label='halo',
        category='words'
    )
    
    if samples:
        print(f"✅ Loaded {len(samples)} samples")
        print(f"   First sample shape: {samples[0].shape}")
        print(f"   First file ID: {file_ids[0]}")
        print(f"   First metadata: {metadatas[0]}")
    
    # Muat beberapa gesture
    print("\n🧪 Testing load_multiple_gestures()...")
    X, y, ids, meta = load_multiple_gestures(
        data_dir=data_dir,
        labels=['halo', 'C'],
        category='words'  # Coba muat keduanya dari kata dulu
    )
    
    print(f"✅ Loaded {len(X)} total samples")
    print(f"   Labels: {set(y)}")
    print(f"   Sample shapes: {[x.shape for x in X[:3]]}")
