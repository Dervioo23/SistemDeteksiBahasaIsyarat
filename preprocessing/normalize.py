import numpy as np
from typing import List, Tuple, Optional


def normalize_landmarks_wrist_relative(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalisasi landmarks relatif terhadap posisi pergelangan tangan
    Membuat gestur invarian terhadap translasi
    
    Bentuk input: (frames, 126)
    - Tangan kanan: fitur 0-62 (21 landmarks × 3)
    - Tangan kiri: fitur 63-125 (21 landmarks × 3)
    
    Pergelangan tangan adalah landmark 0 untuk setiap tangan
    
    Args:
        landmarks: array numpy (frames, 126)
        
    Returns:
        Landmarks yang dinormalisasi (frames, 126)
    """
    normalized = landmarks.copy()
    frames = landmarks.shape[0]
    
    for frame_idx in range(frames):
        # Normalisasi tangan kanan (fitur 0-62)
        right_wrist_x = landmarks[frame_idx, 0]   # x dari landmark 0
        right_wrist_y = landmarks[frame_idx, 1]   # y dari landmark 0
        right_wrist_z = landmarks[frame_idx, 2]   # z dari landmark 0
        
        # Periksa apakah tangan kanan terdeteksi (tidak semua nol)
        if not np.allclose(landmarks[frame_idx, 0:63], 0):
            for i in range(0, 63, 3):  # Setiap 3 fitur (x, y, z)
                normalized[frame_idx, i] -= right_wrist_x
                normalized[frame_idx, i+1] -= right_wrist_y
                normalized[frame_idx, i+2] -= right_wrist_z
        
        # Normalisasi tangan kiri (fitur 63-125)
        left_wrist_x = landmarks[frame_idx, 63]   # x dari landmark 0 (kiri)
        left_wrist_y = landmarks[frame_idx, 64]   # y dari landmark 0 (kiri)
        left_wrist_z = landmarks[frame_idx, 65]   # z dari landmark 0 (kiri)
        
        # Periksa apakah tangan kiri terdeteksi
        if not np.allclose(landmarks[frame_idx, 63:126], 0):
            for i in range(63, 126, 3):  # Setiap 3 fitur (x, y, z)
                normalized[frame_idx, i] -= left_wrist_x
                normalized[frame_idx, i+1] -= left_wrist_y
                normalized[frame_idx, i+2] -= left_wrist_z
    
    return normalized


def normalize_landmarks_scale(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalisasi landmarks dengan menskalakan ke rentang unit
    Membuat gestur invarian terhadap skala
    
    Skala berdasarkan jarak maksimum dari pergelangan tangan (per tangan)
    
    Args:
        landmarks: array numpy (frames, 126)
        
    Returns:
        Landmarks yang diskalakan (frames, 126)
    """
    normalized = landmarks.copy()
    frames = landmarks.shape[0]
    
    for frame_idx in range(frames):
        # Skala tangan kanan (fitur 0-62)
        if not np.allclose(landmarks[frame_idx, 0:63], 0):
            # Dapatkan semua landmarks tangan kanan
            right_landmarks = normalized[frame_idx, 0:63].reshape(-1, 3)
            wrist = right_landmarks[0]
            
            # Hitung jarak dari pergelangan tangan
            distances = np.linalg.norm(right_landmarks - wrist, axis=1)
            max_distance = np.max(distances)
            
            # Skala jika max_distance bukan nol
            if max_distance > 1e-6:
                normalized[frame_idx, 0:63] /= max_distance
        
        # Skala tangan kiri (fitur 63-125)
        if not np.allclose(landmarks[frame_idx, 63:126], 0):
            # Dapatkan semua landmarks tangan kiri
            left_landmarks = normalized[frame_idx, 63:126].reshape(-1, 3)
            wrist = left_landmarks[0]
            
            # Hitung jarak dari pergelangan tangan
            distances = np.linalg.norm(left_landmarks - wrist, axis=1)
            max_distance = np.max(distances)
            
            # Skala jika max_distance bukan nol
            if max_distance > 1e-6:
                normalized[frame_idx, 63:126] /= max_distance
    
    return normalized


def normalize_landmarks_full(landmarks: np.ndarray) -> np.ndarray:
    """
    Pipeline normalisasi penuh:
    1. Relatif pergelangan tangan (invariansi translasi)
    2. Normalisasi skala (invariansi skala)
    
    Args:
        landmarks: array numpy (frames, 126)
        
    Returns:
        Landmarks yang dinormalisasi penuh (frames, 126)
    """
    # Langkah 1: Relatif pergelangan tangan
    normalized = normalize_landmarks_wrist_relative(landmarks)
    
    # Langkah 2: Normalisasi skala
    normalized = normalize_landmarks_scale(normalized)
    
    return normalized


def normalize_batch(
    samples: List[np.ndarray],
    method: str = 'full'
) -> List[np.ndarray]:
    """
    Normalisasi batch sampel
    
    Args:
        samples: Daftar array numpy
        method: 'wrist', 'scale', atau 'full'
        
    Returns:
        Daftar sampel yang dinormalisasi
    """
    if method == 'wrist':
        normalize_fn = normalize_landmarks_wrist_relative
    elif method == 'scale':
        normalize_fn = normalize_landmarks_scale
    elif method == 'full':
        normalize_fn = normalize_landmarks_full
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    normalized_samples = []
    for sample in samples:
        normalized = normalize_fn(sample)
        normalized_samples.append(normalized)
    
    return normalized_samples



def pad_sequences_to_max_length(
    sequences: List[np.ndarray],
    max_length: Optional[int] = None,
    padding: str = 'post',
    value: float = -10.0
) -> np.ndarray:
    """
    Pad urutan ke panjang yang sama
    Berguna untuk pemrosesan batch urutan dengan panjang variabel
    
    Args:
        sequences: Daftar array dengan bentuk (frames, features)
        max_length: Panjang maksimum. Jika None, gunakan urutan terpanjang
        padding: 'pre' atau 'post'
        value: Nilai padding
        
    Returns:
        Array yang dipad (n_samples, max_length, features)
    """
    if not sequences:
        raise ValueError("Empty sequences list")
    
    # Tentukan max_length
    if max_length is None:
        max_length = max(seq.shape[0] for seq in sequences)
    
    n_features = sequences[0].shape[1]
    n_samples = len(sequences)
    
    # Inisialisasi array yang dipad
    padded = np.full(
        (n_samples, max_length, n_features),
        value,
        dtype=np.float32
    )
    
    # Isi dengan urutan
    for i, seq in enumerate(sequences):
        seq_len = min(seq.shape[0], max_length)
        
        if padding == 'post':
            padded[i, :seq_len, :] = seq[:seq_len]
        elif padding == 'pre':
            padded[i, -seq_len:, :] = seq[:seq_len]
        else:
            raise ValueError(f"Unknown padding mode: {padding}")
    
    return padded


# Contoh penggunaan
if __name__ == '__main__':
    print("🧪 Testing Normalization Functions...\n")
    
    # Buat data dummy
    # Mensimulasikan 10 frames, 126 fitur
    dummy_landmarks = np.random.rand(10, 126) * 0.5 + 0.25
    
    print(f"Original landmarks shape: {dummy_landmarks.shape}")
    print(f"Original landmarks range: [{dummy_landmarks.min():.3f}, {dummy_landmarks.max():.3f}]")
    
    # Tes normalisasi relatif pergelangan tangan
    print("\n📍 Wrist-relative normalization...")
    normalized_wrist = normalize_landmarks_wrist_relative(dummy_landmarks)
    print(f"   Shape: {normalized_wrist.shape}")
    print(f"   Range: [{normalized_wrist.min():.3f}, {normalized_wrist.max():.3f}]")
    
    # Tes normalisasi skala
    print("\n📏 Scale normalization...")
    normalized_scale = normalize_landmarks_scale(dummy_landmarks)
    print(f"   Shape: {normalized_scale.shape}")
    print(f"   Range: [{normalized_scale.min():.3f}, {normalized_scale.max():.3f}]")
    
    # Tes normalisasi penuh
    print("\n✅ Full normalization...")
    normalized_full = normalize_landmarks_full(dummy_landmarks)
    print(f"   Shape: {normalized_full.shape}")
    print(f"   Range: [{normalized_full.min():.3f}, {normalized_full.max():.3f}]")
    
    # Tes padding
    print("\n📦 Testing sequence padding...")
    sequences = [
        np.random.rand(5, 126),
        np.random.rand(10, 126),
        np.random.rand(7, 126)
    ]
    padded = pad_sequences_to_max_length(sequences, max_length=15)
    print(f"   Original lengths: {[seq.shape[0] for seq in sequences]}")
    print(f"   Padded shape: {padded.shape}")
    
    print("\n✅ All tests completed!")
