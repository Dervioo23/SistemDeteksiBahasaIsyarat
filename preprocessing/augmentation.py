import numpy as np
from typing import List, Tuple
import random


def augment_rotation(
    landmarks: np.ndarray,
    angle_range: Tuple[float, float] = (-10, 10)
) -> np.ndarray:
    """
    Putar landmarks di sekitar sumbu Z (tegak lurus terhadap layar)
    
    Args:
        landmarks: array (frames, 126)
        angle_range: Rentang sudut rotasi dalam derajat
        
    Returns:
        Landmarks yang diputar
    """
    augmented = landmarks.copy()
    angle = np.random.uniform(angle_range[0], angle_range[1])
    angle_rad = np.deg2rad(angle)
    
    # Matriks rotasi untuk sumbu Z
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    frames = landmarks.shape[0]
    
    for frame_idx in range(frames):
        # Putar tangan kanan (fitur 0-62)
        if not np.allclose(landmarks[frame_idx, 0:63], 0):
            for i in range(0, 63, 3):
                x = landmarks[frame_idx, i]
                y = landmarks[frame_idx, i+1]
                
                # Terapkan rotasi
                augmented[frame_idx, i] = x * cos_a - y * sin_a
                augmented[frame_idx, i+1] = x * sin_a + y * cos_a
                # Koordinat Z tidak berubah
        
        # Putar tangan kiri (fitur 63-125)
        if not np.allclose(landmarks[frame_idx, 63:126], 0):
            for i in range(63, 126, 3):
                x = landmarks[frame_idx, i]
                y = landmarks[frame_idx, i+1]
                
                # Terapkan rotasi
                augmented[frame_idx, i] = x * cos_a - y * sin_a
                augmented[frame_idx, i+1] = x * sin_a + y * cos_a
    
    return augmented


def augment_scale(
    landmarks: np.ndarray,
    scale_range: Tuple[float, float] = (0.9, 1.1)
) -> np.ndarray:
    """
    Skalakan landmarks secara seragam
    Mensimulasikan ukuran tangan atau jarak kamera yang berbeda
    
    Args:
        landmarks: array (frames, 126)
        scale_range: Rentang faktor skala
        
    Returns:
        Landmarks yang diskalakan
    """
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    augmented = landmarks.copy()
    
    # Terapkan skala ke kedua tangan
    augmented = augmented * scale_factor
    
    return augmented


def augment_translation(
    landmarks: np.ndarray,
    translation_range: Tuple[float, float] = (-0.1, 0.1)
) -> np.ndarray:
    """
    Terjemahkan landmarks (geser posisi)
    Mensimulasikan posisi tangan yang berbeda dalam frame
    
    Args:
        landmarks: array (frames, 126)
        translation_range: Rentang nilai translasi
        
    Returns:
        Landmarks yang diterjemahkan
    """
    augmented = landmarks.copy()
    
    # Translasi acak untuk x, y, z
    tx = np.random.uniform(translation_range[0], translation_range[1])
    ty = np.random.uniform(translation_range[0], translation_range[1])
    tz = np.random.uniform(translation_range[0], translation_range[1])
    
    frames = landmarks.shape[0]
    
    for frame_idx in range(frames):
        # Terjemahkan tangan kanan
        if not np.allclose(landmarks[frame_idx, 0:63], 0):
            for i in range(0, 63, 3):
                augmented[frame_idx, i] += tx
                augmented[frame_idx, i+1] += ty
                augmented[frame_idx, i+2] += tz
        
        # Terjemahkan tangan kiri
        if not np.allclose(landmarks[frame_idx, 63:126], 0):
            for i in range(63, 126, 3):
                augmented[frame_idx, i] += tx
                augmented[frame_idx, i+1] += ty
                augmented[frame_idx, i+2] += tz
    
    return augmented


def augment_noise(
    landmarks: np.ndarray,
    noise_level: float = 0.01
) -> np.ndarray:
    """
    Tambahkan noise Gaussian ke landmarks
    Mensimulasikan noise sensor dan variasi kecil
    
    Args:
        landmarks: array (frames, 126)
        noise_level: Standar deviasi noise
        
    Returns:
        Landmarks dengan noise
    """
    noise = np.random.normal(0, noise_level, landmarks.shape)
    augmented = landmarks + noise
    
    return augmented.astype(np.float32)


def augment_temporal_stretch(
    landmarks: np.ndarray,
    stretch_range: Tuple[float, float] = (0.9, 1.1)
) -> np.ndarray:
    """
    Regangkan atau kompres urutan temporal
    Mensimulasikan kecepatan gestur yang berbeda
    
    Hanya berlaku untuk urutan dengan beberapa frame
    
    Args:
        landmarks: array (frames, 126)
        stretch_range: Rentang faktor regangan
        
    Returns:
        Landmarks yang diregangkan waktu
    """
    frames, features = landmarks.shape
    
    if frames <= 1:
        # Tidak dapat meregangkan satu frame
        return landmarks.copy()
    
    stretch_factor = np.random.uniform(stretch_range[0], stretch_range[1])
    new_length = int(frames * stretch_factor)
    
    # Pastikan setidaknya 1 frame
    new_length = max(1, new_length)
    
    # Interpolasi ke panjang baru
    old_indices = np.linspace(0, frames - 1, frames)
    new_indices = np.linspace(0, frames - 1, new_length)
    
    augmented = np.zeros((new_length, features), dtype=np.float32)
    
    for feat_idx in range(features):
        augmented[:, feat_idx] = np.interp(
            new_indices,
            old_indices,
            landmarks[:, feat_idx]
        )
    
    return augmented


def augment_flip_horizontal(landmarks: np.ndarray) -> np.ndarray:
    """
    Balik landmarks secara horizontal (cermin)
    Juga menukar tangan kiri dan kanan
    
    Args:
        landmarks: array (frames, 126)
        
    Returns:
        Landmarks yang dibalik secara horizontal
    """
    augmented = landmarks.copy()
    frames = landmarks.shape[0]
    
    for frame_idx in range(frames):
        # Dapatkan data tangan kanan dan kiri
        right_hand = landmarks[frame_idx, 0:63].copy()
        left_hand = landmarks[frame_idx, 63:126].copy()
        
        # Balik koordinat X (negasikan nilai x)
        for i in range(0, 63, 3):
            right_hand[i] = -right_hand[i]
            left_hand[i] = -left_hand[i]
        
        # Tukar tangan
        augmented[frame_idx, 0:63] = left_hand
        augmented[frame_idx, 63:126] = right_hand
    
    return augmented


def augment_hand_dropout(
    landmarks: np.ndarray,
    dropout_prob: float = 0.5
) -> np.ndarray:
    """Secara acak menolkan satu tangan (kanan atau kiri) di semua frame.

    Ini mendorong model untuk belajar dari setiap tangan secara mandiri dengan
    terkadang menyembunyikan salah satu tangan dalam urutan dua tangan.

    Args:
        landmarks: array (frames, features) di mana fitur digabungkan
                   sebagai [fitur_tangan_kanan, fitur_tangan_kiri]. Ini bekerja
                   untuk representasi 2D (misal, 84 fitur) dan 3D (misal, 126 fitur)
                   selama kedua tangan memiliki panjang yang sama.
        dropout_prob: Probabilitas menerapkan dropout ke satu tangan yang tersedia.

    Returns:
        Landmarks dengan satu tangan dinolkan (atau tidak berubah jika tidak diterapkan).
    """
    if landmarks is None or landmarks.size == 0:
        return landmarks

    augmented = landmarks.copy()

    # Putuskan apakah akan menerapkan dropout untuk sampel ini
    if random.random() >= dropout_prob:
        return augmented

    if augmented.ndim != 2:
        return augmented

    frames, features = augmented.shape
    if features % 2 != 0:
        # Harapkan blok berukuran sama untuk tangan kanan dan kiri
        return augmented

    half = features // 2

    right_block = augmented[:, :half]
    left_block = augmented[:, half:]

    right_present = not np.allclose(right_block, 0)
    left_present = not np.allclose(left_block, 0)

    # Jika tidak ada tangan yang hadir, tidak ada yang perlu dihapus
    if not right_present and not left_present:
        return augmented

    # Pilih tangan mana yang tersedia untuk dihapus
    candidates = []
    if right_present:
        candidates.append('right')
    if left_present:
        candidates.append('left')

    if not candidates:
        return augmented

    drop_hand = random.choice(candidates)

    if drop_hand == 'right':
        augmented[:, :half] = 0.0
    else:
        augmented[:, half:] = 0.0

    return augmented


def augment_sample(
    landmarks: np.ndarray,
    augmentation_config: dict = None
) -> np.ndarray:
    """
    Terapkan kombinasi acak dari augmentasi
    
    Args:
        landmarks: array (frames, 126)
        augmentation_config: Dictionary pengaturan augmentasi
        
    Returns:
        Landmarks yang diaugmentasi
    """
    if augmentation_config is None:
        augmentation_config = {
            'rotation': True,
            'scale': True,
            'translation': True,
            'noise': True,
            'temporal_stretch': True,
            'flip': False,  # Flip mungkin tidak sesuai untuk semua gestur
            'hand_dropout': True,
            'hand_dropout_prob': 0.5,
        }
    
    augmented = landmarks.copy()
    
    # Terapkan augmentasi dengan probabilitas tertentu
    if augmentation_config.get('rotation') and random.random() < 0.5:
        augmented = augment_rotation(augmented)
    
    if augmentation_config.get('scale') and random.random() < 0.5:
        augmented = augment_scale(augmented)
    
    if augmentation_config.get('translation') and random.random() < 0.3:
        augmented = augment_translation(augmented)
    
    if augmentation_config.get('noise') and random.random() < 0.5:
        augmented = augment_noise(augmented)
    
    if augmentation_config.get('temporal_stretch') and random.random() < 0.4:
        if augmented.shape[0] > 1:  # Hanya untuk urutan
            augmented = augment_temporal_stretch(augmented)
    
    if augmentation_config.get('flip') and random.random() < 0.3:
        augmented = augment_flip_horizontal(augmented)
    
    # Dropout tangan (ketahanan dua tangan)
    if augmentation_config.get('hand_dropout'):
        dropout_prob = float(augmentation_config.get('hand_dropout_prob', 0.5))
        augmented = augment_hand_dropout(augmented, dropout_prob=dropout_prob)
    
    return augmented


def augment_dataset(
    samples: List[np.ndarray],
    labels: List[int],
    augmentation_factor: int = 2,
    augmentation_config: dict = None
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Augmentasi seluruh dataset
    
    Args:
        samples: Daftar array landmark
        labels: Daftar label kelas
        augmentation_factor: Berapa banyak versi augmentasi per asli
        augmentation_config: Pengaturan augmentasi
        
    Returns:
        Tuple dari (augmented_samples, augmented_labels)
    """
    augmented_samples = []
    augmented_labels = []
    
    # Simpan sampel asli
    augmented_samples.extend(samples)
    augmented_labels.extend(labels)
    
    # Hasilkan versi yang diaugmentasi
    for _ in range(augmentation_factor):
        for sample, label in zip(samples, labels):
            aug_sample = augment_sample(sample, augmentation_config)
            augmented_samples.append(aug_sample)
            augmented_labels.append(label)
    
    print(f"Dataset augmented: {len(samples)} → {len(augmented_samples)} samples")
    print(f"Augmentation factor: {augmentation_factor}x + original")
    
    return augmented_samples, augmented_labels


# Contoh penggunaan
if __name__ == '__main__':
    print("🧪 Testing Augmentation Functions...\n")
    
    # Buat data dummy (10 frames, 126 features)
    dummy_landmarks = np.random.rand(10, 126) * 0.5 + 0.25
    
    print(f"Original shape: {dummy_landmarks.shape}")
    print(f"Original range: [{dummy_landmarks.min():.3f}, {dummy_landmarks.max():.3f}]")
    
    # Tes rotasi
    print("\n🔄 Testing rotation...")
    rotated = augment_rotation(dummy_landmarks, angle_range=(-15, 15))
    print(f"   Shape: {rotated.shape}")
    print(f"   Difference from original: {np.mean(np.abs(rotated - dummy_landmarks)):.4f}")
    
    # Tes skala
    print("\n📏 Testing scale...")
    scaled = augment_scale(dummy_landmarks, scale_range=(0.8, 1.2))
    print(f"   Shape: {scaled.shape}")
    print(f"   Scale factor applied: ~{np.mean(scaled / (dummy_landmarks + 1e-6)):.3f}")
    
    # Tes translasi
    print("\n📍 Testing translation...")
    translated = augment_translation(dummy_landmarks, translation_range=(-0.05, 0.05))
    print(f"   Shape: {translated.shape}")
    print(f"   Mean shift: {np.mean(translated - dummy_landmarks):.4f}")
    
    # Tes noise
    print("\n🎲 Testing noise...")
    noisy = augment_noise(dummy_landmarks, noise_level=0.02)
    print(f"   Shape: {noisy.shape}")
    print(f"   Noise std: {np.std(noisy - dummy_landmarks):.4f}")
    
    # Tes regangan temporal
    print("\n⏱️  Testing temporal stretch...")
    stretched = augment_temporal_stretch(dummy_landmarks, stretch_range=(0.8, 1.2))
    print(f"   Original frames: {dummy_landmarks.shape[0]}")
    print(f"   Stretched frames: {stretched.shape[0]}")
    
    # Tes flip
    print("\n🔁 Testing horizontal flip...")
    flipped = augment_flip_horizontal(dummy_landmarks)
    print(f"   Shape: {flipped.shape}")
    print(f"   Hands swapped: ✅")
    
    # Tes augmentasi penuh
    print("\n✨ Testing full augmentation pipeline...")
    augmented = augment_sample(dummy_landmarks)
    print(f"   Shape: {augmented.shape}")
    
    # Tes augmentasi dataset
    print("\n📦 Testing dataset augmentation...")
    samples = [dummy_landmarks] * 5
    labels = [0] * 5
    aug_samples, aug_labels = augment_dataset(
        samples, labels,
        augmentation_factor=2
    )
    print(f"   Original samples: {len(samples)}")
    print(f"   Augmented samples: {len(aug_samples)}")
    
    print("\n✅ All augmentation tests completed!")
