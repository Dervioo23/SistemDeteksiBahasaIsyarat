import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

# Tambahkan root proyek ke path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.augmentation import augment_sample
from preprocessing.normalize import normalize_landmarks_full
# from data_collection.utils import load_config

def load_config():
    """Muat konfigurasi dari config.json"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
    with open(config_path, 'r') as f:
        return json.load(f)

def load_random_sample(dataset_dir):
    """Muat sampel JSON acak dari dataset"""
    files = []
    for root, _, filenames in os.walk(dataset_dir):
        for filename in filenames:
            if filename.endswith('.json'):
                files.append(os.path.join(root, filename))
    
    if not files:
        return None, None
    
    filepath = random.choice(files)
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data, filepath

def plot_landmarks(ax, landmarks, title, color='b'):
    """Plot landmark 3D"""
    # Bentuk landmarks: (frames, 126) atau (126,)
    if len(landmarks.shape) == 1:
        landmarks = landmarks.reshape(1, -1)
    
    # Ambil frame pertama untuk visualisasi
    frame_lm = landmarks[0].reshape(-1, 3)
    
    # Pisahkan menjadi tangan (asumsi 21 poin per tangan, maks 2 tangan)
    # 0-20: Tangan kanan, 21-41: Tangan kiri (berdasarkan logika ekstraksi)
    
    # Koneksi untuk tangan MediaPipe
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),       # Ibu jari
        (0, 5), (5, 6), (6, 7), (7, 8),       # Telunjuk
        (0, 9), (9, 10), (10, 11), (11, 12),  # Tengah
        (0, 13), (13, 14), (14, 15), (15, 16), # Manis
        (0, 17), (17, 18), (18, 19), (19, 20)  # Kelingking
    ]
    
    # Plot Tangan Kanan (0-20)
    right_hand = frame_lm[:21]
    if np.any(right_hand): # Periksa jika tidak semua nol
        ax.scatter(right_hand[:, 0], right_hand[:, 1], right_hand[:, 2], c=color, marker='o')
        for i, j in connections:
            ax.plot([right_hand[i, 0], right_hand[j, 0]],
                    [right_hand[i, 1], right_hand[j, 1]],
                    [right_hand[i, 2], right_hand[j, 2]], c=color)
    
    # Plot Tangan Kiri (21-41)
    if len(frame_lm) > 21:
        left_hand = frame_lm[21:42]
        if np.any(left_hand):
            ax.scatter(left_hand[:, 0], left_hand[:, 1], left_hand[:, 2], c='r', marker='^')
            for i, j in connections:
                ax.plot([left_hand[i, 0], left_hand[j, 0]],
                        [left_hand[i, 1], left_hand[j, 1]],
                        [left_hand[i, 2], left_hand[j, 2]], c='r')

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Tetapkan batas yang konsisten untuk melihat penskalaan/rotasi
    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([-0.5, 1.5])
    ax.set_zlim([-0.5, 0.5])
    
    # Balikkan Y dan Z agar sesuai dengan koordinat layar biasanya
    ax.invert_yaxis()

def main():
    print("="*60)
    print("PREPROCESSING VISUALIZATION TOOL")
    print("="*60)
    
    config = load_config()
    dataset_dir = config['dataset']['alphabet_dir'] # Default ke alfabet
    
    print(f"Loading sample from: {dataset_dir}")
    data, filepath = load_random_sample(dataset_dir)
    
    if data is None:
        print("No data found!")
        return
        
    print(f"Loaded: {os.path.basename(filepath)}")
    
    # Ekstrak landmarks dari dictionary jika diperlukan
    if isinstance(data, dict) and 'landmarks' in data:
        landmarks = np.array(data['landmarks'])
    else:
        landmarks = np.array(data)
        
    print(f"Original shape: {landmarks.shape}")
    
    # Pastikan 2D (frames, features)
    if landmarks.ndim == 1:
        landmarks = landmarks.reshape(1, -1)
        print(f"Reshaped to: {landmarks.shape}")
    elif landmarks.ndim == 0:
        print("Error: Loaded data is empty or scalar")
        return
    
    # Terapkan Augmentasi
    print("Applying augmentation...")
    augmented_landmarks = augment_sample(landmarks)
    
    # Terapkan Normalisasi
    print("Applying normalization...")
    normalized_original = normalize_landmarks_full(landmarks.copy())
    normalized_augmented = normalize_landmarks_full(augmented_landmarks.copy())
    
    # Visualisasi
    fig = plt.figure(figsize=(15, 5))
    
    # 1. Asli Mentah
    ax1 = fig.add_subplot(131, projection='3d')
    plot_landmarks(ax1, landmarks, "Original (Raw)", color='g')
    
    # 2. Augmentasi Mentah
    ax2 = fig.add_subplot(132, projection='3d')
    plot_landmarks(ax2, augmented_landmarks, "Augmented (Raw)", color='orange')
    
    # 3. Perbandingan Normalisasi
    ax3 = fig.add_subplot(133, projection='3d')
    # Plot keduanya untuk melihat tumpang tindih/perbedaan
    plot_landmarks(ax3, normalized_original, "Normalized (Blue=Orig, Red=Aug)", color='blue')
    # Overlay augmented dalam warna merah (putus-putus?) - sulit dilakukan putus-putus di plot 3d secara sederhana, gunakan saja warna
    # Sebenarnya plot_landmarks menangani warna. Mari kita plot augmented di atasnya.
    
    # Plot ulang augmented pada ax3 yang sama
    # Kita perlu memodifikasi plot_landmarks untuk menerima ax dan TIDAK membersihkannya. Itu menambahkan.
    # Tapi untuk kejelasan, mari kita tunjukkan saja yang Augmentasi Dinormalisasi.
    # Atau lebih baik: Tunjukkan Augmentasi & Dinormalisasi.
    plot_landmarks(ax3, normalized_augmented, "Augmented & Normalized", color='purple')

    plt.tight_layout()
    print("\nVisualization generated! Check the popup window.")
    plt.show()

if __name__ == "__main__":
    main()
