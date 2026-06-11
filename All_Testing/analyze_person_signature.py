"""
Menganalisis "Tanda Tangan" Orang dalam Data Landmark
Menunjukkan bagaimana model dapat mengenali orang bahkan tanpa gambar
"""
import numpy as np
import json
import os
import glob

print("\n" + "="*70)
print("🔍 ANALYZING PERSON 'SIGNATURE' IN LANDMARK DATA")
print("="*70)

# Muat sampel dari satu isyarat
gesture = "halo"
data_dir = f"dataset/words/{gesture}"

if not os.path.exists(data_dir):
    print(f"❌ Directory not found: {data_dir}")
    exit(1)

json_files = glob.glob(os.path.join(data_dir, "*.json"))[:5]
print(f"\n📁 Loading {len(json_files)} samples from '{gesture}'...")

samples = []
for file_path in json_files:
    with open(file_path, 'r') as f:
        data = json.load(f)
        landmarks = np.array(data['landmarks'])
        samples.append(landmarks)
        print(f"  ✅ Loaded {os.path.basename(file_path)}: shape {landmarks.shape}")

# Analisis pola spesifik orang
print("\n" + "="*70)
print("🔬 PERSON-SPECIFIC PATTERNS (Biometric Signatures)")
print("="*70)

print("\n📊 1. HAND SIZE SIGNATURE:")
print("-" * 70)
for i, sample in enumerate(samples[:3]):
    # Hitung rentang tangan (jarak antara landmark)
    # Pergelangan tangan kanan = landmark 0 (x, y, z pada indeks 0, 1, 2)
    # Ujung jari tengah kanan = landmark 12 (x, y, z pada indeks 36, 37, 38)
    
    # Rata-rata di seluruh frame
    frame = sample[0]  # Frame pertama
    
    # Tangan kanan: landmark 0-20 (0-62 dalam array datar)
    right_wrist = frame[0:3]
    right_middle_tip = frame[36:39]
    hand_span = np.linalg.norm(right_middle_tip - right_wrist)
    
    # Lebar tangan (ibu jari ke pangkal kelingking)
    right_thumb_base = frame[3:6]
    right_pinky_base = frame[57:60]
    hand_width = np.linalg.norm(right_pinky_base - right_thumb_base)
    
    print(f"  Sample {i+1}:")
    print(f"    Hand span (wrist to middle finger): {hand_span:.4f}")
    print(f"    Hand width (thumb to pinky):        {hand_width:.4f}")
    print(f"    Aspect ratio:                       {hand_span/hand_width:.4f}")

print("\n📊 2. GESTURE SPEED SIGNATURE:")
print("-" * 70)
for i, sample in enumerate(samples[:3]):
    # Hitung kecepatan gerakan
    total_movement = 0
    for t in range(len(sample) - 1):
        movement = np.linalg.norm(sample[t+1] - sample[t])
        total_movement += movement
    
    avg_speed = total_movement / (len(sample) - 1)
    print(f"  Sample {i+1}: Average movement per frame = {avg_speed:.6f}")

print("\n📊 3. HAND POSITION PREFERENCE:")
print("-" * 70)
for i, sample in enumerate(samples[:3]):
    # Posisi rata-rata dalam frame
    avg_x = np.mean(sample[:, 0::3])  # Semua koordinat X
    avg_y = np.mean(sample[:, 1::3])  # Semua koordinat Y
    avg_z = np.mean(sample[:, 2::3])  # Semua koordinat Z
    
    print(f"  Sample {i+1}:")
    print(f"    Preferred X position: {avg_x:.4f}")
    print(f"    Preferred Y position: {avg_y:.4f}")
    print(f"    Preferred Z (depth):  {avg_z:.4f}")

print("\n📊 4. MOVEMENT PATTERN CONSISTENCY:")
print("-" * 70)
# Hitung kesamaan pola gerakan antar sampel
from scipy.spatial.distance import cosine

for i in range(min(3, len(samples))):
    for j in range(i+1, min(3, len(samples))):
        # Ratakan dan bandingkan
        vec1 = samples[i].flatten()
        vec2 = samples[j].flatten()
        
        # Pastikan panjang yang sama
        min_len = min(len(vec1), len(vec2))
        vec1 = vec1[:min_len]
        vec2 = vec2[:min_len]
        
        similarity = 1 - cosine(vec1, vec2)
        print(f"  Sample {i+1} vs Sample {j+1}: Similarity = {similarity:.4f} ({similarity*100:.2f}%)")

print("\n" + "="*70)
print("💡 KEY FINDINGS:")
print("="*70)
print("""
Meskipun data BUKAN gambar, model tetap bisa 'mengenali' person karena:

1. 🖐️  HAND SIZE & PROPORTIONS
   → Setiap orang punya ukuran tangan berbeda
   → Rasio panjang jari berbeda
   → Model belajar: "Tangan ini ukuran sekian = Dervio"

2. ⚡ GESTURE SPEED & TIMING
   → Setiap orang punya kecepatan gesture berbeda
   → Pattern temporal unique per person
   → Model belajar: "Gesture secepat ini = Dervio"

3. 📍 POSITION PREFERENCE
   → Setiap orang cenderung gesture di posisi sama
   → Kebiasaan positioning unik
   → Model belajar: "Gesture di posisi ini = Dervio"

4. 🎯 MOVEMENT STYLE
   → Cara menggerakkan tangan (smooth/jerky)
   → Trajectory pattern
   → Model belajar: "Style movement ini = Dervio"

5. 🔄 CONSISTENCY PATTERN
   → Gesture Anda sangat konsisten (similarity 99%+)
   → Orang lain akan punya consistency berbeda
   → Model belajar: "Pattern konsisten ini = Dervio"

INI DISEBUT "BIOMETRIC SIGNATURE" dalam data!
Model TIDAK butuh gambar untuk recognize person!
Landmark coordinates CUKUP untuk identify unique patterns!
""")

print("\n" + "="*70)
print("⚠️  IMPLICATION:")
print("="*70)
print("""
PROBLEM:
  Model belajar: "Gesture dengan signature ini = Class X"
  Bukan belajar: "Bentuk gesture ini = Class X"

SOLUTION:
  ✅ Collect dari multiple people (3-5 orang)
  ✅ Setiap orang punya signature berbeda
  ✅ Model belajar: "Berbagai signature, sama gesture = Class X"
  ✅ Model generalize ke gesture shape, bukan person!
""")

print("\n" + "="*70)
print("✅ ANALYSIS COMPLETE")
print("="*70)
