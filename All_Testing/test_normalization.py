"""
Uji Normalisasi untuk Menemukan Bug
"""
import numpy as np
import json
import os
import pytest

if __name__ != "__main__":
    pytest.skip("Debug script that expects a real dataset; run directly with python, not via pytest.", allow_module_level=True)

print("\n" + "="*70)
print("🧪 TESTING NORMALIZATION FUNCTION")
print("="*70)

# Impor fungsi normalisasi
from preprocessing.normalize import (
    normalize_landmarks_wrist_relative,
    normalize_landmarks_scale,
    normalize_landmarks_full
)

# Muat sampel nyata
sample_file = "dataset/words/halo/001_halo_001.json"
print(f"\n📁 Loading sample: {sample_file}")

with open(sample_file, 'r') as f:
    data = json.load(f)

# Konversi ke array numpy
landmarks = np.array(data['landmarks'], dtype=np.float32)
print(f"✅ Loaded shape: {landmarks.shape}")
print(f"✅ Data type: {landmarks.dtype}")

# Periksa data asli
print(f"\n📊 ORIGINAL DATA:")
print(f"  Min: {landmarks.min():.6f}")
print(f"  Max: {landmarks.max():.6f}")
print(f"  Mean: {landmarks.mean():.6f}")
print(f"  Std: {landmarks.std():.6f}")
print(f"  All zeros frames: {(landmarks.sum(axis=1) == 0).sum()}")
print(f"  First frame sample (first 6 values): {landmarks[0, :6]}")

# Uji Langkah 1: Normalisasi relatif terhadap pergelangan tangan
print(f"\n📊 AFTER WRIST-RELATIVE NORMALIZATION:")
normalized_step1 = normalize_landmarks_wrist_relative(landmarks)
print(f"  Min: {normalized_step1.min():.6f}")
print(f"  Max: {normalized_step1.max():.6f}")
print(f"  Mean: {normalized_step1.mean():.6f}")
print(f"  Std: {normalized_step1.std():.6f}")
print(f"  All zeros frames: {(normalized_step1.sum(axis=1) == 0).sum()}")
print(f"  First frame sample (first 6 values): {normalized_step1[0, :6]}")

# Periksa posisi pergelangan tangan setelah normalisasi relatif
print(f"\n  Wrist positions after wrist-relative:")
print(f"    Right wrist (0:3): {normalized_step1[0, 0:3]}")
print(f"    Left wrist (63:66): {normalized_step1[0, 63:66]}")

# Uji Langkah 2: Normalisasi skala
print(f"\n📊 AFTER SCALE NORMALIZATION:")
normalized_step2 = normalize_landmarks_scale(normalized_step1.copy())
print(f"  Min: {normalized_step2.min():.6f}")
print(f"  Max: {normalized_step2.max():.6f}")
print(f"  Mean: {normalized_step2.mean():.6f}")
print(f"  Std: {normalized_step2.std():.6f}")
print(f"  All zeros frames: {(normalized_step2.sum(axis=1) == 0).sum()}")
print(f"  First frame sample (first 6 values): {normalized_step2[0, :6]}")

# Uji Normalisasi Penuh
print(f"\n📊 FULL NORMALIZATION (wrist + scale):")
normalized_full = normalize_landmarks_full(landmarks)
print(f"  Min: {normalized_full.min():.6f}")
print(f"  Max: {normalized_full.max():.6f}")
print(f"  Mean: {normalized_full.mean():.6f}")
print(f"  Std: {normalized_full.std():.6f}")
print(f"  All zeros frames: {(normalized_full.sum(axis=1) == 0).sum()}")
print(f"  Sample variance: {normalized_full.var():.9f}")

# Periksa jarak setelah normalisasi relatif (untuk normalisasi skala)
print(f"\n🔍 DEBUGGING SCALE NORMALIZATION:")
right_landmarks = normalized_step1[0, 0:63].reshape(-1, 3)
wrist = right_landmarks[0]
print(f"  Right wrist after wrist-relative: {wrist}")
distances = np.linalg.norm(right_landmarks - wrist, axis=1)
print(f"  Distances from wrist: min={distances.min():.6f}, max={distances.max():.6f}")
print(f"  All distances: {distances[:5]}")  # 5 landmark pertama

# MASALAH TERIDENTIFIKASI
print(f"\n" + "="*70)
print("🚨 ISSUE IDENTIFIED!")
print("="*70)

if distances.max() < 1e-6:
    print("❌ Max distance is TOO SMALL after wrist-relative!")
    print("   This causes division by very small numbers")
    print("   Result: Data becomes numerically unstable or zeros")
elif normalized_full.std() < 0.01:
    print("❌ After full normalization, data variance is TOO LOW!")
    print(f"   Std: {normalized_full.std():.9f} (should be > 0.1)")
    print("   Model cannot distinguish between classes")

# Uji dengan data asli (tanpa normalisasi)
print(f"\n" + "="*70)
print("🧪 COMPARISON: NO NORMALIZATION")
print("="*70)
print(f"  Min: {landmarks.min():.6f}")
print(f"  Max: {landmarks.max():.6f}")
print(f"  Mean: {landmarks.mean():.6f}")
print(f"  Std: {landmarks.std():.6f}")
print(f"  Variance: {landmarks.var():.6f}")
print(f"\n💡 Original data has MUCH better variance!")

# Rekomendasi
print(f"\n" + "="*70)
print("✅ RECOMMENDATION")
print("="*70)
print("1. USE ORIGINAL DATA (no normalization)")
print("   OR")
print("2. USE ONLY WRIST-RELATIVE (no scale normalization)")
print("   OR")
print("3. USE DIFFERENT NORMALIZATION METHOD")
print("="*70)
