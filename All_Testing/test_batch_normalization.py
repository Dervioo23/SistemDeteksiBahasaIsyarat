"""
Uji Batch Normalization - Cari tahu mengapa semua data menjadi nol
"""
import numpy as np
import json
import os
import pytest

if __name__ != "__main__":
    pytest.skip("Debug script that expects a real dataset; run directly with python, not via pytest.", allow_module_level=True)

from preprocessing.load_dataset import load_multiple_gestures
from preprocessing.normalize import normalize_batch

print("\n" + "="*70)
print("🧪 TESTING BATCH NORMALIZATION")
print("="*70)

# Muat data nyata
print("\n📁 Loading word samples...")
samples, labels, file_ids, metadatas = load_multiple_gestures(
    data_dir='dataset',
    labels=['halo', 'terimakasih'],
    category='words'
)

print(f"✅ Loaded {len(samples)} samples")
print(f"✅ First sample shape: {samples[0].shape}")

# Periksa SEBELUM normalisasi
print(f"\n📊 BEFORE NORMALIZATION:")
first_sample = samples[0]
print(f"  Sample 0 shape: {first_sample.shape}")
print(f"  Sample 0 min: {first_sample.min():.6f}")
print(f"  Sample 0 max: {first_sample.max():.6f}")
print(f"  Sample 0 mean: {first_sample.mean():.6f}")
print(f"  Sample 0 std: {first_sample.std():.6f}")
print(f"  Sample 0 first frame (first 6): {first_sample[0, :6]}")

# Uji normalize_batch
print(f"\n📊 APPLYING NORMALIZE_BATCH (method='full')...")
normalized_samples = normalize_batch(samples, method='full')

print(f"\n📊 AFTER NORMALIZATION:")
normalized_first = normalized_samples[0]
print(f"  Sample 0 shape: {normalized_first.shape}")
print(f"  Sample 0 min: {normalized_first.min():.6f}")
print(f"  Sample 0 max: {normalized_first.max():.6f}")
print(f"  Sample 0 mean: {normalized_first.mean():.6f}")
print(f"  Sample 0 std: {normalized_first.std():.6f}")
print(f"  Sample 0 first frame (first 6): {normalized_first[0, :6]}")

# Periksa apakah semua nol
print(f"\n⚠️  CHECKING FOR ALL-ZEROS:")
for i, sample in enumerate(normalized_samples[:5]):
    is_zero = np.allclose(sample, 0)
    print(f"  Sample {i}: All zeros = {is_zero}, Sum = {sample.sum():.6f}")

# Uji dengan metode normalisasi yang berbeda
print(f"\n📊 TESTING DIFFERENT METHODS:")

# Metode 1: Hanya pergelangan tangan
normalized_wrist = normalize_batch(samples, method='wrist')
print(f"\n  WRIST-ONLY:")
print(f"    Min: {normalized_wrist[0].min():.6f}")
print(f"    Max: {normalized_wrist[0].max():.6f}")
print(f"    Std: {normalized_wrist[0].std():.6f}")
print(f"    All zeros: {np.allclose(normalized_wrist[0], 0)}")

# Metode 2: Hanya skala
normalized_scale = normalize_batch(samples, method='scale')
print(f"\n  SCALE-ONLY:")
print(f"    Min: {normalized_scale[0].min():.6f}")
print(f"    Max: {normalized_scale[0].max():.6f}")
print(f"    Std: {normalized_scale[0].std():.6f}")
print(f"    All zeros: {np.allclose(normalized_scale[0], 0)}")

print(f"\n" + "="*70)
print("✅ TEST COMPLETE")
print("="*70)
