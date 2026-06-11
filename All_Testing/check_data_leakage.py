"""
Periksa Kebocoran Data - Verifikasi pemisahan train/test
"""
import numpy as np
import os

print("\n" + "="*70)
print("🔍 CHECKING FOR DATA LEAKAGE")
print("="*70)

# Muat data yang sudah diproses
words_dir = "preprocessed_data/words"
alphabet_dir = "preprocessed_data/alphabet"

# Periksa Kata
print("\n📊 WORDS DATASET:")
print("-" * 70)
train_X = np.load(os.path.join(words_dir, "train_X.npy"))
val_X = np.load(os.path.join(words_dir, "val_X.npy"))
test_X = np.load(os.path.join(words_dir, "test_X.npy"))

train_y = np.load(os.path.join(words_dir, "train_y.npy"))
val_y = np.load(os.path.join(words_dir, "val_y.npy"))
test_y = np.load(os.path.join(words_dir, "test_y.npy"))

print(f"Train: {train_X.shape}, Val: {val_X.shape}, Test: {test_X.shape}")
print(f"Total samples: {train_X.shape[0] + val_X.shape[0] + test_X.shape[0]}")

# Periksa duplikat persis antara train dan test
print(f"\n🔍 Checking for exact duplicates...")
duplicates_found = 0
for i, test_sample in enumerate(test_X):
    for j, train_sample in enumerate(train_X):
        if np.allclose(test_sample, train_sample, rtol=1e-5):
            duplicates_found += 1
            print(f"  ⚠️  Test sample {i} (label={test_y[i]}) matches Train sample {j} (label={train_y[j]})")
            if duplicates_found >= 5:
                print(f"  ... (showing first 5 only)")
                break
    if duplicates_found >= 5:
        break

if duplicates_found == 0:
    print(f"  ✅ No exact duplicates found between train and test")
else:
    print(f"  ❌ Found {duplicates_found}+ duplicates! DATA LEAKAGE!")

# Periksa kesamaan (sampel yang sangat mirip)
print(f"\n🔍 Checking for very similar samples...")
similar_count = 0
for i in range(min(10, len(test_X))):  # Periksa 10 sampel uji pertama
    test_sample = test_X[i]
    for j in range(len(train_X)):
        train_sample = train_X[j]
        # Hitung kesamaan kosinus
        similarity = np.dot(test_sample.flatten(), train_sample.flatten()) / (
            np.linalg.norm(test_sample.flatten()) * np.linalg.norm(train_sample.flatten()) + 1e-8
        )
        if similarity > 0.99:  # Sangat mirip
            similar_count += 1
            print(f"  ⚠️  Test[{i}] very similar to Train[{j}] (similarity={similarity:.4f})")
            break

if similar_count > 5:
    print(f"  ❌ Many similar samples found! Possible data leakage or augmentation issue")
elif similar_count > 0:
    print(f"  ⚠️  Some similar samples found ({similar_count}/10)")
else:
    print(f"  ✅ No highly similar samples in first 10 test samples")

# Periksa distribusi kelas
print(f"\n📊 Class Distribution:")
from collections import Counter
train_dist = Counter(train_y)
val_dist = Counter(val_y)
test_dist = Counter(test_y)

for label in sorted(set(train_y)):
    print(f"  Class {label}: Train={train_dist[label]}, Val={val_dist[label]}, Test={test_dist[label]}")

# Periksa apakah set pengujian terlalu mudah
print(f"\n🔍 Test Set Difficulty Analysis:")
print(f"  Test samples per class: {test_dist.most_common(1)[0][1]}")
print(f"  Training samples per class: {train_dist.most_common(1)[0][1]}")
ratio = train_dist.most_common(1)[0][1] / test_dist.most_common(1)[0][1]
print(f"  Train/Test ratio: {ratio:.1f}x")

if ratio > 10:
    print(f"  ⚠️  Very small test set - easy to get 100% by chance")
elif ratio > 5:
    print(f"  ⚠️  Small test set - 100% accuracy might not generalize")
else:
    print(f"  ✅ Reasonable train/test ratio")

# Periksa varians data
print(f"\n📊 Data Variance Analysis:")
train_var = np.var(train_X)
test_var = np.var(test_X)
print(f"  Train variance: {train_var:.6f}")
print(f"  Test variance: {test_var:.6f}")
print(f"  Ratio: {train_var/test_var:.2f}")

if abs(train_var/test_var - 1.0) > 0.5:
    print(f"  ⚠️  Train and test have different distributions!")
else:
    print(f"  ✅ Similar variance in train and test")

print("\n" + "="*70)
print("✅ DATA LEAKAGE CHECK COMPLETE")
print("="*70)
