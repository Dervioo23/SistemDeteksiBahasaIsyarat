"""
DEEP DIAGNOSTIC - Investigasi Sistem Komprehensif
Memeriksa semua aspek dari data collection hingga training
"""
import os
import sys
import json
import numpy as np
import pickle
from pathlib import Path
from collections import Counter

print("\n" + "="*70)
print("DEEP DIAGNOSTIC - COMPREHENSIVE INVESTIGATION")
print("="*70)

# ============================================================================
# 1. PERIKSA DATASET MENTAH
# ============================================================================
print("\n" + "="*70)
print("📁 STEP 1: INSPECTING RAW DATASET")
print("="*70)

dataset_dir = "dataset"
words_dir = os.path.join(dataset_dir, "words")
alphabet_dir = os.path.join(dataset_dir, "alphabet")

# Periksa Kata
print("\n🔤 WORDS DATASET:")
print("-" * 70)
word_gestures = {}
if os.path.exists(words_dir):
    for gesture in os.listdir(words_dir):
        gesture_path = os.path.join(words_dir, gesture)
        if os.path.isdir(gesture_path):
            json_files = [f for f in os.listdir(gesture_path) if f.endswith('.json')]
            word_gestures[gesture] = len(json_files)
            print(f"  {gesture}: {len(json_files)} samples")
            
            # Sampel satu file untuk memeriksa struktur
            if json_files:
                sample_file = os.path.join(gesture_path, json_files[0])
                with open(sample_file, 'r') as f:
                    sample_data = json.load(f)
                
                print(f"    Sample file: {json_files[0]}")
                print(f"    Keys: {list(sample_data.keys())}")
                print(f"    Num frames: {len(sample_data['landmarks'])}")
                
                # Periksa struktur frame pertama
                first_frame = sample_data['landmarks'][0]
                print(f"    First frame type: {type(first_frame)}")
                
                if isinstance(first_frame, dict):
                    print(f"    First frame keys: {list(first_frame.keys())}")
                    if 'right' in first_frame:
                        print(f"    Right hand landmarks: {len(first_frame['right'])}")
                    if 'left' in first_frame:
                        print(f"    Left hand landmarks: {len(first_frame['left'])}")
                elif isinstance(first_frame, list):
                    print(f"    First frame is list with {len(first_frame)} elements")
                    print(f"    First 3 elements: {first_frame[:3]}")

print(f"\n📊 Total word gestures: {len(word_gestures)}")
print(f"📊 Total word samples: {sum(word_gestures.values())}")

# Periksa Alfabet
print("\n🔤 ALPHABET DATASET:")
print("-" * 70)
alphabet_letters = {}
if os.path.exists(alphabet_dir):
    for letter in os.listdir(alphabet_dir):
        letter_path = os.path.join(alphabet_dir, letter)
        if os.path.isdir(letter_path):
            json_files = [f for f in os.listdir(letter_path) if f.endswith('.json')]
            alphabet_letters[letter] = len(json_files)
            print(f"  {letter}: {len(json_files)} samples")

print(f"\n📊 Total alphabet letters: {len(alphabet_letters)}")
print(f"📊 Total alphabet samples: {sum(alphabet_letters.values())}")

# ============================================================================
# 2. PERIKSA DATA YANG SUDAH DIPROSES
# ============================================================================
print("\n" + "="*70)
print("📦 STEP 2: INSPECTING PREPROCESSED DATA")
print("="*70)

# Periksa Kata yang Sudah Diproses
words_prep_dir = "preprocessed_data/words"
if os.path.exists(words_prep_dir):
    print("\n🔤 PREPROCESSED WORDS:")
    print("-" * 70)
    
    # Muat data
    train_X = np.load(os.path.join(words_prep_dir, "train_X.npy"))
    train_y = np.load(os.path.join(words_prep_dir, "train_y.npy"))
    val_X = np.load(os.path.join(words_prep_dir, "val_X.npy"))
    val_y = np.load(os.path.join(words_prep_dir, "val_y.npy"))
    test_X = np.load(os.path.join(words_prep_dir, "test_X.npy"))
    test_y = np.load(os.path.join(words_prep_dir, "test_y.npy"))
    
    print(f"  Train X shape: {train_X.shape}, dtype: {train_X.dtype}")
    print(f"  Train y shape: {train_y.shape}, dtype: {train_y.dtype}")
    print(f"  Val X shape:   {val_X.shape}")
    print(f"  Test X shape:  {test_X.shape}")
    
    # Periksa distribusi label
    print(f"\n  📊 Label Distribution:")
    train_dist = Counter(train_y)
    val_dist = Counter(val_y)
    test_dist = Counter(test_y)
    
    all_labels = sorted(set(train_y) | set(val_y) | set(test_y))
    print(f"  Unique labels: {all_labels}")
    for label in all_labels:
        print(f"    Label {label}: Train={train_dist[label]}, Val={val_dist[label]}, Test={test_dist[label]}")
    
    # Muat metadata
    with open(os.path.join(words_prep_dir, "metadata.pkl"), 'rb') as f:
        metadata = pickle.load(f)
    print(f"\n  📝 Class Names: {metadata['class_names']}")
    print(f"  📝 Num Classes: {metadata['num_classes']}")
    
    # Periksa statistik data
    print(f"\n  📈 Data Statistics (Train X):")
    print(f"    Min: {train_X.min():.6f}")
    print(f"    Max: {train_X.max():.6f}")
    print(f"    Mean: {train_X.mean():.6f}")
    print(f"    Std: {train_X.std():.6f}")
    
    # Periksa NaN atau Inf
    print(f"\n  ⚠️  Data Quality Checks:")
    print(f"    Contains NaN: {np.isnan(train_X).any()}")
    print(f"    Contains Inf: {np.isinf(train_X).any()}")
    print(f"    All zeros samples: {(train_X.sum(axis=(1,2)) == 0).sum()}")
    
    # Periksa varians sampel
    print(f"\n  📊 Sample Variance:")
    sample_variances = train_X.var(axis=(1,2))
    print(f"    Min variance: {sample_variances.min():.6f}")
    print(f"    Max variance: {sample_variances.max():.6f}")
    print(f"    Mean variance: {sample_variances.mean():.6f}")
    print(f"    Zero variance samples: {(sample_variances == 0).sum()}")
    
    # Periksa keterpisahan kelas
    print(f"\n  🔍 Class Separability Analysis:")
    for label in all_labels:
        label_samples = train_X[train_y == label]
        label_mean = label_samples.mean(axis=0)
        print(f"    Label {label} ({metadata['class_names'][label]}):")
        print(f"      Samples: {len(label_samples)}")
        print(f"      Mean feature magnitude: {np.abs(label_mean).mean():.6f}")
        print(f"      Std feature magnitude: {label_samples.std():.6f}")

# Periksa Alfabet yang Sudah Diproses
alphabet_prep_dir = "preprocessed_data/alphabet"
if os.path.exists(alphabet_prep_dir):
    print("\n🔤 PREPROCESSED ALPHABET:")
    print("-" * 70)
    
    # Muat data
    train_X_alpha = np.load(os.path.join(alphabet_prep_dir, "train_X.npy"))
    train_y_alpha = np.load(os.path.join(alphabet_prep_dir, "train_y.npy"))
    val_X_alpha = np.load(os.path.join(alphabet_prep_dir, "val_X.npy"))
    val_y_alpha = np.load(os.path.join(alphabet_prep_dir, "val_y.npy"))
    test_X_alpha = np.load(os.path.join(alphabet_prep_dir, "test_X.npy"))
    test_y_alpha = np.load(os.path.join(alphabet_prep_dir, "test_y.npy"))
    
    print(f"  Train X shape: {train_X_alpha.shape}, dtype: {train_X_alpha.dtype}")
    print(f"  Train y shape: {train_y_alpha.shape}, dtype: {train_y_alpha.dtype}")
    
    # Periksa distribusi label
    print(f"\n  📊 Label Distribution:")
    train_dist_alpha = Counter(train_y_alpha)
    val_dist_alpha = Counter(val_y_alpha)
    test_dist_alpha = Counter(test_y_alpha)
    
    all_labels_alpha = sorted(set(train_y_alpha) | set(val_y_alpha) | set(test_y_alpha))
    print(f"  Unique labels: {all_labels_alpha}")
    for label in all_labels_alpha:
        print(f"    Label {label}: Train={train_dist_alpha[label]}, Val={val_dist_alpha[label]}, Test={test_dist_alpha[label]}")
    
    # Muat metadata
    with open(os.path.join(alphabet_prep_dir, "metadata.pkl"), 'rb') as f:
        metadata_alpha = pickle.load(f)
    print(f"\n  📝 Class Names: {metadata_alpha['class_names']}")
    
    # Periksa statistik data
    print(f"\n  📈 Data Statistics (Train X):")
    print(f"    Min: {train_X_alpha.min():.6f}")
    print(f"    Max: {train_X_alpha.max():.6f}")
    print(f"    Mean: {train_X_alpha.mean():.6f}")
    print(f"    Std: {train_X_alpha.std():.6f}")
    
    # Periksa NaN atau Inf
    print(f"\n  ⚠️  Data Quality Checks:")
    print(f"    Contains NaN: {np.isnan(train_X_alpha).any()}")
    print(f"    Contains Inf: {np.isinf(train_X_alpha).any()}")
    print(f"    All zeros samples: {(train_X_alpha.sum(axis=(1,2)) == 0).sum()}")

# ============================================================================
# 3. PERIKSA KONFIGURASI PELATIHAN
# ============================================================================
print("\n" + "="*70)
print("⚙️  STEP 3: CHECKING TRAINING CONFIGURATION")
print("="*70)

# Periksa skrip pelatihan
print("\n📋 Training Scripts Configuration:")
print("-" * 70)

# Baca skrip pelatihan
train_script = "training/train_multiclass_word.py"
if os.path.exists(train_script):
    with open(train_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"✅ {train_script} exists")
    
    # Periksa konfigurasi kunci
    checks = {
        "sparse_categorical_crossentropy": "Loss Function" in content,
        "Adam": "Optimizer" in content,
        "EarlyStopping": "Early Stopping Callback" in content,
        "ModelCheckpoint": "Model Checkpoint" in content,
    }
    
    for key, found in checks.items():
        status = "✅" if key in content else "❌"
        print(f"  {status} {key}: {'Found' if key in content else 'Not found'}")

# ============================================================================
# 4. ANALISIS KESAMAAN KELAS
# ============================================================================
print("\n" + "="*70)
print("🔬 STEP 4: CLASS SIMILARITY ANALYSIS")
print("="*70)

if os.path.exists(words_prep_dir):
    print("\n📊 WORD CLASSES SIMILARITY:")
    print("-" * 70)
    
    # Hitung rata-rata untuk setiap kelas
    class_means = {}
    for label in all_labels:
        label_samples = train_X[train_y == label]
        class_means[label] = label_samples.mean(axis=0)
    
    # Hitung jarak berpasangan
    from scipy.spatial.distance import cosine
    
    print(f"\n  Distance Matrix (Cosine Distance):")
    print(f"  {'':>15}", end="")
    for label in all_labels:
        print(f"{metadata['class_names'][label]:>15}", end="")
    print()
    
    for label1 in all_labels:
        print(f"  {metadata['class_names'][label1]:>15}", end="")
        for label2 in all_labels:
            if label1 == label2:
                print(f"{'0.000':>15}", end="")
            else:
                dist = cosine(class_means[label1].flatten(), class_means[label2].flatten())
                print(f"{dist:>15.3f}", end="")
        print()
    
    print(f"\n  ⚠️  Interpretation:")
    print(f"    - Distance = 0: Identik (sangat mirip)")
    print(f"    - Distance < 0.1: Sangat mirip (PROBLEM!)")
    print(f"    - Distance > 0.3: Berbeda (GOOD!)")

# ============================================================================
# 5. RINGKASAN DIAGNOSTIK AKHIR
# ============================================================================
print("\n" + "="*70)
print("📋 DIAGNOSTIC SUMMARY")
print("="*70)

issues_found = []
warnings_found = []

# Periksa 1: Ukuran dataset
if sum(word_gestures.values()) < 400:
    issues_found.append("Dataset size terlalu kecil untuk model kompleks")
    
# Periksa 2: Keseimbangan kelas
if word_gestures:
    counts = list(word_gestures.values())
    if max(counts) / min(counts) > 1.5:
        warnings_found.append("Class imbalance detected")

# Periksa 3: Varians data
if os.path.exists(words_prep_dir):
    if (sample_variances == 0).sum() > 0:
        issues_found.append(f"Found {(sample_variances == 0).sum()} samples with zero variance")
    
    if train_X.std() < 0.01:
        warnings_found.append("Data variance very low - might indicate normalization issue")

print("\n🔴 ISSUES FOUND:")
if issues_found:
    for i, issue in enumerate(issues_found, 1):
        print(f"  {i}. {issue}")
else:
    print("  ✅ No critical issues found")

print("\n⚠️  WARNINGS:")
if warnings_found:
    for i, warning in enumerate(warnings_found, 1):
        print(f"  {i}. {warning}")
else:
    print("  ✅ No warnings")

print("\n" + "="*70)
print("✅ DIAGNOSTIC COMPLETE")
print("="*70)
