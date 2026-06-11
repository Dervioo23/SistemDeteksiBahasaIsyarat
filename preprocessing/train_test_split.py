import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from collections import Counter, defaultdict
import random
import os
import re


def stratified_split(
    samples: List[np.ndarray],
    labels: List[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[
    List[np.ndarray], List[int],
    List[np.ndarray], List[int],
    List[np.ndarray], List[int]
]:
    """
    Pemisahan bertingkat: pertahankan distribusi kelas di setiap pemisahan
    
    Args:
        samples: Daftar array landmark
        labels: Daftar label kelas
        train_ratio: Proporsi untuk set pelatihan
        val_ratio: Proporsi untuk set validasi
        test_ratio: Proporsi untuk set pengujian
        random_seed: Seed acak untuk reproduktifitas
        
    Returns:
        Tuple dari (train_samples, train_labels, val_samples, val_labels, test_samples, test_labels)
    """
    # Validasi rasio
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Atur seed acak
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Kelompokkan sampel berdasarkan kelas
    class_to_samples = {}
    for sample, label in zip(samples, labels):
        if label not in class_to_samples:
            class_to_samples[label] = []
        class_to_samples[label].append((sample, label))
    
    # Pisahkan setiap kelas secara terpisah
    train_samples, train_labels = [], []
    val_samples, val_labels = [], []
    test_samples, test_labels = [], []
    
    for class_label, class_samples in class_to_samples.items():
        # Acak sampel kelas
        random.shuffle(class_samples)
        
        n_samples = len(class_samples)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        # Pisahkan
        train_split = class_samples[:n_train]
        val_split = class_samples[n_train:n_train+n_val]
        test_split = class_samples[n_train+n_val:]
        
        # Tambahkan ke set masing-masing
        for sample, label in train_split:
            train_samples.append(sample)
            train_labels.append(label)
        
        for sample, label in val_split:
            val_samples.append(sample)
            val_labels.append(label)
        
        for sample, label in test_split:
            test_samples.append(sample)
            test_labels.append(label)
    
    # Acak setiap set
    train_indices = list(range(len(train_samples)))
    random.shuffle(train_indices)
    train_samples = [train_samples[i] for i in train_indices]
    train_labels = [train_labels[i] for i in train_indices]
    
    val_indices = list(range(len(val_samples)))
    random.shuffle(val_indices)
    val_samples = [val_samples[i] for i in val_indices]
    val_labels = [val_labels[i] for i in val_indices]
    
    test_indices = list(range(len(test_samples)))
    random.shuffle(test_indices)
    test_samples = [test_samples[i] for i in test_indices]
    test_labels = [test_labels[i] for i in test_indices]
    
    return (train_samples, train_labels, 
            val_samples, val_labels, 
            test_samples, test_labels)


def print_split_info(
    train_labels: List[int],
    val_labels: List[int],
    test_labels: List[int],
    class_names: List[str] = None
):
    """Cetak informasi tentang pemisahan dataset"""
    total = len(train_labels) + len(val_labels) + len(test_labels)
    
    print("\n" + "="*60)
    print("DATASET SPLIT SUMMARY")
    print("="*60)
    
    print(f"\nTotal samples: {total}")
    print(f"  Training:   {len(train_labels):3d} ({len(train_labels)/total*100:.1f}%)")
    print(f"  Validation: {len(val_labels):3d} ({len(val_labels)/total*100:.1f}%)")
    print(f"  Test:       {len(test_labels):3d} ({len(test_labels)/total*100:.1f}%)")
    
    train_dist = Counter(train_labels)
    val_dist = Counter(val_labels)
    test_dist = Counter(test_labels)
    
    all_classes = sorted(set(train_labels + val_labels + test_labels))
    
    print(f"\nClass Distribution:")
    for class_idx in all_classes:
        class_name = class_names[class_idx] if class_names else f"Class {class_idx}"
        print(f"  {class_name}: Train={train_dist.get(class_idx, 0)}, Val={val_dist.get(class_idx, 0)}, Test={test_dist.get(class_idx, 0)}")
    
    print("="*70 + "\n")


def get_available_participants(file_ids: List[str]) -> List[str]:
    """
    Dapatkan daftar peserta yang tersedia dari file_ids
    
    Args:
        file_ids: Daftar ID file (format: "participant_gesture_sample")
        
    Returns:
        Daftar ID peserta unik
    """
    participants = set()
    for file_id in file_ids:
        participant_id = file_id.split('_')[0]
        participants.add(participant_id)
    
    return sorted(list(participants))


def analyze_participant_distribution(
    file_ids: List[str], 
    labels: List[int],
    class_names: List[str] = None
) -> Dict:
    """
    Analisis bagaimana sampel didistribusikan di seluruh peserta
    
    Args:
        file_ids: Daftar ID file
        labels: Daftar label kelas  
        class_names: Nama kelas opsional untuk tampilan
        
    Returns:
        Dictionary dengan analisis peserta
    """
    participant_data = defaultdict(lambda: defaultdict(int))
    
    for file_id, label in zip(file_ids, labels):
        participant_id = file_id.split('_')[0]
        participant_data[participant_id][label] += 1
    
    print(f"\n📊 PARTICIPANT DISTRIBUTION ANALYSIS")
    print("="*60)
    
    total_participants = len(participant_data)
    participants = sorted(participant_data.keys())
    
    print(f"Total participants: {total_participants}")
    print(f"Available participants: {participants}")
    
    print(f"\nSamples per participant:")
    for p_id in participants:
        class_dist = participant_data[p_id]
        total_samples = sum(class_dist.values())
        print(f"  {p_id}: {total_samples} samples")
        
        for class_idx, count in class_dist.items():
            class_name = class_names[class_idx] if class_names else f"Class {class_idx}"
            print(f"    {class_name}: {count}")
    
    # Rekomendasi
    print(f"\n💡 MANUAL ASSIGNMENT RECOMMENDATIONS:")
    
    if total_participants >= 5:
        # Rekomendasikan pemisahan 3:1:1
        n_train = max(1, int(total_participants * 0.6))
        n_val = 1  
        n_test = 1
        
        suggested_train = participants[:n_train]
        suggested_val = participants[n_train:n_train+n_val]
        suggested_test = participants[n_train+n_val:n_train+n_val+n_test]
        
        print(f"  For {total_participants} participants, suggested split:")
        print(f"    Training: {suggested_train}")
        print(f"    Validation: {suggested_val}")
        print(f"    Test: {suggested_test}")
        
    elif total_participants >= 3:
        # Pemisahan 1:1:1
        print(f"  For {total_participants} participants, suggested 1:1:1 split:")
        print(f"    Training: {participants[:1]}")
        print(f"    Validation: {participants[1:2]}")
        print(f"    Test: {participants[2:3]}")
        if total_participants > 3:
            print(f"    Remaining for training: {participants[3:]}")
    else:
        print(f"  ⚠️  Only {total_participants} participants - consider session-based split instead")
    
    return {
        'total_participants': total_participants,
        'participants': participants,
        'participant_data': dict(participant_data),
        'recommendations': {
            'min_participants_for_manual': 3,
            'suggested_assignment': participants if total_participants >= 3 else None
        }
    }


def create_manual_participant_config(
    train_participants: List[str],
    val_participants: List[str] = None,
    test_participants: List[str] = None
) -> Dict:
    """
    Buat konfigurasi untuk penugasan peserta manual
    
    Args:
        train_participants: Daftar ID peserta untuk pelatihan
        val_participants: Daftar ID peserta untuk validasi
        test_participants: Daftar ID peserta untuk pengujian
        
    Returns:
        Dictionary konfigurasi
    """
    config = {
        'prevent_data_leakage': True,
        'split_method': 'participant',
        'manual_assignment': True,
        'train_participants': train_participants,
        'val_participants': val_participants or [],
        'test_participants': test_participants or [],
    }
    
    print(f"\n⚙️  MANUAL PARTICIPANT CONFIGURATION CREATED:")
    print(f"   Training: {config['train_participants']}")
    print(f"   Validation: {config['val_participants']}")
    print(f"   Test: {config['test_participants']}")
    
    return config


def participant_based_split(
    samples: List[np.ndarray],
    labels: List[int],
    file_ids: List[str],
    train_participants: List[str] = None,
    val_participants: List[str] = None,
    test_participants: List[str] = None,
    auto_split_participants: bool = True,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[
    List[np.ndarray], List[int], List[str],
    List[np.ndarray], List[int], List[str],
    List[np.ndarray], List[int], List[str],
    Dict
]:
    """
    Split data berdasarkan partisipan untuk mencegah data leakage
    
    Args:
        samples: Daftar array landmark
        labels: Daftar label kelas
        file_ids: Daftar ID file (format: "participant_gesture_sample.json")
        train_participants: Peserta spesifik untuk pelatihan (opsional)
        val_participants: Peserta spesifik untuk validasi (opsional) 
        test_participants: Peserta spesifik untuk pengujian (opsional)
        auto_split_participants: Pisahkan peserta secara otomatis jika tidak ditentukan
        train_ratio: Rasio untuk peserta pelatihan (jika auto_split)
        val_ratio: Rasio untuk peserta validasi (jika auto_split)
        test_ratio: Rasio untuk peserta pengujian (jika auto_split)
        random_seed: Seed acak untuk reproduktifitas
        
    Returns:
        Tuple dari (train_samples, train_labels, train_file_ids,
                 val_samples, val_labels, val_file_ids,
                 test_samples, test_labels, test_file_ids,
                 split_info)
    """
    
    # Atur seed acak
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Ekstrak ID peserta dari file_ids
    participant_data = defaultdict(list)
    
    for i, file_id in enumerate(file_ids):
        # Ekstrak ID peserta dari nama file (format: "participant_gesture_sample")
        participant_id = file_id.split('_')[0]
        participant_data[participant_id].append((i, samples[i], labels[i], file_id))
    
    unique_participants = list(participant_data.keys())
    print(f"\n🔍 Found {len(unique_participants)} unique participants: {unique_participants}")
    
    # Analisis distribusi data peserta
    participant_stats = {}
    for p_id, data in participant_data.items():
        class_dist = Counter([item[2] for item in data])
        participant_stats[p_id] = {
            'total_samples': len(data),
            'class_distribution': dict(class_dist)
        }
    
    # Tentukan pemisahan peserta
    if train_participants or val_participants or test_participants:
        # MODE PENUGASAN PESERTA MANUAL
        print(f"\n🎯 MANUAL PARTICIPANT ASSIGNMENT:")
        
        # Konversi ke daftar jika diberikan sebagai None
        train_participants = train_participants or []
        val_participants = val_participants or []
        test_participants = test_participants or []
        
        # Validasi bahwa semua peserta yang ditentukan ada
        all_specified = set(train_participants + val_participants + test_participants)
        missing_participants = all_specified - set(unique_participants)
        if missing_participants:
            raise ValueError(f"❌ Specified participants not found in data: {missing_participants}")
        
        # Tetapkan peserta yang tidak ditugaskan ke pelatihan secara default
        unassigned_participants = set(unique_participants) - all_specified
        if unassigned_participants:
            print(f"⚠️  Unassigned participants {list(unassigned_participants)} will be added to TRAINING set")
            train_participants.extend(list(unassigned_participants))
        
        print(f"   Training: {train_participants}")
        print(f"   Validation: {val_participants}")  
        print(f"   Test: {test_participants}")
        
        # Validasi bahwa setiap set memiliki peserta
        if not train_participants:
            raise ValueError("❌ Training set cannot be empty!")
        
    elif auto_split_participants and len(unique_participants) >= 3:
        # AUTO-SPLIT peserta berdasarkan rasio
        n_participants = len(unique_participants)
        n_train = max(1, int(n_participants * train_ratio))
        n_val = max(1, int(n_participants * val_ratio))
        n_test = max(1, n_participants - n_train - n_val)
        
        # Acak peserta untuk penugasan acak
        shuffled_participants = unique_participants.copy()
        random.shuffle(shuffled_participants)
        
        train_participants = shuffled_participants[:n_train]
        val_participants = shuffled_participants[n_train:n_train+n_val]
        test_participants = shuffled_participants[n_train+n_val:]
        
        print(f"\n🎯 AUTO-SPLIT PARTICIPANTS:")
        print(f"   Training: {train_participants}")
        print(f"   Validation: {val_participants}")
        print(f"   Test: {test_participants}")
    
    else:
        # Fallback ke pemisahan berbasis sesi untuk peserta terbatas
        print(f"\n⚠️  Only {len(unique_participants)} participants found - using session-based split")
        return session_based_split(samples, labels, file_ids, train_ratio, val_ratio, test_ratio, random_seed)
    
    # Tetapkan sampel ke pemisahan berdasarkan peserta
    train_samples, train_labels, train_file_ids = [], [], []
    val_samples, val_labels, val_file_ids = [], [], []
    test_samples, test_labels, test_file_ids = [], [], []
    
    for p_id, data in participant_data.items():
        if p_id in (train_participants or []):
            for _, sample, label, file_id in data:
                train_samples.append(sample)
                train_labels.append(label)
                train_file_ids.append(file_id)
        elif p_id in (val_participants or []):
            for _, sample, label, file_id in data:
                val_samples.append(sample)
                val_labels.append(label)
                val_file_ids.append(file_id)
        elif p_id in (test_participants or []):
            for _, sample, label, file_id in data:
                test_samples.append(sample)
                test_labels.append(label)
                test_file_ids.append(file_id)
    
    # Buat info pemisahan
    split_info = {
        'split_method': 'participant_based',
        'train_participants': train_participants or [],
        'val_participants': val_participants or [],
        'test_participants': test_participants or [],
        'participant_stats': participant_stats,
        'total_participants': len(unique_participants)
    }
    
    return (train_samples, train_labels, train_file_ids,
            val_samples, val_labels, val_file_ids,
            test_samples, test_labels, test_file_ids,
            split_info)


def session_based_split(
    samples: List[np.ndarray],
    labels: List[int], 
    file_ids: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[
    List[np.ndarray], List[int], List[str],
    List[np.ndarray], List[int], List[str], 
    List[np.ndarray], List[int], List[str],
    Dict
]:
    """
    Split data berdasarkan sesi untuk mencegah data leakage dalam single participant
    
    Args:
        samples: Daftar array landmark
        labels: Daftar label kelas
        file_ids: Daftar ID file (format: "participant_gesture_sample.json")
        train_ratio, val_ratio, test_ratio: Rasio pemisahan
        random_seed: Seed acak untuk reproduktifitas
        
    Returns:
        Tuple data pemisahan dan info pemisahan
    """
    
    # Atur seed acak
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Kelompokkan berdasarkan peserta dan sesi
    session_data = defaultdict(lambda: defaultdict(list))
    
    for i, file_id in enumerate(file_ids):
        parts = file_id.split('_')
        participant_id = parts[0]
        
        # Coba ekstrak info sesi dari CSV peserta atau gunakan nomor sampel sebagai proxy
        if len(parts) >= 3:
            sample_num = int(parts[2])
            # Kelompokkan sampel ke dalam sesi (setiap 5 sampel = 1 sesi)
            session_id = f"session_{sample_num // 5 + 1:03d}"
        else:
            session_id = "session_001"
            
        session_data[participant_id][session_id].append((i, samples[i], labels[i], file_id))
    
    # Kumpulkan semua sesi unik
    all_sessions = []
    for p_id, sessions in session_data.items():
        for s_id, data in sessions.items():
            all_sessions.append((p_id, s_id, data))
    
    print(f"\n🔍 Found {len(all_sessions)} unique sessions across participants")
    
    # Acak sesi untuk memastikan distribusi acak
    random.shuffle(all_sessions)
    
    # Pisahkan sesi berdasarkan rasio
    n_sessions = len(all_sessions)
    n_train = int(n_sessions * train_ratio)
    n_val = int(n_sessions * val_ratio)
    
    train_sessions = all_sessions[:n_train]
    val_sessions = all_sessions[n_train:n_train+n_val]
    test_sessions = all_sessions[n_train+n_val:]
    
    # Tetapkan sampel ke pemisahan
    train_samples, train_labels, train_file_ids = [], [], []
    val_samples, val_labels, val_file_ids = [], [], []
    test_samples, test_labels, test_file_ids = [], [], []
    
    for p_id, s_id, data in train_sessions:
        for _, sample, label, file_id in data:
            train_samples.append(sample)
            train_labels.append(label)
            train_file_ids.append(file_id)
    
    for p_id, s_id, data in val_sessions:
        for _, sample, label, file_id in data:
            val_samples.append(sample)
            val_labels.append(label)
            val_file_ids.append(file_id)
    
    for p_id, s_id, data in test_sessions:
        for _, sample, label, file_id in data:
            test_samples.append(sample)
            test_labels.append(label)
            test_file_ids.append(file_id)
    
    # Buat info pemisahan
    split_info = {
        'split_method': 'session_based',
        'total_sessions': len(all_sessions),
        'train_sessions': len(train_sessions),
        'val_sessions': len(val_sessions),
        'test_sessions': len(test_sessions)
    }
    
    return (train_samples, train_labels, train_file_ids,
            val_samples, val_labels, val_file_ids,
            test_samples, test_labels, test_file_ids,
            split_info)


def enhanced_split(
    samples: List[np.ndarray],
    labels: List[int],
    file_ids: List[str],
    split_method: str = 'auto',
    **kwargs
) -> Tuple[
    List[np.ndarray], List[int], List[str],
    List[np.ndarray], List[int], List[str],
    List[np.ndarray], List[int], List[str],
    Dict
]:
    """
    Enhanced split dengan multiple strategies untuk mencegah data leakage
    
    Args:
        samples: Daftar array landmark
        labels: Daftar label kelas 
        file_ids: Daftar ID file
        split_method: 'auto', 'participant', 'session', atau 'stratified'
        **kwargs: Argumen tambahan untuk metode pemisahan spesifik
        
    Returns:
        Tuple data pemisahan dan info pemisahan
    """
    
    print(f"\n🚀 ENHANCED DATA SPLITTING")
    print(f"="*50)
    print(f"Split method: {split_method}")
    print(f"Total samples: {len(samples)}")
    
    if split_method == 'auto':
        # Deteksi otomatis metode pemisahan terbaik
        participant_ids = set(file_id.split('_')[0] for file_id in file_ids)
        
        if len(participant_ids) >= 3:
            print(f"✅ {len(participant_ids)} participants found - using PARTICIPANT-based split")
            return participant_based_split(samples, labels, file_ids, **kwargs)
        elif len(participant_ids) >= 1:
            print(f"⚠️  Only {len(participant_ids)} participant(s) found - using SESSION-based split")
            return session_based_split(samples, labels, file_ids, **kwargs)
        else:
            print(f"❌ No participant info found - falling back to STRATIFIED split")
            train_X, train_y, val_X, val_y, test_X, test_y = stratified_split(samples, labels, **kwargs)
            return (train_X, train_y, file_ids[:len(train_X)],
                   val_X, val_y, file_ids[len(train_X):len(train_X)+len(val_X)],
                   test_X, test_y, file_ids[len(train_X)+len(val_X):],
                   {'split_method': 'stratified_fallback'})
    
    elif split_method == 'participant':
        return participant_based_split(samples, labels, file_ids, **kwargs)
    
    elif split_method == 'session':
        return session_based_split(samples, labels, file_ids, **kwargs)
    
    elif split_method == 'stratified':
        train_X, train_y, val_X, val_y, test_X, test_y = stratified_split(samples, labels, **kwargs)
        # Catatan: urutan file_ids mungkin tidak cocok setelah pemisahan bertingkat
        return (train_X, train_y, file_ids[:len(train_X)],
               val_X, val_y, file_ids[len(train_X):len(train_X)+len(val_X)],
               test_X, test_y, file_ids[len(train_X)+len(val_X):],
               {'split_method': 'stratified'})
    
    else:
        raise ValueError(f"Unknown split_method: {split_method}")


def print_enhanced_split_info(
    train_labels: List[int],
    val_labels: List[int], 
    test_labels: List[int],
    train_file_ids: List[str],
    val_file_ids: List[str],
    test_file_ids: List[str],
    split_info: Dict,
    class_names: List[str] = None
):
    """Cetak informasi rinci tentang pemisahan dataset yang ditingkatkan"""
    
    total = len(train_labels) + len(val_labels) + len(test_labels)
    
    print(f"\n" + "="*70)
    print(f"ENHANCED DATASET SPLIT SUMMARY")
    print(f"="*70)
    
    print(f"\n📊 SPLIT METHOD: {split_info.get('split_method', 'unknown').upper()}")
    
    print(f"\n📈 SAMPLE DISTRIBUTION:")
    print(f"  Total samples: {total}")
    print(f"  Training:   {len(train_labels):3d} ({len(train_labels)/total*100:.1f}%)")
    print(f"  Validation: {len(val_labels):3d} ({len(val_labels)/total*100:.1f}%)")
    print(f"  Test:       {len(test_labels):3d} ({len(test_labels)/total*100:.1f}%)")
    
    # Distribusi kelas
    train_dist = Counter(train_labels)
    val_dist = Counter(val_labels)
    test_dist = Counter(test_labels)
    
    all_classes = sorted(set(train_labels + val_labels + test_labels))
    
    print(f"\n📊 CLASS DISTRIBUTION:")
    for class_idx in all_classes:
        class_name = class_names[class_idx] if class_names else f"Class {class_idx}"
        train_count = train_dist.get(class_idx, 0)
        val_count = val_dist.get(class_idx, 0)
        test_count = test_dist.get(class_idx, 0)
        total_class = train_count + val_count + test_count
        
        print(f"  {class_name}:")
        print(f"    Train: {train_count:2d} ({train_count/total_class*100:.1f}%), "
              f"Val: {val_count:2d} ({val_count/total_class*100:.1f}%), "
              f"Test: {test_count:2d} ({test_count/total_class*100:.1f}%)")
    
    # Informasi spesifik metode
    if split_info.get('split_method') == 'participant_based':
        print(f"\n👥 PARTICIPANT-BASED SPLIT:")
        print(f"  Total participants: {split_info.get('total_participants', 0)}")
        print(f"  Training participants: {split_info.get('train_participants', [])}")
        print(f"  Validation participants: {split_info.get('val_participants', [])}")
        print(f"  Test participants: {split_info.get('test_participants', [])}")
        
        print(f"\n📊 PARTICIPANT STATISTICS:")
        for p_id, stats in split_info.get('participant_stats', {}).items():
            print(f"  {p_id}: {stats['total_samples']} samples, classes: {stats['class_distribution']}")
    
    elif split_info.get('split_method') == 'session_based':
        print(f"\n📅 SESSION-BASED SPLIT:")
        print(f"  Total sessions: {split_info.get('total_sessions', 0)}")
        print(f"  Training sessions: {split_info.get('train_sessions', 0)}")
        print(f"  Validation sessions: {split_info.get('val_sessions', 0)}")
        print(f"  Test sessions: {split_info.get('test_sessions', 0)}")
    
    # Pemeriksaan pencegahan kebocoran data
    train_participants = set(file_id.split('_')[0] for file_id in train_file_ids)
    val_participants = set(file_id.split('_')[0] for file_id in val_file_ids)  
    test_participants = set(file_id.split('_')[0] for file_id in test_file_ids)
    
    participant_overlap = (
        (train_participants & val_participants) |
        (train_participants & test_participants) |
        (val_participants & test_participants)
    )
    
    print(f"\n🛡️  DATA LEAKAGE CHECK:")
    if participant_overlap:
        print(f"  ⚠️  WARNING: Participant overlap detected: {participant_overlap}")
        print(f"  ⚠️  This may cause data leakage and overfitting!")
    else:
        print(f"  ✅ No participant overlap - good separation!")
    
    print(f"="*70 + "\n")


# Contoh penggunaan
if __name__ == '__main__':
    print("Testing train/test split...")
    
    # Data dummy
    samples = [np.random.rand(10, 126) for _ in range(60)]
    labels = [0] * 30 + [1] * 30
    
    train_X, train_y, val_X, val_y, test_X, test_y = stratified_split(
        samples, labels,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    print_split_info(train_y, val_y, test_y, class_names=['halo', 'C'])
    print("Test completed!")
