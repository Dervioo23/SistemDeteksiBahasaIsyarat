"""
Test Enhanced Data Splitting
Script untuk menguji pemisahan data berdasarkan partisipan dan sesi
"""
import os
import numpy as np

from preprocessing.train_test_split import (
    enhanced_split,
    print_enhanced_split_info,
    participant_based_split,
    session_based_split
)


def create_dummy_data():
    """Create dummy data untuk testing"""
    
    # Simulasikan beberapa peserta dengan beberapa sesi
    participants = ['001', '002', '003']
    gestures = ['halo', 'terimakasih']
    samples_per_gesture = 15
    
    samples = []
    labels = []
    file_ids = []
    
    for p_idx, participant in enumerate(participants):
        for g_idx, gesture in enumerate(gestures):
            for sample_idx in range(samples_per_gesture):
                # Buat data landmark dummy
                sample = np.random.rand(45, 126)  # 45 frames, 126 features
                samples.append(sample)
                labels.append(g_idx)
                
                # Buat file_id dengan format participant_gesture_sample
                file_id = f"{participant}_{gesture}_{sample_idx+1:03d}"
                file_ids.append(file_id)
    
    print(f"✅ Created dummy data:")
    print(f"   - Participants: {participants}")
    print(f"   - Gestures: {gestures}")
    print(f"   - Total samples: {len(samples)}")
    print(f"   - Sample file_ids: {file_ids[:5]}...")
    
    return samples, labels, file_ids, gestures


def test_participant_based_split():
    """Uji pemisahan berbasis peserta"""
    print(f"\n" + "="*70)
    print("TEST 1: PARTICIPANT-BASED SPLITTING")
    print("="*70)
    
    samples, labels, file_ids, class_names = create_dummy_data()
    
    # Uji pemisahan peserta
    (train_X, train_y, train_file_ids,
     val_X, val_y, val_file_ids,
     test_X, test_y, test_file_ids,
     split_info) = participant_based_split(
        samples=samples,
        labels=labels,
        file_ids=file_ids,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        random_seed=42
    )
    
    print_enhanced_split_info(
        train_y, val_y, test_y,
        train_file_ids, val_file_ids, test_file_ids,
        split_info, class_names=class_names
    )


def test_session_based_split():
    """Uji pemisahan berbasis sesi"""
    print(f"\n" + "="*70)
    print("TEST 2: SESSION-BASED SPLITTING")
    print("="*70)
    
    # Buat data dari satu peserta dengan beberapa sesi
    samples = []
    labels = []
    file_ids = []
    
    participant = '001'
    gestures = ['halo', 'terimakasih']
    samples_per_session = 10
    num_sessions = 6
    
    for session_idx in range(num_sessions):
        for g_idx, gesture in enumerate(gestures):
            for sample_idx in range(samples_per_session):
                sample = np.random.rand(45, 126)
                samples.append(sample)
                labels.append(g_idx)
                
                # Gunakan nomor sampel sebagai proxy sesi
                overall_sample_num = (session_idx * samples_per_session) + sample_idx + 1
                file_id = f"{participant}_{gesture}_{overall_sample_num:03d}"
                file_ids.append(file_id)
    
    print(f"✅ Created session data:")
    print(f"   - Participant: {participant}")
    print(f"   - Sessions: {num_sessions}")
    print(f"   - Total samples: {len(samples)}")
    
    # Uji pemisahan sesi
    (train_X, train_y, train_file_ids,
     val_X, val_y, val_file_ids,
     test_X, test_y, test_file_ids,
     split_info) = session_based_split(
        samples=samples,
        labels=labels,
        file_ids=file_ids,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        random_seed=42
    )
    
    print_enhanced_split_info(
        train_y, val_y, test_y,
        train_file_ids, val_file_ids, test_file_ids,
        split_info, class_names=gestures
    )


def test_enhanced_auto_split():
    """Uji pemisahan deteksi otomatis yang ditingkatkan"""
    print(f"\n" + "="*70)
    print("TEST 3: ENHANCED AUTO-DETECTION SPLITTING")
    print("="*70)
    
    samples, labels, file_ids, class_names = create_dummy_data()
    
    # Uji pemisahan otomatis (harus memilih berbasis peserta)
    (train_X, train_y, train_file_ids,
     val_X, val_y, val_file_ids,
     test_X, test_y, test_file_ids,
     split_info) = enhanced_split(
        samples=samples,
        labels=labels,
        file_ids=file_ids,
        split_method='auto',
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        random_seed=42
    )
    
    print_enhanced_split_info(
        train_y, val_y, test_y,
        train_file_ids, val_file_ids, test_file_ids,
        split_info, class_names=class_names
    )


def test_data_leakage_check():
    """Uji deteksi kebocoran data"""
    print(f"\n" + "="*70)
    print("TEST 4: DATA LEAKAGE DETECTION")
    print("="*70)
    
    # Buat data dengan sengaja ada overlap partisipan
    samples = []
    labels = []
    file_ids = []
    
    # Partisipan 001 akan ada di train dan test (simulasi pemisahan yang buruk)
    participants = ['001', '001', '002']  # Duplikat 001
    splits = ['train', 'test', 'val']
    
    for p_idx, (participant, split_type) in enumerate(zip(participants, splits)):
        for sample_idx in range(10):
            sample = np.random.rand(45, 126)
            samples.append(sample)
            labels.append(0)  # Isyarat yang sama untuk kesederhanaan
            
            file_id = f"{participant}_halo_{sample_idx+1:03d}"
            file_ids.append(file_id)
    
    # Pemisahan manual untuk membuat overlap
    train_samples = samples[:10]
    train_labels = labels[:10]
    train_file_ids = file_ids[:10]
    
    val_samples = samples[20:30]
    val_labels = labels[20:30] 
    val_file_ids = file_ids[20:30]
    
    test_samples = samples[10:20]
    test_labels = labels[10:20]
    test_file_ids = file_ids[10:20]
    
    split_info = {'split_method': 'manual_with_leakage'}
    
    print("🧪 Testing data leakage detection...")
    print_enhanced_split_info(
        train_labels, val_labels, test_labels,
        train_file_ids, val_file_ids, test_file_ids,
        split_info, class_names=['halo']
    )


def main():
    """Jalankan semua tes"""
    print("🧪 ENHANCED DATA SPLITTING TESTS")
    print("="*70)
    
    try:
        # Tes 1: Pemisahan berbasis peserta
        test_participant_based_split()
        
        # Tes 2: Pemisahan berbasis sesi
        test_session_based_split()
        
        # Tes 3: Deteksi otomatis
        test_enhanced_auto_split()
        
        # Tes 4: Deteksi kebocoran data
        test_data_leakage_check()
        
        print(f"\n" + "="*70)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        print(f"\n💡 KEY BENEFITS:")
        print(f"   ✅ Participant-based split prevents person-specific overfitting")
        print(f"   ✅ Session-based split prevents temporal overfitting")
        print(f"   ✅ Auto-detection chooses best method based on data")
        print(f"   ✅ Data leakage detection warns about overlap")
        print(f"   ✅ Enhanced statistics show detailed split information")
        
        print(f"\n📊 USAGE RECOMMENDATIONS:")
        print(f"   • Use 'participant' split when ≥3 participants available")
        print(f"   • Use 'session' split for single participant with multiple sessions")
        print(f"   • Use 'auto' for automatic best method selection")
        print(f"   • Always check data leakage warnings in output")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
