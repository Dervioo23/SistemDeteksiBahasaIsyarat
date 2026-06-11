"""
Setup Manual Participant Split
Alat untuk menganalisis dan mengkonfigurasi penugasan peserta manual
"""
import os
import json
from typing import Dict, List

# Tentukan direktori penting
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MANUAL_PARTICIPANT_DIR = os.path.join(CURRENT_DIR, "Manual_Participant")

from preprocessing import (
    load_multiple_gestures,
    analyze_participant_distribution,
    create_manual_participant_config,
    get_available_participants
)
from preprocessing.prepare_data import DataPreprocessor
from data_collection.utils import load_config


def analyze_current_dataset():
    """Menganalisis distribusi peserta dalam dataset saat ini"""
    
    print("\n" + "="*70)
    print("📊 DATASET PARTICIPANT ANALYSIS")
    print("="*70)
    
    # Muat konfigurasi
    config = load_config()
    
    # Dapatkan kategori isyarat yang tersedia
    print("\nAvailable gesture categories:")
    print("1. Words")
    print("2. Alphabet")
    
    while True:
        choice = input("\nSelect category to analyze (1-2): ").strip()
        if choice == '1':
            category = 'words'
            labels = config['vocabulary']['words']
            break
        elif choice == '2':
            category = 'alphabet'
            labels = config['vocabulary']['alphabet']
            break
        else:
            print("❌ Invalid choice! Please enter 1 or 2")
    
    print(f"\n🔍 Loading {category} data...")
    print(f"Labels: {labels}")
    
    # Muat data
    try:
        # Gunakan direktori dataset root - load_gesture_data akan menambahkan kategori
        data_dir = config['dataset']['root_dir']
        
        samples, class_indices, file_ids, metadatas = load_multiple_gestures(
            data_dir=data_dir,
            labels=labels,
            category=category
        )
        
        if not samples:
            print("❌ No data found! Please collect some data first.")
            return None
        
        print(f"✅ Loaded {len(samples)} samples")
        
        # Analisis peserta
        analysis = analyze_participant_distribution(
            file_ids=file_ids,
            labels=class_indices,
            class_names=labels
        )
        
        return {
            'category': category,
            'labels': labels,
            'samples': samples,
            'class_indices': class_indices,
            'file_ids': file_ids,
            'analysis': analysis
        }
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


def create_manual_assignment_interactive(analysis: Dict):
    """Pengaturan interaktif untuk penugasan peserta manual"""
    
    participants = analysis['analysis']['participants']
    total_participants = len(participants)
    
    print(f"\n" + "="*70)
    print("🎯 MANUAL PARTICIPANT ASSIGNMENT SETUP")
    print("="*70)
    
    print(f"Available participants: {participants}")
    print(f"Total participants: {total_participants}")
    
    if total_participants < 3:
        print(f"⚠️  Warning: Only {total_participants} participants available.")
        print("   Recommended minimum: 3 participants for proper splitting")
        print("   Consider collecting more data or using session-based split")
        
        proceed = input("\nProceed anyway? (y/n): ").strip().lower()
        if proceed not in ['y', 'yes']:
            return None
    
    print(f"\n📋 Assignment Options:")
    print(f"1. Quick Setup (recommended split)")
    print(f"2. Custom Assignment (manual selection)")
    print(f"3. Show Recommendations Only")
    
    while True:
        choice = input(f"\nSelect option (1-3): ").strip()
        if choice in ['1', '2', '3']:
            break
        print("❌ Invalid choice! Please enter 1, 2, or 3")
    
    if choice == '1':
        # Pengaturan cepat dengan rekomendasi
        print(f"\n🚀 QUICK SETUP - Using recommended split:")
        
        if total_participants >= 5:
            # Beberapa peserta untuk pelatihan, 1 masing-masing untuk val/test
            n_train = max(1, total_participants - 2)
            train_participants = participants[:n_train]
            val_participants = participants[n_train:n_train+1]
            test_participants = participants[n_train+1:n_train+2]
            
        elif total_participants >= 3:
            # Pembagian 1:1:1 dengan sisanya masuk ke pelatihan
            train_participants = participants[:1]
            val_participants = participants[1:2]
            test_participants = participants[2:3]
            
            # Tambahkan sisanya ke pelatihan
            if total_participants > 3:
                train_participants.extend(participants[3:])
        else:
            # Kasus minimal
            train_participants = participants[:1] if total_participants >= 1 else []
            val_participants = participants[1:2] if total_participants >= 2 else []
            test_participants = participants[2:3] if total_participants >= 3 else []
    
    elif choice == '2':
        # Penugasan manual kustom
        print(f"\n✏️  CUSTOM ASSIGNMENT:")
        print(f"Available participants: {participants}")
        print(f"Note: Unassigned participants will automatically go to Training set")
        
        # Dapatkan peserta pelatihan
        print(f"\n📚 Training Set Assignment:")
        train_input = input(f"Enter training participants (comma-separated, e.g., '001,002') or press Enter for auto: ").strip()
        train_participants = [p.strip() for p in train_input.split(',') if p.strip()] if train_input else []
        
        # Dapatkan peserta validasi
        print(f"\n🔍 Validation Set Assignment:")
        val_input = input(f"Enter validation participants (comma-separated) or press Enter for auto: ").strip()
        val_participants = [p.strip() for p in val_input.split(',') if p.strip()] if val_input else []
        
        # Dapatkan peserta pengujian
        print(f"\n🧪 Test Set Assignment:")
        test_input = input(f"Enter test participants (comma-separated) or press Enter for auto: ").strip()
        test_participants = [p.strip() for p in test_input.split(',') if p.strip()] if test_input else []
        
        # Validasi peserta ada
        all_specified = set(train_participants + val_participants + test_participants)
        invalid_participants = all_specified - set(participants)
        if invalid_participants:
            print(f"❌ Error: Invalid participants specified: {invalid_participants}")
            return None
    
    else:
        # Tampilkan rekomendasi saja
        return {'show_recommendations_only': True}
    
    # Buat dan validasi penugasan
    assignment = {
        'train_participants': train_participants,
        'val_participants': val_participants,
        'test_participants': test_participants
    }
    
    # Tampilkan penugasan
    print(f"\n📋 FINAL ASSIGNMENT:")
    print(f"   Training: {assignment['train_participants']}")
    print(f"   Validation: {assignment['val_participants']}")
    print(f"   Test: {assignment['test_participants']}")
    
    # Hitung distribusi sampel
    participant_data = analysis['analysis']['participant_data']
    
    train_samples = sum(sum(participant_data[p].values()) for p in assignment['train_participants'] if p in participant_data)
    val_samples = sum(sum(participant_data[p].values()) for p in assignment['val_participants'] if p in participant_data)  
    test_samples = sum(sum(participant_data[p].values()) for p in assignment['test_participants'] if p in participant_data)
    total_samples = train_samples + val_samples + test_samples
    
    print(f"\n📊 SAMPLE DISTRIBUTION:")
    print(f"   Training: {train_samples} samples ({train_samples/total_samples*100:.1f}%)")
    print(f"   Validation: {val_samples} samples ({val_samples/total_samples*100:.1f}%)")
    print(f"   Test: {test_samples} samples ({test_samples/total_samples*100:.1f}%)")
    print(f"   Total: {total_samples} samples")
    
    # Konfirmasi penugasan
    confirm = input(f"\n✅ Confirm this assignment? (y/n): ").strip().lower()
    if confirm in ['y', 'yes']:
        return assignment
    else:
        print("❌ Assignment cancelled")
        return None


def save_manual_config(assignment: Dict, analysis: Dict):
    """Simpan konfigurasi penugasan manual"""
    
    print(f"\n" + "="*70)
    print("💾 SAVING MANUAL PARTICIPANT CONFIGURATION")
    print("="*70)

    # Pastikan direktori Manual_Participant ada
    os.makedirs(MANUAL_PARTICIPANT_DIR, exist_ok=True)

    # Buat konfigurasi dengan penugasan manual
    config = {
        'data_dir': 'dataset',
        'output_dir': 'preprocessed_data',
        'normalize': True,
        'augment': True,
        'augmentation_factor': 2,
        'pad_sequences': True,
        'max_length': 45,
        'train_ratio': 0.67,
        'val_ratio': 0.17,
        'test_ratio': 0.16,
        'random_seed': 42,
        
        # Pemisahan yang ditingkatkan dengan penugasan manual
        'prevent_data_leakage': True,
        'split_method': 'participant',
        'analyze_participants': True,
        'train_participants': assignment['train_participants'],
        'val_participants': assignment['val_participants'],
        'test_participants': assignment['test_participants']
    }
    
    # Simpan konfigurasi
    config_path = os.path.join(
        MANUAL_PARTICIPANT_DIR,
        f"manual_participant_config_{analysis['category']}.json"
    )
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Configuration saved to: {config_path}")
    
    # Simpan info penugasan untuk referensi
    assignment_info = {
        'category': analysis['category'],
        'labels': analysis['labels'],
        'total_participants': len(analysis['analysis']['participants']),
        'assignment': assignment,
        'participant_data': analysis['analysis']['participant_data']
    }
    
    info_path = os.path.join(
        MANUAL_PARTICIPANT_DIR,
        f"participant_assignment_info_{analysis['category']}.json"
    )
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(assignment_info, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Assignment info saved to: {info_path}")
    
    return config_path


def run_preprocessing_with_manual_assignment(config_path: str, analysis: Dict):
    """Jalankan pra-pemrosesan dengan penugasan manual"""
    
    print(f"\n" + "="*70)
    print("🚀 RUNNING PREPROCESSING WITH MANUAL ASSIGNMENT")
    print("="*70)
    
    # Muat konfigurasi manual
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Inisialisasi pra-pemroses
    preprocessor = DataPreprocessor(config=config)
    
    try:
        # Jalankan pra-pemrosesan
        data_splits = preprocessor.run_full_pipeline(
            labels=analysis['labels'],
            category=analysis['category']
        )
        
        print(f"\n🎉 PREPROCESSING COMPLETED WITH MANUAL ASSIGNMENT!")
        print(f"   Output directory: {config['output_dir']}")
        print(f"   Split method: {data_splits.get('split_info', {}).get('split_method', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Pengaturan interaktif utama"""
    
    print("\n" + "="*70)
    print("🎯 MANUAL PARTICIPANT SPLIT SETUP TOOL")
    print("="*70)
    print("This tool helps you:")
    print("• Analyze participant distribution in your dataset")
    print("• Setup manual participant assignment for train/val/test")
    print("• Run preprocessing with proper participant separation")
    print("="*70)
    
    # Langkah 1: Analisis dataset
    print(f"\n📊 STEP 1: DATASET ANALYSIS")
    analysis = analyze_current_dataset()
    
    if not analysis:
        print("❌ Cannot proceed without dataset analysis")
        return
    
    # Langkah 2: Pengaturan penugasan manual
    print(f"\n🎯 STEP 2: MANUAL ASSIGNMENT SETUP")
    assignment = create_manual_assignment_interactive(analysis)
    
    if not assignment:
        print("❌ Setup cancelled or no assignment created")
        return
    
    if assignment.get('show_recommendations_only'):
        print("✅ Recommendations shown. Run again to create actual assignment.")
        return
    
    # Langkah 3: Simpan konfigurasi
    print(f"\n💾 STEP 3: SAVE CONFIGURATION")
    config_path = save_manual_config(assignment, analysis)
    
    # Langkah 4: Opsi untuk menjalankan pra-pemrosesan segera
    print(f"\n🚀 STEP 4: RUN PREPROCESSING (OPTIONAL)")
    run_now = input("Run preprocessing now with this configuration? (y/n): ").strip().lower()
    
    if run_now in ['y', 'yes']:
        success = run_preprocessing_with_manual_assignment(config_path, analysis)
        if success:
            print(f"\n✅ COMPLETE! Your data has been split with manual participant assignment")
        else:
            print(f"\n❌ Preprocessing failed, but configuration is saved for later use")
    else:
        print(f"\n💡 TO RUN LATER:")
        print(f"   python run_preprocessing.py --config {config_path}")
        print(f"   Or modify your existing preprocessing script to use: {config_path}")
    
    print(f"\n🎉 MANUAL PARTICIPANT SETUP COMPLETED!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n❌ Setup cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n❌ Setup cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
