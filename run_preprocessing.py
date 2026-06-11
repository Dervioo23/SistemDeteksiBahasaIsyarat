import os

def check_dataset():
    """Periksa apakah dataset ada dan deteksi semua label"""
    word_data = os.path.exists('dataset/words')
    alphabet_data = os.path.exists('dataset/alphabet')
    
    word_labels = []
    alphabet_labels = []
    word_total_samples = 0
    alphabet_total_samples = 0
    
    # Deteksi otomatis label kata
    if word_data:
        word_labels = [d for d in os.listdir('dataset/words') 
                      if os.path.isdir(os.path.join('dataset/words', d))]
        for label in word_labels:
            files = [f for f in os.listdir(os.path.join('dataset/words', label)) if f.endswith('.json')]
            word_total_samples += len(files)
    
    # Deteksi otomatis label alfabet
    if alphabet_data:
        alphabet_labels = [d for d in os.listdir('dataset/alphabet') 
                          if os.path.isdir(os.path.join('dataset/alphabet', d))]
        for label in alphabet_labels:
            files = [f for f in os.listdir(os.path.join('dataset/alphabet', label)) if f.endswith('.json')]
            alphabet_total_samples += len(files)
    
    return word_data, alphabet_data, word_labels, alphabet_labels, word_total_samples, alphabet_total_samples


def run_preprocessing():
    """Jalankan pipeline pra-pemrosesan"""
    from preprocessing.prepare_data import DataPreprocessor
    
    print("\n" + "="*60)
    print("DATA PREPROCESSING")
    print("="*60)
    
    # Periksa dataset
    print("\n📊 Checking dataset...")
    word_exists, alphabet_exists, word_labels, alphabet_labels, word_count, alphabet_count = check_dataset()
    
    if not word_exists and not alphabet_exists:
        print("\n❌ No dataset found!")
        print("\n📝 Please collect data first:")
        print("   python data_collection/collect_words.py")
        print("   python data_collection/collect_alphabet.py")
        return False
    
    print(f"\n✅ Dataset found:")
    if word_exists:
        print(f"   Words: {len(word_labels)} gestures ({', '.join(word_labels)})")
        print(f"   Total word samples: {word_count}")
    if alphabet_exists:
        print(f"   Alphabet: {len(alphabet_labels)} letters ({', '.join(alphabet_labels)})")
        print(f"   Total alphabet samples: {alphabet_count}")
    
    # Menu pra-pemrosesan
    print("\n" + "="*60)
    print("SELECT PREPROCESSING MODE")
    print("="*60)
    print("\n1. Preprocess Word Data only")
    print("2. Preprocess Alphabet Data only")
    print("3. Preprocess Both (Recommended)")
    print("4. Exit")
    
    choice = input("\nSelect option (1/2/3/4): ").strip()
    
    if choice == '1':
        # Pra-pemrosesan kata
        if not word_exists:
            print("\n❌ Word data not available!")
            return False
        
        print("\n🚀 Preprocessing word data...")
        config = {
            'data_dir': 'dataset',
            'output_dir': 'preprocessed_data/words',
            'normalize': True,
            'augment': False,  # DINONAKTIFKAN: Perbaiki masalah semua-nol
            'augmentation_factor': 0,
            'pad_sequences': True,
            'max_length': 45,
            'train_ratio': 0.67,
            'val_ratio': 0.17,
            'test_ratio': 0.16,
            'random_seed': 42
        }
        
        preprocessor = DataPreprocessor(config=config)
        preprocessor.run_full_pipeline(labels=word_labels, category='words')
        
        print(f"\n✅ Word data preprocessing completed!")
        print(f"   Processed {len(word_labels)} gestures: {', '.join(word_labels)}")
        return True
    
    elif choice == '2':
        # Pra-pemrosesan alfabet
        if not alphabet_exists:
            print("\n❌ Alphabet data not available!")
            return False
        
        print("\n🚀 Preprocessing alphabet data...")
        config = {
            'data_dir': 'dataset',
            'output_dir': 'preprocessed_data/alphabet',
            'normalize': True,
            'augment': False,  # DINONAKTIFKAN: Perbaiki masalah semua-nol
            'augmentation_factor': 0,
            'pad_sequences': False,
            'max_length': 1,
            'train_ratio': 0.67,
            'val_ratio': 0.17,
            'test_ratio': 0.16,
            'random_seed': 42
        }
        
        preprocessor = DataPreprocessor(config=config)
        preprocessor.run_full_pipeline(labels=alphabet_labels, category='alphabet')
        
        print(f"\n✅ Alphabet data preprocessing completed!")
        print(f"   Processed {len(alphabet_labels)} letters: {', '.join(alphabet_labels)}")
        return True
    
    elif choice == '3':
        # Pra-pemrosesan keduanya
        print("\n🚀 Preprocessing word and alphabet data...")
        
        # Proses kata
        if word_exists:
            print("\n[1/2] Processing word data...")
            config = {
                'data_dir': 'dataset',
                'output_dir': 'preprocessed_data/words',
                'normalize': True,
                'augment': False,  # DINONAKTIFKAN: Perbaiki masalah semua-nol
                'augmentation_factor': 0,
                'pad_sequences': True,
                'max_length': 45,
                'train_ratio': 0.67,
                'val_ratio': 0.17,
                'test_ratio': 0.16,
                'random_seed': 42
            }
            
            preprocessor_words = DataPreprocessor(config=config)
            preprocessor_words.run_full_pipeline(labels=word_labels, category='words')
            print(f"✅ Word data complete! ({len(word_labels)} gestures)")
        
        # Proses alfabet
        if alphabet_exists:
            print("\n[2/2] Processing alphabet data...")
            config = {
                'data_dir': 'dataset',
                'output_dir': 'preprocessed_data/alphabet',
                'normalize': True,
                'augment': False,  # DINONAKTIFKAN: Perbaiki masalah semua-nol
                'augmentation_factor': 0,
                'pad_sequences': False,
                'max_length': 1,
                'train_ratio': 0.67,
                'val_ratio': 0.17,
                'test_ratio': 0.16,
                'random_seed': 42
            }
            
            preprocessor_alphabet = DataPreprocessor(config=config)
            preprocessor_alphabet.run_full_pipeline(labels=alphabet_labels, category='alphabet')
            print(f"✅ Alphabet data complete! ({len(alphabet_labels)} letters)")
        
        print("\n✅ All data preprocessing completed!")
        return True
    
    elif choice == '4':
        print("\n👋 Exiting...")
        return False
    
    else:
        print("\n❌ Invalid choice!")
        return False


def main():
    """Fungsi utama"""
    
    print("\n" + "="*60)
    print("SIGN LANGUAGE DETECTION - DATA PREPROCESSING")
    print("="*60)
    
    success = run_preprocessing()
    
    if success:
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETED")
        print("="*60)
        print("\n✅ Preprocessed data saved in: preprocessed_data/")
        print("\n🚀 Next step - Train models:")
        print("   python run_training.py")
        print("\n" + "="*60)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Preprocessing interrupted by user")
        print("👋 Goodbye!")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
