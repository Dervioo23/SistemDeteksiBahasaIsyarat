import os
import time

from data_collection.utils import load_config


def check_preprocessed_data():
    """Periksa apakah data yang telah diproses sebelumnya ada"""
    config = load_config()
    dataset_cfg = config.get("dataset", {})
    words_pre_dir = dataset_cfg.get("preprocessed_words_dir", "preprocessed_data/words")
    alphabet_pre_dir = dataset_cfg.get("preprocessed_alphabet_dir", "preprocessed_data/alphabet")
    
    word_data = os.path.exists(os.path.join(words_pre_dir, 'train_X.npy'))
    alphabet_data = os.path.exists(os.path.join(alphabet_pre_dir, 'train_X.npy'))
    
    return word_data, alphabet_data


def train_word_model():
    """Latih model kata"""
    print("\n" + "="*60)
    print("TRAINING WORD MODEL")
    print("="*60)
    
    from training.train_word_model import train_word_model
    config = load_config()
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    data_dir = dataset_cfg.get("preprocessed_words_dir", "preprocessed_data/words")
    output_dir = model_cfg.get("trained_models_dir", "trained_models")
    
    try:
        model, history = train_word_model(
            model_name='word_halo_model',
            model_type='default',
            data_dir=data_dir,
            output_dir=output_dir,
            epochs=50,
            batch_size=8,
            learning_rate=0.001
        )
        
        print("\n✅ Word model training completed!")
        return True
    
    except Exception as e:
        print(f"\n❌ Error training word model: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_alphabet_model():
    """Latih model alfabet"""
    print("\n" + "="*60)
    print("TRAINING ALPHABET MODEL")
    print("="*60)
    
    from training.train_alphabet_model import train_alphabet_model
    config = load_config()
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    data_dir = dataset_cfg.get("preprocessed_alphabet_dir", "preprocessed_data/alphabet")
    output_dir = model_cfg.get("trained_models_dir", "trained_models")
    
    try:
        model, history = train_alphabet_model(
            model_name='alphabet_C_model',
            model_type='simple',
            data_dir=data_dir,
            output_dir=output_dir,
            epochs=50,
            batch_size=8,
            learning_rate=0.001
        )
        
        print("\n✅ Alphabet model training completed!")
        return True
    
    except Exception as e:
        print(f"\n❌ Error training alphabet model: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_multiclass_word_model():
    """Latih model kata multi-kelas"""
    print("\n" + "="*60)
    print("TRAINING MULTI-CLASS WORD MODEL")
    print("="*60)
    
    from training.train_multiclass_word import train_multiclass_word_model
    config = load_config()
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    data_dir = dataset_cfg.get("preprocessed_words_dir", "preprocessed_data/words")
    output_dir = model_cfg.get("trained_models_dir", "trained_models")
    
    try:
        model, history = train_multiclass_word_model(
            model_name='multiclass_word_model',
            model_type='default',
            data_dir=data_dir,
            output_dir=output_dir,
            epochs=50,
            batch_size=8,
            learning_rate=0.001
        )
        
        print("\n✅ Multi-class word model training completed!")
        return True
    
    except Exception as e:
        print(f"\n❌ Error training multi-class word model: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_multiclass_alphabet_model():
    """Latih model alfabet multi-kelas"""
    print("\n" + "="*60)
    print("TRAINING MULTI-CLASS ALPHABET MODEL")
    print("="*60)
    
    from training.train_multiclass_alphabet import train_multiclass_alphabet_model
    config = load_config()
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    data_dir = dataset_cfg.get("preprocessed_alphabet_dir", "preprocessed_data/alphabet")
    output_dir = model_cfg.get("trained_models_dir", "trained_models")
    
    try:
        model, history = train_multiclass_alphabet_model(
            model_name='multiclass_alphabet_model',
            model_type='default',
            data_dir=data_dir,
            output_dir=output_dir,
            epochs=50,
            batch_size=16,
            learning_rate=0.001
        )
        
        print("\n✅ Multi-class alphabet model training completed!")
        return True
    
    except Exception as e:
        print(f"\n❌ Error training multi-class alphabet model: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Peluncur pelatihan utama"""
    
    print("\n" + "="*60)
    print("SIGN LANGUAGE DETECTION - MODEL TRAINING")
    print("="*60)
    
    # Muat konfigurasi
    config = load_config()
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    words_pre_dir = dataset_cfg.get("preprocessed_words_dir", "preprocessed_data/words")
    alphabet_pre_dir = dataset_cfg.get("preprocessed_alphabet_dir", "preprocessed_data/alphabet")
    
    # Periksa data yang telah diproses sebelumnya
    print("\n📊 Checking preprocessed data...")
    word_data, alphabet_data = check_preprocessed_data()
    
    if not word_data:
        print(f"   ❌ Word data not found: {words_pre_dir}/")
    else:
        print("   ✅ Word data found")
    
    if not alphabet_data:
        print(f"   ❌ Alphabet data not found: {alphabet_pre_dir}/")
    else:
        print("   ✅ Alphabet data found")
    
    if not word_data and not alphabet_data:
        print("\n❌ No preprocessed data found!")
        print("\n📝 Please preprocess data first:")
        print("   python preprocessing/prepare_data.py")
        return
    
    # Periksa kelas latar belakang
    has_background = False
    if word_data:
        # Kita tidak dapat dengan mudah memeriksa konten file .npy di sini tanpa memuatnya, 
        # tetapi kita dapat memeriksa folder dataset sebagai proxy atau hanya mengandalkan pilihan pengguna.
        # Lebih baik: Periksa jika 'dataset/words/_background' ada
        if os.path.exists('dataset/words/_background'):
            has_background = True
    
    # Menu pelatihan
    print("\n" + "="*60)
    print("SELECT TRAINING MODE")
    print("="*60)
    
    if has_background:
        print("\n⚠️  'dataset/words/_background' detected!")
        print("   You MUST use Multi-Class Training (Option 4, 5, or 6).")
        print("   Binary models (1, 2, 3) might not work as expected with background data.")
        
    print("\n📚 BINARY CLASSIFICATION (Single gesture per model):")
    print("   1. Train Word Model (Binary)")
    print("   2. Train Alphabet Model (Binary)")
    print("   3. Train Both Binary Models")
    print("\n🎯 MULTI-CLASS CLASSIFICATION (Multiple gestures per model):")
    print("   4. Train Multi-Class Word Model ⭐ (RECOMMENDED)")
    print("   5. Train Multi-Class Alphabet Model ⭐")
    print("   6. Train Both Multi-Class Models ⭐")
    print("\n   7. Exit")
    
    choice = input("\nSelect option (1-7): ").strip()
    
    start_time = time.time()
    
    if choice == '1':
        # Latih model kata saja
        if not word_data:
            print("\n❌ Word data not available!")
            return
        
        print("\n🚀 Starting word model training...")
        success = train_word_model()
        
        if success:
            print("\n🎉 Word model ready!")
            word_model_path = model_cfg.get("word_model_path", "trained_models/word_halo_model_best.keras")
            print(f"   Model: {word_model_path}")
    
    elif choice == '2':
        # Latih model alfabet saja
        if not alphabet_data:
            print("\n❌ Alphabet data not available!")
            return
        
        print("\n🚀 Starting alphabet model training...")
        success = train_alphabet_model()
        
        if success:
            print("\n🎉 Alphabet model ready!")
            alphabet_model_path = model_cfg.get("alphabet_model_path", "trained_models/alphabet_C_model_best.keras")
            print(f"   Model: {alphabet_model_path}")
    
    elif choice == '3':
        # Latih kedua model
        results = []
        
        # Latih model kata
        if word_data:
            print("\n🚀 [1/2] Starting word model training...")
            success_word = train_word_model()
            results.append(('Word', success_word))
            
            if success_word:
                print("\n✅ Word model training completed!")
            else:
                print("\n❌ Word model training failed!")
        else:
            print("\n⚠️  Skipping word model (data not available)")
        
        # Latih model alfabet
        if alphabet_data:
            print("\n🚀 [2/2] Starting alphabet model training...")
            success_alphabet = train_alphabet_model()
            results.append(('Alphabet', success_alphabet))
            
            if success_alphabet:
                print("\n✅ Alphabet model training completed!")
            else:
                print("\n❌ Alphabet model training failed!")
        else:
            print("\n⚠️  Skipping alphabet model (data not available)")
        
        # Ringkasan
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        for model_name, success in results:
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"   {model_name} Model: {status}")
        
        all_success = all(success for _, success in results)
        
        if all_success:
            print("\n🎉 All models trained successfully!")
            print("\n📦 Trained models saved in: trained_models/")
            word_model_path = model_cfg.get("word_model_path", "trained_models/word_halo_model_best.keras")
            alphabet_model_path = model_cfg.get("alphabet_model_path", "trained_models/alphabet_C_model_best.keras")
            print(f"   - {os.path.basename(word_model_path)}")
            print(f"   - {os.path.basename(alphabet_model_path)}")
        else:
            print("\n⚠️  Some models failed to train. Check errors above.")
    
    elif choice == '4':
        # Latih model kata multi-kelas
        if not word_data:
            print("\n❌ Word data not available!")
            return
        
        print("\n🚀 Starting multi-class word model training...")
        success = train_multiclass_word_model()
        
        if success:
            print("\n🎉 Multi-class word model ready!")
            mc_word_model_path = model_cfg.get("multiclass_word_model_path", "trained_models/multiclass_word_model_final.keras")
            print(f"   Model: {mc_word_model_path}")
    
    elif choice == '5':
        # Latih model alfabet multi-kelas
        if not alphabet_data:
            print("\n❌ Alphabet data not available!")
            return
        
        print("\n🚀 Starting multi-class alphabet model training...")
        success = train_multiclass_alphabet_model()
        
        if success:
            print("\n🎉 Multi-class alphabet model ready!")
            mc_alpha_model_path = model_cfg.get("multiclass_alphabet_model_path", "trained_models/multiclass_alphabet_model_final.keras")
            print(f"   Model: {mc_alpha_model_path}")
    
    elif choice == '6':
        # Latih kedua model multi-kelas
        results = []
        
        # Latih model kata multi-kelas
        if word_data:
            print("\n🚀 [1/2] Starting multi-class word model training...")
            success_word = train_multiclass_word_model()
            results.append(('Multi-Class Word', success_word))
            
            if success_word:
                print("\n✅ Multi-class word model training completed!")
            else:
                print("\n❌ Multi-class word model training failed!")
        else:
            print("\n⚠️  Skipping word model (data not available)")
        
        # Latih model alfabet multi-kelas
        if alphabet_data:
            print("\n🚀 [2/2] Starting multi-class alphabet model training...")
            success_alphabet = train_multiclass_alphabet_model()
            results.append(('Multi-Class Alphabet', success_alphabet))
            
            if success_alphabet:
                print("\n✅ Multi-class alphabet model training completed!")
            else:
                print("\n❌ Multi-class alphabet model training failed!")
        else:
            print("\n⚠️  Skipping alphabet model (data not available)")
        
        # Ringkasan
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        for model_name, success in results:
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"   {model_name}: {status}")
        
        all_success = all(success for _, success in results)
        
        if all_success:
            print("\n🎉 All multi-class models trained successfully!")
            print("\n📦 Trained models saved in: trained_models/")
            mc_word_model_path = model_cfg.get("multiclass_word_model_path", "trained_models/multiclass_word_model_final.keras")
            mc_alpha_model_path = model_cfg.get("multiclass_alphabet_model_path", "trained_models/multiclass_alphabet_model_final.keras")
            print(f"   - {os.path.basename(mc_word_model_path)}")
            print(f"   - {os.path.basename(mc_alpha_model_path)}")
            print("\n💡 Use these models for NO-CONFUSION detection!")
        else:
            print("\n⚠️  Some models failed to train. Check errors above.")
    
    elif choice == '7':
        print("\n👋 Goodbye!")
        return
    
    else:
        print("\n❌ Invalid choice!")
        return
    
    # Hitung total waktu
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    
    print(f"\n⏱️  Total training time: {minutes}m {seconds}s")
    
    # Langkah selanjutnya
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("\n✅ Models trained successfully!")
    print("\n🚀 Run inference system:")
    print("   python run_inference.py")
    print("\n📊 Or evaluate models:")
    print("   python training/evaluate.py")
    print("\n" + "="*60)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("👋 Goodbye!")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
