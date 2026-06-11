import os
import json
from datetime import datetime


def create_collection_plan(
    word_list: list = None,
    alphabet_list: list = None,
    samples_per_gesture: int = 30
):
    """
    Buat rencana pengumpulan untuk data multi-kelas
    
    Args:
        word_list: Daftar kata untuk dikumpulkan
        alphabet_list: Daftar huruf untuk dikumpulkan
        samples_per_gesture: Sampel per gestur
    """
    
    print("\n" + "="*60)
    print("MULTI-CLASS DATA COLLECTION PLAN")
    print("="*60)
    
    plan = {
        'created': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'words': word_list or [],
        'alphabet': alphabet_list or [],
        'samples_per_gesture': samples_per_gesture,
        'estimated_time': {},
        'progress': {}
    }
    
    # Hitung estimasi
    if word_list:
        num_words = len(word_list)
        time_per_word = 3  # menit (rata-rata)
        total_time_words = num_words * time_per_word
        plan['estimated_time']['words'] = f"{total_time_words} minutes ({total_time_words/60:.1f} hours)"
        plan['progress']['words'] = {word: 0 for word in word_list}
    
    if alphabet_list:
        num_letters = len(alphabet_list)
        time_per_letter = 2  # menit (alfabet lebih cepat)
        total_time_alphabet = num_letters * time_per_letter
        plan['estimated_time']['alphabet'] = f"{total_time_alphabet} minutes ({total_time_alphabet/60:.1f} hours)"
        plan['progress']['alphabet'] = {letter: 0 for letter in alphabet_list}
    
    # Cetak rencana
    print("\n📋 COLLECTION PLAN:")
    
    if word_list:
        print(f"\n🔤 Words ({len(word_list)} gestures):")
        for i, word in enumerate(word_list, 1):
            print(f"   {i:2d}. {word}")
        print(f"\n   Samples per word: {samples_per_gesture}")
        print(f"   Total samples: {len(word_list) * samples_per_gesture}")
        print(f"   Estimated time: {plan['estimated_time']['words']}")
    
    if alphabet_list:
        print(f"\n🔠 Alphabet ({len(alphabet_list)} letters):")
        print(f"   Letters: {', '.join(alphabet_list)}")
        print(f"\n   Samples per letter: {samples_per_gesture}")
        print(f"   Total samples: {len(alphabet_list) * samples_per_gesture}")
        print(f"   Estimated time: {plan['estimated_time']['alphabet']}")
    
    # Total
    total_gestures = len(word_list or []) + len(alphabet_list or [])
    total_samples = total_gestures * samples_per_gesture
    
    print(f"\n📊 TOTAL:")
    print(f"   Gestures: {total_gestures}")
    print(f"   Samples: {total_samples}")
    
    if word_list and alphabet_list:
        total_minutes = (len(word_list) * 3) + (len(alphabet_list) * 2)
        print(f"   Estimated time: {total_minutes} minutes ({total_minutes/60:.1f} hours)")
    
    # Simpan rencana
    plan_path = 'data_collection/collection_plan.json'
    with open(plan_path, 'w') as f:
        json.dump(plan, f, indent=2)
    
    print(f"\n✅ Plan saved: {plan_path}")
    print("\n" + "="*60)
    
    return plan


def show_collection_guide():
    """Tampilkan panduan untuk pengumpulan multi-kelas yang efisien"""
    
    print("\n" + "="*60)
    print("MULTI-CLASS COLLECTION GUIDE")
    print("="*60)
    
    print("""
📚 BEST PRACTICES:

1. PLANNING (Before collection)
   ✅ List all gestures you need
   ✅ Prioritize important gestures first
   ✅ Estimate total time needed
   ✅ Schedule collection sessions

2. ENVIRONMENT SETUP
   ✅ Good lighting (natural or bright)
   ✅ Plain background (wall, curtain)
   ✅ Stable camera position
   ✅ Clear space for hand movement

3. COLLECTION STRATEGY
   ✅ Start with easy gestures (warm-up)
   ✅ Collect similar gestures in batches
   ✅ Take breaks every 30 minutes
   ✅ Check quality after each gesture

4. QUALITY CHECKS
   ✅ All 30 samples collected per gesture
   ✅ Hands clearly visible
   ✅ Consistent gesture execution
   ✅ No duplicate/corrupted files

5. SESSIONS (Recommended)
   Session 1 (1 hour):  5-10 gestures
   Break (15 min)
   Session 2 (1 hour):  5-10 gestures
   Break (15 min)
   Session 3 (1 hour):  5-10 gestures
   
   Total: 15-30 gestures per day

6. TRACKING PROGRESS
   ✅ Use collection plan file
   ✅ Update after each gesture
   ✅ Check dataset folder structure
   ✅ Run verification script

""")
    
    print("="*60)


def verify_collection_progress(dataset_dir='dataset'):
    """Verifikasi progres pengumpulan"""
    
    print("\n" + "="*60)
    print("COLLECTION PROGRESS VERIFICATION")
    print("="*60)
    
    # Cek kata-kata
    words_dir = os.path.join(dataset_dir, 'words')
    if os.path.exists(words_dir):
        print("\n🔤 WORDS:")
        word_folders = [f for f in os.listdir(words_dir) if os.path.isdir(os.path.join(words_dir, f))]
        
        for word in sorted(word_folders):
            word_path = os.path.join(words_dir, word)
            files = [f for f in os.listdir(word_path) if f.endswith('.json')]
            count = len(files)
            
            status = "✅" if count >= 30 else "⚠️" if count >= 20 else "❌"
            print(f"   {status} {word:20s} {count:3d} samples")
        
        total_word_samples = sum(len([f for f in os.listdir(os.path.join(words_dir, w)) if f.endswith('.json')]) 
                                  for w in word_folders)
        print(f"\n   Total word gestures: {len(word_folders)}")
        print(f"   Total word samples: {total_word_samples}")
    
    # Cek alfabet
    alphabet_dir = os.path.join(dataset_dir, 'alphabet')
    if os.path.exists(alphabet_dir):
        print("\n🔠 ALPHABET:")
        letter_folders = [f for f in os.listdir(alphabet_dir) if os.path.isdir(os.path.join(alphabet_dir, f))]
        
        for letter in sorted(letter_folders):
            letter_path = os.path.join(alphabet_dir, letter)
            files = [f for f in os.listdir(letter_path) if f.endswith('.json')]
            count = len(files)
            
            status = "✅" if count >= 30 else "⚠️" if count >= 20 else "❌"
            print(f"   {status} {letter:5s} {count:3d} samples")
        
        total_alphabet_samples = sum(len([f for f in os.listdir(os.path.join(alphabet_dir, l)) if f.endswith('.json')]) 
                                      for l in letter_folders)
        print(f"\n   Total alphabet gestures: {len(letter_folders)}")
        print(f"   Total alphabet samples: {total_alphabet_samples}")
    
    print("\n" + "="*60)
    print("\n💡 Recommendations:")
    print("   - ✅ 30+ samples: Ready to train")
    print("   - ⚠️  20-29 samples: Need more data")
    print("   - ❌ <20 samples: Insufficient data")
    print("\n" + "="*60)


def show_menu():
    """Menu interaktif untuk pengumpulan batch"""
    
    print("\n" + "="*60)
    print("MULTI-CLASS DATA COLLECTION HELPER")
    print("="*60)
    
    while True:
        print("\n📋 MENU:")
        print("   1. Create collection plan")
        print("   2. Show collection guide")
        print("   3. Verify progress")
        print("   4. Start collecting words")
        print("   5. Start collecting alphabet")
        print("   6. Exit")
        
        choice = input("\n👉 Select option (1-6): ").strip()
        
        if choice == '1':
            # Buat rencana
            print("\n📝 Define your collection plan:")
            
            # Kata-kata
            print("\n🔤 Words (comma-separated, or press Enter to skip):")
            print("   Example: halo,terimakasih,tolong,maaf,permisi")
            word_input = input("👉 ").strip()
            word_list = [w.strip() for w in word_input.split(',')] if word_input else []
            
            # Alfabet
            print("\n🔠 Alphabet letters (comma-separated, or press Enter for all A-Z):")
            print("   Example: A,B,C,D,E")
            alphabet_input = input("👉 ").strip()
            
            if alphabet_input:
                alphabet_list = [l.strip().upper() for l in alphabet_input.split(',')]
            else:
                use_all = input("   Use all A-Z? (y/n): ").strip().lower()
                alphabet_list = [chr(i) for i in range(65, 91)] if use_all == 'y' else []
            
            # Sampel
            samples_input = input("\n📊 Samples per gesture (default 30): ").strip()
            samples = int(samples_input) if samples_input else 30
            
            create_collection_plan(word_list, alphabet_list, samples)
        
        elif choice == '2':
            show_collection_guide()
        
        elif choice == '3':
            verify_collection_progress()
        
        elif choice == '4':
            print("\n🚀 Starting word collection...")
            print("   Run: python data_collection/collect_words.py")
            os.system('python -m data_collection.collect_words')
        
        elif choice == '5':
            print("\n🚀 Starting alphabet collection...")
            print("   Run: python data_collection/collect_alphabet.py")
            os.system('python -m data_collection.collect_alphabet')
        
        elif choice == '6':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("\n❌ Invalid option!")


if __name__ == '__main__':
    show_menu()
