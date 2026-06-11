import os
import json
import csv
from datetime import datetime


def create_dataset_structure():
    """Buat struktur dataset lengkap"""
    
    print("🚀 Memulai pembuatan struktur dataset...\n")
    
    # Muat konfigurasi
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # ========================================
    # 1. BUAT DIREKTORI UTAMA
    # ========================================
    print("📁 Membuat direktori utama...")
    directories = [
        'dataset',
        'dataset/words',
        'dataset/alphabet',
        'data_collection',
        'preprocessing',
        'models',
        'training',
        'inference',
        'app',
        'app/resources',
        'trained_models',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✅ {directory}")
    
    # ========================================
    # 2. CATATAN: FOLDER KELAS DIBUAT SECARA DINAMIS
    # ========================================
    print("\n📝 Struktur folder class:")
    print("   ℹ️  Folder untuk kata/huruf akan dibuat otomatis saat collection")
    print("   ℹ️  Ini memungkinkan vocabulary dinamis tanpa perlu setup ulang")
    print("   ✅ dataset/words/ (ready)")
    print("   ✅ dataset/alphabet/ (ready)")
    
    # Muat kosakata dari konfigurasi (untuk README)
    words = config['vocabulary']['words']
    alphabet = config['vocabulary']['alphabet']
    
    # ========================================
    # 4. BUAT participants.csv
    # ========================================
    print("\n👥 Membuat participants.csv...")
    participants_csv = 'dataset/participants.csv'
    
    if not os.path.exists(participants_csv):
        with open(participants_csv, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['participant_id', 'session_id', 'usia', 'jenis_kelamin', 
                          'tangan_dominan', 'keterangan', 'tanggal']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        print(f"   ✅ {participants_csv} (dengan header)")
    else:
        print(f"   ⚠️  {participants_csv} sudah ada, skip...")
    
    # ========================================
    # 5. BUAT manifest.csv
    # ========================================
    print("\n📋 Membuat manifest.csv...")
    manifest_csv = 'dataset/manifest.csv'
    
    if not os.path.exists(manifest_csv):
        with open(manifest_csv, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['file_id', 'participant_id', 'session_id', 'category', 
                          'label', 'frames', 'file_path', 'timestamp']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        print(f"   ✅ {manifest_csv} (dengan header)")
    else:
        print(f"   ⚠️  {manifest_csv} sudah ada, skip...")
    
    # ========================================
    # 6. BUAT FILE README
    # ========================================
    print("\n📖 Membuat README files...")
    
    # README untuk dataset/words
    words_readme = os.path.join('dataset', 'words', 'README.md')
    with open(words_readme, 'w', encoding='utf-8') as f:
        f.write("""# Dataset Words (Kata Lengkap)

## 📁 Struktur Folder (Dynamic)
Folder untuk setiap kata **dibuat otomatis** saat data collection pertama kali.

Contoh struktur:
```
dataset/words/
  ├── halo/           (dibuat saat collect "halo" pertama kali)
  ├── terimakasih/    (dibuat saat collect "terimakasih" pertama kali)
  └── permisi/        (bisa tambah kata baru kapan saja!)
```

## 📝 Format File
- **Nama file**: `{participant_id}_{label}_{number}.json`
- **Contoh**: `001_halo_001.json`, `001_halo_002.json`

## 📊 Format JSON (Two-Hands)
```json
{
  "landmarks": [
    {
      "right": [{"x": 0.5, "y": 0.3, "z": -0.1}, ...],
      "left": [{"x": 0.2, "y": 0.4, "z": 0.0}, ...]
    }
  ],
  "metadata": {
    "participant_id": "001",
    "session_id": "session_001",
    "label": "halo",
    "category": "words",
    "format": "both_hands",
    "features_per_frame": 126
  },
  "timestamp": "2025-11-08T15:41:23"
}
```

## ✅ Vocabulary Dinamis
Anda bisa menambah kata baru:
1. Edit `config.json` → tambah kata di vocabulary
2. Atau langsung collect dengan kata baru
3. Folder akan dibuat otomatis!

## 📝 Kata Default di Config
""")
        for word in words:
            f.write(f"- {word}\n")
        f.write("""
## 🚀 Cara Menambah Kata Baru
1. Edit `config.json`:
   ```json
   "vocabulary": {
     "words": ["halo", "terimakasih", "namasaya", "kata_baru"]
   }
   ```
2. Jalankan collection: `python run_collect_words.py`
3. Folder `kata_baru/` akan dibuat otomatis!
""")
    
    print(f"   ✅ {words_readme}")
    
    # README untuk dataset/alphabet
    alphabet_readme = os.path.join('dataset', 'alphabet', 'README.md')
    with open(alphabet_readme, 'w', encoding='utf-8') as f:
        f.write("""# Dataset Alphabet (Abjad A-Z)

## 📁 Struktur Folder (Dynamic)
Folder untuk setiap huruf **dibuat otomatis** saat data collection pertama kali.

Contoh struktur:
```
dataset/alphabet/
  ├── A/    (dibuat saat collect "A" pertama kali)
  ├── B/    (dibuat saat collect "B" pertama kali)
  ├── C/
  └── ...   (up to Z)
```

## 📝 Format File
- **Nama file**: `{participant_id}_{letter}_{number}.json`
- **Contoh**: `001_A_001.json`, `001_A_002.json`

## 📊 Format JSON (Two-Hands)
```json
{
  "landmarks": [
    {
      "frame": 0,
      "hand_landmarks": [
        {"x": 0.5, "y": 0.3, "z": -0.1},
        ...
      ]
    }
  ],
  "metadata": {
    "participant_id": "subj01",
    "session_id": "session_001",
    "label": "A",
    "category": "alphabet"
  },
  "timestamp": "2025-11-08T15:41:23"
}
```

## Huruf yang Tersedia
A-Z (26 huruf)
""")
    
    print(f"   ✅ {alphabet_readme}")
    
    # ========================================
    # 7. BUAT FILE .gitkeep
    # ========================================
    print("\n📌 Membuat .gitkeep untuk folder kosong...")
    gitkeep_dirs = [
        'trained_models',
        'logs',
        'app/resources'
    ]
    
    for directory in gitkeep_dirs:
        gitkeep_path = os.path.join(directory, '.gitkeep')
        with open(gitkeep_path, 'w') as f:
            f.write('')
        print(f"   ✅ {gitkeep_path}")
    
    # ========================================
    # 8. CETAK RINGKASAN
    # ========================================
    print("\n" + "="*60)
    print("✅ STRUKTUR DATASET BERHASIL DIBUAT!")
    print("="*60)
    print(f"\n📊 Ringkasan:")
    print(f"   - Jumlah kata: {len(words)}")
    print(f"   - Jumlah huruf: {len(alphabet)}")
    print(f"   - Total direktori gestures: {len(words) + len(alphabet)}")
    print(f"   - participants.csv: ✅")
    print(f"   - manifest.csv: ✅")
    
    print(f"\n📁 Struktur lengkap:")
    print_tree('dataset', prefix='')
    
    print("\n🎯 Langkah selanjutnya:")
    print("   1. Jalankan: python run_collect_words.py")
    print("   2. Atau: python run_collect_alphabet.py")
    print("   3. Mulai kumpulkan data dari partisipan!")
    print("\n")


def print_tree(directory, prefix='', max_depth=3, current_depth=0):
    """Cetak struktur pohon direktori"""
    if current_depth >= max_depth:
        return
    
    try:
        items = sorted(os.listdir(directory))
        dirs = [item for item in items if os.path.isdir(os.path.join(directory, item)) and not item.startswith('.')]
        files = [item for item in items if os.path.isfile(os.path.join(directory, item)) and not item.startswith('.')]
        
        # Cetak direktori terlebih dahulu
        for i, item in enumerate(dirs):
            is_last = (i == len(dirs) - 1) and len(files) == 0
            print(f"{prefix}{'└── ' if is_last else '├── '}{item}/")
            
            new_prefix = prefix + ('    ' if is_last else '│   ')
            item_path = os.path.join(directory, item)
            
            # Tampilkan jumlah untuk folder kata/alfabet
            if directory.endswith('words') or directory.endswith('alphabet'):
                count = len([f for f in os.listdir(item_path) if f.endswith('.json')])
                if count > 0:
                    print(f"{new_prefix}└── ({count} files)")
            else:
                print_tree(item_path, new_prefix, max_depth, current_depth + 1)
        
        # Cetak file
        for i, item in enumerate(files):
            is_last = i == len(files) - 1
            print(f"{prefix}{'└── ' if is_last else '├── '}{item}")
            
    except PermissionError:
        pass


if __name__ == '__main__':
    create_dataset_structure()
