import os
import csv
import json
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional


def load_config(config_path: str = "config.json") -> Dict:
    """Muat konfigurasi dari file JSON"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_config(config: Dict, config_path: str = "config.json") -> None:
    """Simpan konfigurasi ke file JSON"""
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def create_directories(config: Dict) -> None:
    """Buat struktur direktori root untuk dataset (folder kelas dibuat secara dinamis)"""
    # Hanya direktori root - folder kelas akan dibuat saat pengumpulan data
    os.makedirs(config['dataset']['words_dir'], exist_ok=True)
    os.makedirs(config['dataset']['alphabet_dir'], exist_ok=True)
    os.makedirs("trained_models", exist_ok=True)

def validate_participant_id(participant_id: str) -> bool:
    """Validasi format ID Partisipan"""
    # Hanya alfanumerik (tanpa garis bawah), minimal 3 karakter
    # PENTING: Jangan gunakan underscore (_) karena dipakai sebagai pemisah di nama file
    pattern = r'^[a-zA-Z0-9]{3,20}$'
    return bool(re.match(pattern, participant_id))


def validate_word(word: str) -> bool:
    """Validasi format kata/label untuk pengumpulan data"""
    # Hanya huruf kecil alfanumerik dan garis bawah, minimal 2 karakter, maks 30
    pattern = r'^[a-z0-9_]{2,30}$'
    return bool(re.match(pattern, word))


def add_word_to_config(word: str, config_path: str = "config.json") -> bool:
    """Tambah kata baru ke config.json secara permanen"""
    try:
        config = load_config(config_path)
        
        # Cek jika kata sudah ada
        if word in config['vocabulary']['words']:
            return False
        
        # Tambahkan kata
        config['vocabulary']['words'].append(word)
        
        # Simpan konfigurasi
        save_config(config, config_path)
        return True
    except Exception as e:
        print(f"Error adding word to config: {e}")
        return False


def get_custom_word_input() -> Tuple[Optional[str], bool]:
    """
    Dapatkan kata kustom dari input pengguna
    Returns: (word, save_to_config)
    """
    print("\n" + "="*60)
    print("  ➕ TAMBAH KATA BARU")
    print("="*60)
    print("\n  📝 Aturan penamaan kata:")
    print("     - Hanya huruf kecil (a-z)")
    print("     - Boleh gunakan angka (0-9)")
    print("     - Boleh gunakan underscore (_)")
    print("     - Minimal 2 karakter, maksimal 30 karakter")
    print("     - Contoh: halo, selamat_pagi, apa_kabar, kata123\n")
    
    while True:
        word = input("  Masukkan kata baru (atau 'batal' untuk kembali): ").strip().lower()
        
        if word == 'batal':
            return None, False
        
        if not word:
            print("  ❌ Kata tidak boleh kosong!")
            continue
        
        if not validate_word(word):
            print("  ❌ Format kata tidak valid! Gunakan huruf kecil, angka, atau underscore saja.")
            continue
        
        # Kata valid
        print(f"\n  ✅ Kata '{word}' valid!")
        
        # Tanya apakah simpan ke config
        while True:
            save_choice = input("\n  Simpan ke config.json? (y/n): ").strip().lower()
            if save_choice in ['y', 'n']:
                save_to_config = (save_choice == 'y')
                break
            print("  ❌ Input 'y' atau 'n'!")
        
        return word, save_to_config


def validate_age(age_str: str) -> Optional[int]:
    """Validasi usia"""
    try:
        age = int(age_str)
        if 10 <= age <= 100:
            return age
        return None
    except ValueError:
        return None


def validate_gender(gender_input: str) -> Optional[str]:
    """Validasi jenis kelamin"""
    gender_map = {
        'l': 'Laki-laki',
        'laki-laki': 'Laki-laki',
        'laki': 'Laki-laki',
        'p': 'Perempuan',
        'perempuan': 'Perempuan'
    }
    return gender_map.get(gender_input.lower().strip())


def validate_hand_dominance(hand_input: str) -> Optional[str]:
    """Validasi tangan dominan"""
    hand_map = {
        'kanan': 'Kanan',
        'r': 'Kanan',
        'right': 'Kanan',
        'kiri': 'Kiri',
        'l': 'Kiri',
        'left': 'Kiri'
    }
    return hand_map.get(hand_input.lower().strip())



def get_next_sample_number(directory: str, participant_id: str, label: str) -> int:
    """Dapatkan nomor sampel berikutnya untuk partisipan dan label tertentu"""
    if not os.path.exists(directory):
        return 1
    
    files = os.listdir(directory)
    prefix = f"{participant_id}_{label}_"
    
    numbers = []
    for f in files:
        if f.startswith(prefix) and f.endswith('.json'):
            try:
                num = int(f.replace(prefix, '').replace('.json', ''))
                numbers.append(num)
            except ValueError:
                continue
    
    return max(numbers) + 1 if numbers else 1


def count_existing_samples(directory: str, participant_id: str, label: str) -> int:
    """Hitung berapa banyak sampel yang sudah ada untuk partisipan dan label tertentu"""
    if not os.path.exists(directory):
        return 0
    
    files = os.listdir(directory)
    prefix = f"{participant_id}_{label}_"
    
    count = 0
    for f in files:
        if f.startswith(prefix) and f.endswith('.json'):
            count += 1
    
    return count


def save_landmarks(landmarks: List[List[Dict]], file_path: str, metadata: Dict = None) -> None:
    """Simpan landmarks ke file JSON"""
    data = {
        'landmarks': landmarks,
        'metadata': metadata or {},
        'timestamp': datetime.now().isoformat()
    }
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def add_to_manifest(manifest_path: str, entry: Dict) -> None:
    """Tambahkan entri ke manifest.csv"""
    file_exists = os.path.exists(manifest_path)
    
    with open(manifest_path, 'a', newline='', encoding='utf-8') as f:
        fieldnames = ['file_id', 'participant_id', 'session_id', 'category', 
                      'label', 'frames', 'file_path', 'timestamp']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(entry)




def print_instructions():
    """Cetak instruksi penggunaan"""
    print("\n" + "="*60)
    print("           PETUNJUK PENGGUNAAN")
    print("="*60)
    print("  1. Posisikan tangan di depan kamera")
    print("  2. Pastikan pencahayaan cukup")
    print("  3. Background kontras dengan tangan")
    print("  4. Jarak optimal: 50-100 cm dari kamera")
    print("\n  Tombol:")
    print("    [SPACE]  - Mulai merekam gestur")
    print("    [Q]      - Keluar")
    print("    [N]      - Gestur berikutnya")
    print("    [P]      - Gestur sebelumnya")
    print("    [C]      - Kata kustom (tambah kata baru)")
    print("    [R]      - Ulangi gestur terakhir")
    print("="*60 + "\n")


def countdown(seconds: int):
    """Hitung mundur"""
    import time
    for i in range(seconds, 0, -1):
        print(f"  Mulai dalam {i}...", end='\r')
        time.sleep(1)
    print("  🎬 PEREKAMAN...     ")
