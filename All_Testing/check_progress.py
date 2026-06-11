import subprocess
import sys
import os

def main():
    print("\n" + "="*70)
    print("  QUICK PROGRESS CHECK")
    print("="*70)
    
    print("\n  Pilih mode:")
    print("  1. Lihat ringkasan semua partisipan")
    print("  2. Lihat detail partisipan tertentu")
    print("  3. Lihat detail semua partisipan")
    print("  0. Keluar")
    
    choice = input("\n  Pilih (1/2/3/0): ").strip()
    
    if choice == '1':
        # Ringkasan
        subprocess.run([sys.executable, 'view_collection_progress.py'])
    
    elif choice == '2':
        # Partisipan tertentu
        participant_id = input("\n  Masukkan Participant ID (contoh: 001): ").strip()
        if participant_id:
            subprocess.run([sys.executable, 'view_collection_progress.py', 
                          '--participant', participant_id])
        else:
            print("  ⚠️  Participant ID tidak boleh kosong!")
    
    elif choice == '3':
        # Semua detail
        subprocess.run([sys.executable, 'view_collection_progress.py', '--all'])
    
    elif choice == '0':
        print("\n  👋 Bye!")
        return
    
    else:
        print("\n  ⚠️  Pilihan tidak valid!")

if __name__ == '__main__':
    main()
