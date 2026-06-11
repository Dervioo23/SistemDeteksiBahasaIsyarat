import os
import csv
from datetime import datetime
from typing import Dict, List, Optional
from .utils import (
    validate_participant_id,
    validate_age,
    validate_gender,
    validate_hand_dominance
)


class ParticipantManager:
    """Kelas untuk mengelola data partisipan"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.participants = self._load_participants(csv_path)
    
    def _load_participants(self, csv_path: str) -> List[Dict]:
        """Muat data partisipan dari CSV"""
        if not os.path.exists(csv_path):
            return []
        
        participants = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                participants.append(row)
        return participants
    
    def _save_participant(self, csv_path: str, participant_data: Dict) -> None:
        """Simpan data partisipan ke CSV"""
        file_exists = os.path.exists(csv_path)
        
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            fieldnames = ['participant_id', 'session_id', 'age', 'gender', 
                          'dominant_hand', 'notes', 'date']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(participant_data)
        
        print(f"✅ Data participant '{participant_data['participant_id']}' berhasil disimpan!")
    
    def _get_next_session_id(self, participants: List[Dict], participant_id: str) -> str:
        """Hasilkan ID sesi berikutnya untuk partisipan"""
        sessions = [p['session_id'] for p in participants if p['participant_id'] == participant_id]
        if not sessions:
            return 'session_001'
        
        # Ekstrak angka dari sesi terakhir
        last_session = max(sessions)
        session_num = int(last_session.split('_')[1]) + 1
        return f'session_{session_num:03d}'
    
    def _get_participant_by_id(self, participants: List[Dict], participant_id: str) -> Optional[Dict]:
        """Cari partisipan berdasarkan ID"""
        for p in participants:
            if p['participant_id'] == participant_id:
                return p
        return None
    
    def _update_participant_data(self, csv_path: str, participant_id: str, updated_data: Dict) -> None:
        """Perbarui data partisipan di CSV"""
        participants = self._load_participants(csv_path)
        
        # Perbarui semua entri dengan participant_id yang sama
        for p in participants:
            if p['participant_id'] == participant_id:
                for key, value in updated_data.items():
                    if key not in ['participant_id', 'session_id', 'date']:
                        p[key] = value
        
        # Tulis ulang CSV
        if participants:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['participant_id', 'session_id', 'age', 'gender', 
                              'dominant_hand', 'notes', 'date']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(participants)
            
            print(f"✅ Data participant '{participant_id}' berhasil diupdate!")
    
    def _print_participant_summary(self, participant_data: Dict) -> None:
        """Cetak ringkasan data partisipan"""
        print("\n" + "="*60)
        print("           DATA PARTISIPAN")
        print("="*60)
        print(f"  ID              : {participant_data['participant_id']}")
        print(f"  Session         : {participant_data['session_id']}")
        print(f"  Usia            : {participant_data['age']} tahun")
        print(f"  Jenis Kelamin   : {participant_data['gender']}")
        print(f"  Tangan Dominan  : {participant_data['dominant_hand']}")
        print(f"  Keterangan      : {participant_data['notes']}")
        print("="*60)
    
    def reload(self):
        """Muat ulang data dari CSV"""
        self.participants = self._load_participants(self.csv_path)
    
    def get_all_participants(self) -> List[Dict]:
        """Dapatkan semua partisipan"""
        return self.participants
    
    def get_unique_ids(self) -> List[str]:
        """Dapatkan daftar ID partisipan unik"""
        return list(set([p['participant_id'] for p in self.participants]))
    
    def participant_exists(self, participant_id: str) -> bool:
        """Cek apakah ID partisipan sudah ada"""
        return participant_id in self.get_unique_ids()
    
    def add_new_participant(self) -> Optional[Dict]:
        """Input data partisipan baru dengan validasi"""
        print("\n" + "="*60)
        print("           TAMBAH PARTISIPAN BARU")
        print("="*60)
        
        # Input Participant ID dengan validasi
        while True:
            participant_id = input("\n  Masukkan Participant ID: ").strip()
            
            if not participant_id:
                print("  ❌ Participant ID tidak boleh kosong!")
                continue
            
            if not validate_participant_id(participant_id):
                print("  ❌ Format tidak valid! Gunakan huruf dan angka SAJA (tanpa spasi/underscore, 3-20 karakter)")
                continue
            
            if self.participant_exists(participant_id):
                print(f"  ⚠️  Participant ID '{participant_id}' sudah ada!")
                use_existing = input("  Lanjut session lama? (Y/N): ").strip().upper()
                if use_existing == 'Y':
                    return self.continue_session(participant_id)
                else:
                    continue
            
            break
        
        # Input Usia dengan validasi
        while True:
            usia_input = input("  Usia: ").strip()
            usia = validate_age(usia_input)
            
            if usia is None:
                print("  ❌ Usia harus angka antara 10-100!")
                continue
            
            break
        
        # Input Jenis Kelamin dengan validasi
        while True:
            gender_input = input("  Jenis Kelamin (L/P): ").strip()
            jenis_kelamin = validate_gender(gender_input)
            
            if jenis_kelamin is None:
                print("  ❌ Input tidak valid! Masukkan L (Laki-laki) atau P (Perempuan)")
                continue
            
            break
        
        # Input Tangan Dominan dengan validasi
        while True:
            hand_input = input("  Tangan Dominan (Kanan/Kiri): ").strip()
            tangan_dominan = validate_hand_dominance(hand_input)
            
            if tangan_dominan is None:
                print("  ❌ Input tidak valid! Masukkan 'Kanan' atau 'Kiri'")
                continue
            
            break
        
        # Input Keterangan (opsional)
        keterangan = input("  Keterangan: ").strip()
        if not keterangan:
            keterangan = "-"
        
        # Generate session ID
        session_id = 'session_001'
        
        # Buat data partisipan
        participant_data = {
            'participant_id': participant_id,
            'session_id': session_id,
            'age': str(usia),
            'gender': jenis_kelamin,
            'dominant_hand': tangan_dominan,
            'notes': keterangan,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Konfirmasi
        self._print_participant_summary(participant_data)
        confirm = input("\n  Simpan data ini? (Y/N): ").strip().upper()
        
        if confirm == 'Y':
            self._save_participant(self.csv_path, participant_data)
            self.reload()
            return participant_data
        else:
            print("  ❌ Data tidak disimpan.")
            return None
    
    def continue_session(self, participant_id: str) -> Optional[Dict]:
        """Lanjutkan sesi untuk partisipan yang sudah ada"""
        self.reload()
        
        if not self.participant_exists(participant_id):
            print(f"  ❌ Participant ID '{participant_id}' tidak ditemukan!")
            return None
        
        # Dapatkan data terakhir partisipan
        existing = self._get_participant_by_id(self.participants, participant_id)
        
        # Hasilkan ID sesi baru
        session_id = self._get_next_session_id(self.participants, participant_id)
        
        # Buat data sesi baru dengan data yang sama
        participant_data = {
            'participant_id': participant_id,
            'session_id': session_id,
            'age': existing['age'],
            'gender': existing['gender'],
            'dominant_hand': existing['dominant_hand'],
            'notes': existing['notes'],
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"\n  ✅ Melanjutkan session baru untuk '{participant_id}'")
        self._print_participant_summary(participant_data)
        
        self._save_participant(self.csv_path, participant_data)
        self.reload()
        return participant_data
    
    def edit_participant(self, participant_id: str) -> bool:
        """Edit data partisipan"""
        self.reload()
        
        if not self.participant_exists(participant_id):
            print(f"  ❌ Participant ID '{participant_id}' tidak ditemukan!")
            return False
        
        existing = self._get_participant_by_id(self.participants, participant_id)
        
        print("\n" + "="*60)
        print(f"         EDIT DATA: {participant_id}")
        print("="*60)
        print(f"  Data saat ini:")
        print(f"    Usia           : {existing['age']}")
        print(f"    Jenis Kelamin  : {existing['gender']}")
        print(f"    Tangan Dominan : {existing['dominant_hand']}")
        print(f"    Keterangan     : {existing['notes']}")
        print("\n  Tekan ENTER untuk skip (tidak mengubah)")
        print("="*60)
        
        updated_data = {}
        
        # Edit Usia
        usia_input = input(f"\n  Usia baru [{existing['age']}]: ").strip()
        if usia_input:
            usia = validate_age(usia_input)
            if usia:
                updated_data['age'] = str(usia)
            else:
                print("  ⚠️  Usia tidak valid, menggunakan nilai lama")
        
        # Edit Jenis Kelamin
        gender_input = input(f"  Jenis Kelamin baru [{existing['gender']}]: ").strip()
        if gender_input:
            jenis_kelamin = validate_gender(gender_input)
            if jenis_kelamin:
                updated_data['gender'] = jenis_kelamin
            else:
                print("  ⚠️  Input tidak valid, menggunakan nilai lama")
        
        # Edit Tangan Dominan
        hand_input = input(f"  Tangan Dominan baru [{existing['dominant_hand']}]: ").strip()
        if hand_input:
            tangan_dominan = validate_hand_dominance(hand_input)
            if tangan_dominan:
                updated_data['dominant_hand'] = tangan_dominan
            else:
                print("  ⚠️  Input tidak valid, menggunakan nilai lama")
        
        # Edit Keterangan
        keterangan_input = input(f"  Keterangan baru [{existing['notes']}]: ").strip()
        if keterangan_input:
            updated_data['notes'] = keterangan_input
        
        if updated_data:
            self._update_participant_data(self.csv_path, participant_id, updated_data)
            self.reload()
            return True
        else:
            print("  ℹ️  Tidak ada perubahan data")
            return False
    
    def show_menu(self) -> Optional[Dict]:
        """Tampilkan menu interaktif untuk mengelola partisipan"""
        while True:
            print("\n" + "="*60)
            print("           PARTICIPANT MANAGER")
            print("="*60)
            print("  [1] Tambah Participant Baru")
            print("  [2] Lanjut Session Lama")
            print("  [3] Edit Data Participant")
            print("  [4] Lihat Daftar Participant")
            print("  [0] Lanjut ke Pengumpulan Data")
            print("="*60)
            
            choice = input("\n  Pilih menu: ").strip()
            
            if choice == '1':
                result = self.add_new_participant()
                if result:
                    return result
            
            elif choice == '2':
                unique_ids = self.get_unique_ids()
                if not unique_ids:
                    print("  ❌ Belum ada participant yang terdaftar!")
                    continue
                
                print("\n  Daftar Participant ID:")
                for idx, pid in enumerate(unique_ids, 1):
                    print(f"    {idx}. {pid}")
                
                pid_input = input("\n  Masukkan Participant ID: ").strip()
                result = self.continue_session(pid_input)
                if result:
                    return result
            
            elif choice == '3':
                unique_ids = self.get_unique_ids()
                if not unique_ids:
                    print("  ❌ Belum ada participant yang terdaftar!")
                    continue
                
                print("\n  Daftar Participant ID:")
                for idx, pid in enumerate(unique_ids, 1):
                    print(f"    {idx}. {pid}")
                
                pid_input = input("\n  Masukkan Participant ID untuk diedit: ").strip()
                self.edit_participant(pid_input)
            
            elif choice == '4':
                self.show_all_participants()
            
            elif choice == '0':
                # User ingin lanjut, pastikan ada participant aktif
                if not self.participants:
                    print("  ❌ Belum ada participant! Tambahkan dulu.")
                    continue
                
                # Gunakan participant terakhir atau biarkan user pilih
                print("\n  Gunakan participant terakhir atau pilih yang lain?")
                print("  [1] Gunakan terakhir")
                print("  [2] Pilih manual")
                
                sub_choice = input("\n  Pilih: ").strip()
                
                if sub_choice == '1':
                    # Gunakan yang terakhir ditambahkan
                    return self.participants[-1]
                else:
                    unique_ids = self.get_unique_ids()
                    print("\n  Daftar Participant ID:")
                    for idx, pid in enumerate(unique_ids, 1):
                        print(f"    {idx}. {pid}")
                    
                    pid_input = input("\n  Pilih Participant ID: ").strip()
                    participant = self._get_participant_by_id(self.participants, pid_input)
                    if participant:
                        return participant
                    else:
                        print("  ❌ Participant tidak ditemukan!")
            
            else:
                print("  ❌ Pilihan tidak valid!")
    
    def show_all_participants(self):
        """Tampilkan semua partisipan"""
        self.reload()
        
        if not self.participants:
            print("\n  ℹ️  Belum ada participant yang terdaftar")
            return
        
        print("\n" + "="*80)
        print("                        DAFTAR PARTISIPAN")
        print("="*80)
        
        # Kelompokkan berdasarkan participant_id
        grouped = {}
        for p in self.participants:
            pid = p['participant_id']
            if pid not in grouped:
                grouped[pid] = []
            grouped[pid].append(p)
        
        for pid, sessions in grouped.items():
            print(f"\n  📌 {pid}")
            print(f"     Usia: {sessions[0]['age']}, "
                  f"Gender: {sessions[0]['gender']}, "
                  f"Hand: {sessions[0]['dominant_hand']}")
            print(f"     Keterangan: {sessions[0]['notes']}")
            print(f"     Total sessions: {len(sessions)}")
            for session in sessions:
                print(f"       - {session['session_id']} ({session['date']})")
        
        print("="*80)
        input("\n  Tekan ENTER untuk kembali...")
