import time
from typing import List, Optional, Tuple
from collections import deque
from data_collection.utils import load_config


class SequenceHandler:
    """Handler untuk membangun urutan dari huruf yang terdeteksi"""
    
    def __init__(
        self,
        max_sequence_length: int = 20,
        letter_timeout: float = 3.0,
        word_complete_timeout: float = 2.0
    ):
        """
        Inisialisasi handler urutan
        
        Args:
            max_sequence_length: Maksimum huruf dalam urutan
            letter_timeout: Detik sebelum urutan direset
            word_complete_timeout: Detik menunggu sebelum menyelesaikan kata
        """
        self.max_sequence_length = max_sequence_length
        self.letter_timeout = letter_timeout
        self.word_complete_timeout = word_complete_timeout
        
        # Urutan saat ini
        self.current_sequence = []
        self.last_letter_time = 0
        self.sequence_start_time = 0
        
        # Kata yang selesai
        self.completed_words = []
        
        # Konfigurasi autocorrect dan kosakata
        config = {}
        try:
            config = load_config()
        except Exception:
            config = {}
        vocab_cfg = config.get("vocabulary", {})
        words_cfg = vocab_cfg.get("words") or []
        self.vocabulary_words = [str(w).lower() for w in words_cfg]
        inference_cfg = config.get("inference", {})
        autocorrect_cfg = inference_cfg.get("autocorrect", {})
        self.enable_autocorrect = bool(autocorrect_cfg.get("enabled", True))
        self.autocorrect_max_distance = int(autocorrect_cfg.get("max_distance", 2))
    
    def add_letter(self, letter: str) -> bool:
        """
        Tambahkan huruf ke urutan saat ini
        
        Args:
            letter: Huruf yang terdeteksi
            
        Returns:
            True jika huruf ditambahkan, False jika ditolak (duplikat, dll.)
        """
        current_time = time.time()
        
        # Mulai urutan baru jika timeout
        if self.is_timed_out():
            self.reset_sequence()
        
        # Mulai urutan jika kosong
        if not self.current_sequence:
            self.sequence_start_time = current_time
        
        # Cek duplikat (tolak duplikat cepat) - Increased from 0.5s to 1.0s
        if self.current_sequence and self.current_sequence[-1] == letter:
            # Izinkan jika cukup waktu berlalu (1 detik untuk menghindari spam)
            if (current_time - self.last_letter_time) < 1.0:
                return False
        
        # Cek panjang maksimum
        if len(self.current_sequence) >= self.max_sequence_length:
            return False
        
        # Tambahkan huruf
        self.current_sequence.append(letter)
        self.last_letter_time = current_time
        
        return True
    
    def get_current_word(self) -> str:
        """Dapatkan kata saat ini yang sedang dibangun"""
        return ''.join(self.current_sequence)
    
    def is_timed_out(self) -> bool:
        """Cek jika urutan sudah timeout"""
        if not self.current_sequence:
            return False
        
        current_time = time.time()
        return (current_time - self.last_letter_time) > self.letter_timeout
    
    def is_word_complete(self) -> bool:
        """
        Cek jika kata sudah selesai (siap untuk output)
        Pengguna berhenti mengeja selama word_complete_timeout detik
        """
        if not self.current_sequence:
            return False
        
        current_time = time.time()
        time_since_last = current_time - self.last_letter_time
        
        return time_since_last >= self.word_complete_timeout
    
    def complete_word(self) -> Optional[str]:
        """
        Selesaikan kata saat ini dan reset
        
        Returns:
            Kata yang selesai atau None jika kosong
        """
        if not self.current_sequence:
            return None
        
        raw_word = self.get_current_word()
        corrected_word = self._autocorrect_word(raw_word)
        self.completed_words.append({
            'word': corrected_word,
            'length': len(corrected_word),
            'time': time.time(),
            'raw_word': raw_word
        })
        
        self.reset_sequence()
        return corrected_word

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        if s1 == s2:
            return 0
        if not s1:
            return len(s2)
        if not s2:
            return len(s1)
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1, start=1):
            current_row = [i]
            for j, c2 in enumerate(s2, start=1):
                insert_cost = previous_row[j] + 1
                delete_cost = current_row[j - 1] + 1
                replace_cost = previous_row[j - 1] + (c1 != c2)
                current_row.append(min(insert_cost, delete_cost, replace_cost))
            previous_row = current_row
        return previous_row[-1]

    def _autocorrect_word(self, word: str) -> str:
        if not self.enable_autocorrect or not self.vocabulary_words:
            return word
        word_lower = word.lower()
        if word_lower in self.vocabulary_words:
            return word
        best_candidate = None
        best_distance = None
        for candidate in self.vocabulary_words:
            dist = self._levenshtein_distance(word_lower, candidate)
            if best_distance is None or dist < best_distance:
                best_distance = dist
                best_candidate = candidate
        if best_candidate is None or best_distance is None:
            return word
        if best_distance > self.autocorrect_max_distance:
            return word
        if word.isupper():
            return best_candidate.upper()
        return best_candidate
    
    def reset_sequence(self):
        """Reset urutan saat ini"""
        self.current_sequence = []
        self.last_letter_time = 0
        self.sequence_start_time = 0
    
    def remove_last_letter(self) -> bool:
        """
        Hapus huruf terakhir (backspace)
        
        Returns:
            True jika dihapus, False jika urutan kosong
        """
        if self.current_sequence:
            self.current_sequence.pop()
            return True
        return False
    
    def get_sequence_info(self) -> dict:
        """Dapatkan informasi urutan saat ini"""
        current_time = time.time()
        
        return {
            'current_word': self.get_current_word(),
            'length': len(self.current_sequence),
            'timed_out': self.is_timed_out(),
            'word_complete': self.is_word_complete(),
            'time_since_last': current_time - self.last_letter_time if self.last_letter_time else 0,
            'sequence_duration': current_time - self.sequence_start_time if self.sequence_start_time else 0
        }
    
    def get_completed_words(self) -> List[dict]:
        """Dapatkan daftar kata yang selesai"""
        return self.completed_words
    
    def clear_completed_words(self):
        """Hapus riwayat kata yang selesai"""
        self.completed_words = []


# Contoh penggunaan
if __name__ == '__main__':
    print("\n" + "="*60)
    print("SEQUENCE HANDLER TEST")
    print("="*60 + "\n")
    
    # Inisialisasi handler
    handler = SequenceHandler(
        max_sequence_length=20,
        letter_timeout=3.0,
        word_complete_timeout=2.0
    )
    
    print("✅ Handler initialized")
    print(f"   Max length: {handler.max_sequence_length}")
    print(f"   Letter timeout: {handler.letter_timeout}s")
    print(f"   Word complete timeout: {handler.word_complete_timeout}s")
    
    # Tes pembangunan urutan
    print("\n🧪 Testing sequence building...")
    
    test_letters = ['H', 'A', 'L', 'O']
    
    for letter in test_letters:
        added = handler.add_letter(letter)
        if added:
            print(f"   ✅ Added '{letter}': {handler.get_current_word()}")
            time.sleep(0.6)  # Simulasikan waktu antar huruf
        else:
            print(f"   ❌ Rejected '{letter}'")
    
    # Cek info
    info = handler.get_sequence_info()
    print(f"\n📊 Sequence Info:")
    print(f"   Current word: {info['current_word']}")
    print(f"   Length: {info['length']}")
    print(f"   Time since last: {info['time_since_last']:.2f}s")
    
    # Tunggu penyelesaian kata
    print("\n⏳ Waiting for word completion...")
    time.sleep(2.5)
    
    if handler.is_word_complete():
        word = handler.complete_word()
        print(f"   ✅ Word completed: '{word}'")
    
    # Tes timeout
    print("\n🧪 Testing timeout...")
    handler.add_letter('T')
    print(f"   Current: {handler.get_current_word()}")
    time.sleep(3.5)
    
    if handler.is_timed_out():
        print(f"   ⏰ Sequence timed out")
        handler.reset_sequence()
    
    # Tes backspace
    print("\n🧪 Testing backspace...")
    handler.add_letter('T')
    handler.add_letter('E')
    handler.add_letter('S')
    print(f"   Before: {handler.get_current_word()}")
    handler.remove_last_letter()
    print(f"   After backspace: {handler.get_current_word()}")
    
    print("\n✅ Sequence handler test completed!")
