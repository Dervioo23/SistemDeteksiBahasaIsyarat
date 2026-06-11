import time
from enum import Enum, auto
from typing import List, Optional, Dict, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)

class InputMode(Enum):
    WORD = auto()       # Input kata utuh (Halo, Terimakasih)
    SPELLING = auto()   # Input ejaan (D-E-R-V-I-O)

class SentenceBuilder:
    """
    Engine penyusun kalimat (The Translator - Layer 2)
    
    Tanggung Jawab:
    1. Noise Filtering: Memastikan deteksi stabil sebelum diterima.
    2. Mode Management: Menangani input Kata vs Ejaan.
    3. Sentence Assembly: Menggabungkan komponen menjadi kalimat.
    """
    
    def __init__(
        self,
        min_stability_duration: float = 0.5,  # Detik minimal deteksi stabil (Noise Filter)
        word_sequence_timeout: float = 2.0,    # Waktu untuk menyelesaikan kata saat mengeja
        sentence_timeout: float = 5.0,         # Waktu idle sebelum kalimat dianggap selesai (opsional)
        auto_commit_interval: float = 20.0     # Waktu idle sebelum ejaan otomatis di-commit jadi kata
    ):
        self.min_stability_duration = min_stability_duration
        self.word_sequence_timeout = word_sequence_timeout
        self.auto_commit_interval = auto_commit_interval
        
        # Buffer Utama
        self.sentence_parts: List[str] = []    # ["Halo", "Nama", "Saya", "DERVIO"]
        self.current_spelling: List[str] = []  # ["D", "E", "R", "V", "I", "O"]
        
        # Noise Filtering State
        self.raw_detection_buffer = deque(maxlen=20) # Buffer untuk raw detection
        self.stable_label: Optional[str] = None      # Label yang sedang distabilkan
        self.stability_start_time = 0.0
        self.last_committed_label: Optional[str] = None
        self.last_commit_time = 0.0
        
        # Mode State
        self.current_mode = InputMode.WORD
        
        # Tracking waktu
        self.last_action_time = time.time()
        
    def set_mode(self, mode: InputMode):
        """Ubah mode input"""
        if self.current_mode != mode:
            # Jika pindah DARI spelling, commit apa yang ada di buffer ejaan
            if self.current_mode == InputMode.SPELLING and self.current_spelling:
                self.commit_spelling()
            
            logger.info(f"Mode switched: {self.current_mode.name} -> {mode.name}")
            self.current_mode = mode
            self.last_action_time = time.time()

    def process_detection(self, label: str, confidence: float) -> Tuple[bool, Optional[str]]:
        """
        Proses raw detection dari model.
        
        Returns:
            (committed: bool, committed_content: str)
            committed: Apakah deteksi ini akhirnya diterima (lolos noise filter).
            committed_content: Apa yang ditambahkan ke kalimat (kata atau huruf).
        """
        current_time = time.time()
        
        # --- PHASE 1: NOISE FILTERING ---
        # Logic: Label harus sama terus menerus selama `min_stability_duration`
        
        if label != self.stable_label:
            # Deteksi berubah, reset timer
            self.stable_label = label
            self.stability_start_time = current_time
            return False, None
        
        # Label stabil, cek durasi
        duration = current_time - self.stability_start_time
        
        if duration < self.min_stability_duration:
            # Belum cukup lama
            return False, None
        
        # --- PHASE 2: COMMITMENT ---
        # Sudah stabil cukup lama. 
        # Cek duplicasi (debounce) - jangan commit hal yang sama berulang kali dalam jeda singkat
        # KECUALI jika pengguna melakukan "reset" gesture (tangan hilang/neutral) -> tapi ini handled by `label != unstable` above
        # Kita perlu cooldown sederhana agar "Halo" tidak diketik "Halo Halo Halo" jika user menahan pose 2 detik.
        
        if label == self.last_committed_label:
            # Masih gesture yang sama yang baru saja dicommit
            # Butuh jeda (neutral/gesture lain) atau cooldown panjang untuk re-trigger
            if (current_time - self.last_commit_time) < 2.0: # 2 detik cooldown untuk kata yang SAMA
                 return False, None
        
        # Lolos semua filter -> COMMIT
        self.last_committed_label = label
        self.last_commit_time = current_time
        self.last_action_time = current_time
        
        content_added = self._handle_logic_core(label)
        return True, content_added

    def _handle_logic_core(self, label: str) -> str:
        """Logika penggabungan kalimat berdasarkan mode"""
        
        if self.current_mode == InputMode.WORD:
            # Mode Kata: Langsung tambahkan ke kalimat
            # Jika ada buffer ejaan yang menggantung, selesaikan dulu
            if self.current_spelling:
                self.commit_spelling()
                
            self.sentence_parts.append(label)
            return label
            
        elif self.current_mode == InputMode.SPELLING:
            # Mode Ejaan: Tambahkan ke buffer ejaan
            # Biasanya label alfabet cuma 1 huruf
            if len(label) == 1:
                self.current_spelling.append(label)
                return label
            else:
                # Jika mendeteksi kata (misal 'Halo') saat mode spelling?
                # Opsional: auto-switch atau abaikan.
                # Kita anggap abaikan atau force commit spelling dulu.
                self.commit_spelling()
                self.sentence_parts.append(label)
                return label
                
        return label

    def commit_spelling(self):
        """Gabungkan buffer ejaan menjadi satu kata dan masukkan ke kalimat"""
        if not self.current_spelling:
            return
            
        full_word = "".join(self.current_spelling)
        self.sentence_parts.append(full_word)
        self.current_spelling = []
        logger.info(f"Spelling committed: {full_word}")
    
    def add_word(self, word: str):
        """Tambahkan kata ke kalimat (safe public method)"""
        if word:
            self.sentence_parts.append(word)
            self.last_action_time = time.time()
            logger.info(f"Word added: {word}")

    def get_built_sentence(self) -> str:
        """
        Ambil kalimat lengkap saat ini.
        """
        # Gabungkan bagian kalimat
        raw_sentence = " ".join(self.sentence_parts)
        
        # Tampilkan ejaan yang sedang berjalan (preview)
        if self.current_spelling:
            raw_sentence += " " + "".join(self.current_spelling) + "_"
            
        return raw_sentence.strip()

    def backspace(self):
        """Hapus elemen terakhir"""
        if self.current_mode == InputMode.SPELLING and self.current_spelling:
            self.current_spelling.pop()
        elif self.sentence_parts:
            self.sentence_parts.pop()
            
    def clear(self):
        """Hapus semua"""
        self.sentence_parts = []
        self.current_spelling = []
        self.stable_label = None
        self.last_committed_label = None

    def force_commit_spelling(self):
        """Manual trigger untuk menyelesaikan ejaan (misal tombol SPACE)"""
        self.commit_spelling()

    def check_auto_commit(self) -> Tuple[bool, str]:
        """
        Cek apakah harus auto-commit karena timeout (Logic Phase: End-of-Word Trigger)
        Returns: (triggered, message)
        """
        if self.current_mode == InputMode.SPELLING and self.current_spelling:
            elapsed = time.time() - self.last_action_time
            if elapsed > self.auto_commit_interval:
                word = "".join(self.current_spelling)
                self.commit_spelling()
                return True, f"Auto-committed: {word}"
        return False, ""

    def get_auto_commit_remaining_time(self) -> float:
        """
        Dapatkan sisa waktu sebelum auto-commit (untuk visualisasi UI).
        Returns: seconds remaining (float) or -1 if inactive
        """
        if self.current_mode == InputMode.SPELLING and self.current_spelling:
            elapsed = time.time() - self.last_action_time
            remaining = self.auto_commit_interval - elapsed
            return max(0.0, remaining)
        return -1.0

