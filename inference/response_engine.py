import time
from typing import Optional, Dict, List
from datetime import datetime

from data_collection.utils import load_config


class ResponseEngine:
    """Engine untuk respons kontekstual"""
    
    def __init__(self):
        """Inisialisasi engine respons"""
        # Template respons
        self.greeting_responses = [
            "Halo! Selamat datang!",
            "Hai! Apa kabar?",
            "Halo! Senang bertemu Anda!"
        ]
        
        # Inisialisasi respons huruf dari kosakata config untuk konsistensi
        self.letter_responses = self._init_letter_responses_from_config()
        
        # Riwayat interaksi
        self.interaction_history = []
        self.last_response_time = 0

    def _init_letter_responses_from_config(self) -> Dict[str, str]:
        """Bangun peta respons huruf berdasarkan config.vocabulary.alphabet.

        Kembali ke default A–E jika config tidak tersedia
        atau tidak mendefinisikan daftar alfabet.
        """
        # Pemetaan default (disimpan untuk kompatibilitas mundur)
        default_map = {
            'A': "Huruf A",
            'B': "Huruf B",
            'C': "Huruf C",
            'D': "Huruf D",
            'E': "Huruf E"
            # Tambahkan lebih banyak sesuai kebutuhan
        }

        try:
            config = load_config()
            vocab = config.get("vocabulary", {})
            alphabet = vocab.get("alphabet") or []

            if not alphabet:
                return default_map

            # Hasilkan respons "Huruf X" sederhana untuk setiap huruf yang dikonfigurasi
            return {str(letter): f"Huruf {letter}" for letter in alphabet}
        except Exception:
            # Jika ada error, pertahankan perilaku identik dengan peta hard-coded sebelumnya
            return default_map
    
    def process_word(self, word: str) -> str:
        """
        Proses kata yang terdeteksi dan hasilkan respons
        
        Args:
            word: Kata yang terdeteksi
            
        Returns:
            Teks respons
        """
        word_lower = word.lower()
        
        # Rekam interaksi
        self._record_interaction('word', word)
        
        # Hasilkan respons berdasarkan kata
        if word_lower == 'halo':
            response = self._get_greeting_response()
        else:
            response = f"Terdeteksi: {word}"
        
        return response
    
    def process_letter(self, letter: str) -> str:
        """
        Proses huruf yang terdeteksi
        
        Args:
            letter: Huruf yang terdeteksi
            
        Returns:
            Teks respons
        """
        # Rekam interaksi
        self._record_interaction('letter', letter)
        
        # Dapatkan respons
        if letter in self.letter_responses:
            response = self.letter_responses[letter]
        else:
            response = f"Huruf {letter}"
        
        return response
    
    def process_spelled_word(self, word: str) -> str:
        """
        Proses kata yang dieja huruf demi huruf.
        Mengembalikan kata yang sudah diproses (cleaned).
        
        Args:
            word: Kata yang selesai dieja
            
        Returns:
            Kata yang sudah diproses (cleaned/normalized)
        """
        # Rekam interaksi
        self._record_interaction('spelled_word', word)
        
        # Normalisasi kata - capitalize first letter
        processed = word.strip()
        if processed:
            processed = processed.capitalize()
        
        return processed
    
    def get_response_for_spelled_word(self, word: str) -> str:
        """
        Dapatkan respons TTS untuk kata yang dieja
        
        Args:
            word: Kata yang selesai dieja
            
        Returns:
            Teks respons untuk TTS
        """
        return f"Kata yang dieja: {word}"
    
    def _get_greeting_response(self) -> str:
        """Dapatkan respons salam (berputar melalui opsi)"""
        import random
        return random.choice(self.greeting_responses)
    
    def _record_interaction(self, interaction_type: str, content: str):
        """Rekam interaksi dalam riwayat"""
        self.interaction_history.append({
            'type': interaction_type,
            'content': content,
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat()
        })
        
        # Simpan 100 interaksi terakhir
        if len(self.interaction_history) > 100:
            self.interaction_history = self.interaction_history[-100:]
    
    def get_interaction_history(self, limit: int = 10) -> List[Dict]:
        """
        Dapatkan riwayat interaksi terbaru
        
        Args:
            limit: Jumlah interaksi terbaru
            
        Returns:
            Daftar interaksi
        """
        return self.interaction_history[-limit:]
    
    def get_statistics(self) -> Dict:
        """Dapatkan statistik interaksi"""
        if not self.interaction_history:
            return {
                'total_interactions': 0,
                'word_detections': 0,
                'letter_detections': 0,
                'spelled_words': 0
            }
        
        word_count = sum(1 for i in self.interaction_history if i['type'] == 'word')
        letter_count = sum(1 for i in self.interaction_history if i['type'] == 'letter')
        spelled_count = sum(1 for i in self.interaction_history if i['type'] == 'spelled_word')
        
        return {
            'total_interactions': len(self.interaction_history),
            'word_detections': word_count,
            'letter_detections': letter_count,
            'spelled_words': spelled_count,
            'first_interaction': self.interaction_history[0]['datetime'],
            'last_interaction': self.interaction_history[-1]['datetime']
        }
    
    def clear_history(self):
        """Hapus riwayat interaksi"""
        self.interaction_history = []


# Contoh penggunaan
if __name__ == '__main__':
    print("\n" + "="*60)
    print("RESPONSE ENGINE TEST")
    print("="*60 + "\n")
    
    # Inisialisasi engine
    engine = ResponseEngine()
    
    print("✅ Response engine initialized\n")
    
    # Tes respons kata
    print("🧪 Testing word responses...")
    response = engine.process_word("halo")
    print(f"   Input: 'halo'")
    print(f"   Response: {response}\n")
    
    response = engine.process_word("terimakasih")
    print(f"   Input: 'terimakasih'")
    print(f"   Response: {response}\n")
    
    # Tes respons huruf
    print("🧪 Testing letter responses...")
    for letter in ['C', 'A', 'T']:
        response = engine.process_letter(letter)
        print(f"   Letter: {letter} → {response}")
    
    print()
    
    # Tes kata yang dieja
    print("🧪 Testing spelled word...")
    response = engine.process_spelled_word("CAT")
    print(f"   Spelled word: 'CAT'")
    print(f"   Response: {response}\n")
    
    # Cek riwayat
    print("📊 Interaction History:")
    history = engine.get_interaction_history(limit=5)
    for i, interaction in enumerate(history, 1):
        print(f"   {i}. [{interaction['type']}] {interaction['content']}")
    
    # Statistik
    print("\n📊 Statistics:")
    stats = engine.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Response engine test completed!")
