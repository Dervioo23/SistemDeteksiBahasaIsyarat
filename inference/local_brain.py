import json
import logging
from typing import Dict

from data_collection.utils import load_config

logger = logging.getLogger(__name__)

class LocalBrain:
    """
    Bridge Pengganti AI (Local Dictionary).
    Menggunakan respon statis dari config.json.
    """
    
    def __init__(self):
        self.responses: Dict[str, str] = {}
        self.load_responses()
        
    def load_responses(self):
        """Muat kamus respon dari config.json"""
        try:
            config = load_config()
            self.responses = config.get("responses", {})
            # Normalisasi key ke lowercase agar pencarian case-insensitive
            self.responses = {k.lower(): v for k, v in self.responses.items()}
            logger.info(f"Local Brain loaded with {len(self.responses)} responses.")
        except Exception as e:
            logger.error(f"Failed to load responses: {e}")
            self.responses = {}

    def generate_response(self, user_text: str) -> str:
        """
        Cari jawaban di kamus lokal.
        """
        if not user_text:
            return ""
            
        # Bersihkan input dan cari di kamus
        key = user_text.strip().lower()
        
        # 1. Exact Match
        if key in self.responses:
            return self.responses[key]
        
        # 2. Fuzzy / Keyword simple (Optional)
        # Jika input mengandung keyword tertentu
        for known_key, response in self.responses.items():
            if known_key in key:
                return response
                
        # 3. Default
        return self.responses.get("default", "Maaf, saya tidak mengerti.")

# Test
if __name__ == "__main__":
    brain = LocalBrain()
    print(f"Halo -> {brain.generate_response('Halo')}")
    print(f"HELLO -> {brain.generate_response('HELLO')}")
    print(f"Apa Kabar -> {brain.generate_response('Apa Kabar')}")
    print(f"Unknown -> {brain.generate_response('blabla')}")
