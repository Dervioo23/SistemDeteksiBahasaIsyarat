import os
import time
from typing import Optional, Dict
from abc import ABC, abstractmethod
import threading


class TTSEngine(ABC):
    """Kelas dasar abstrak untuk mesin TTS"""
    
    @abstractmethod
    def speak(self, text: str):
        """Ucapkan teks yang diberikan"""
        pass
    
    @abstractmethod
    def is_speaking(self) -> bool:
        """Periksa apakah sedang berbicara"""
        pass


class PyttxTTSEngine(TTSEngine):
    """
    Mesin TTS menggunakan pyttsx3 (offline)
    Direkomendasikan untuk POC karena tidak butuh internet
    """
    
    def __init__(self, rate: int = 150, volume: float = 1.0):
        """
        Inisialisasi mesin pyttsx3
        
        Args:
            rate: Kecepatan bicara (kata per menit)
            volume: Volume (0.0 hingga 1.0)
        """
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', volume)
            self._speaking = False
            self.available = True
            print("✅ pyttsx3 TTS engine initialized")
        except Exception as e:
            print(f"❌ Error initializing pyttsx3: {e}")
            self.available = False
    
    def speak(self, text: str):
        """Ucapkan teks (memblokir)"""
        if not self.available:
            print(f"TTS not available, would say: {text}")
            return
        
        try:
            self._speaking = True
            self.engine.say(text)
            self.engine.runAndWait()
            self._speaking = False
        except Exception as e:
            print(f"❌ TTS error: {e}")
            self._speaking = False
    
    def speak_async(self, text: str):
        """Ucapkan teks (non-memblokir)"""
        thread = threading.Thread(target=self.speak, args=(text,))
        thread.daemon = True
        thread.start()
    
    def is_speaking(self) -> bool:
        """Periksa apakah sedang berbicara"""
        return self._speaking


class GTTSTTSEngine(TTSEngine):
    """
    Mesin TTS menggunakan gTTS (online)
    Lebih natural tapi butuh internet
    """
    
    def __init__(self, lang: str = 'id'):
        """
        Inisialisasi mesin gTTS
        
        Args:
            lang: Kode bahasa ('id' untuk Indonesia, 'en' untuk Inggris)
        """
        try:
            from gtts import gTTS
            import pygame
            pygame.mixer.init()
            self.gTTS = gTTS
            self.pygame = pygame
            self.lang = lang
            self._speaking = False
            self.available = True
            self.temp_file = 'temp_tts.mp3'
            print("✅ gTTS engine initialized")
        except Exception as e:
            print(f"❌ Error initializing gTTS: {e}")
            self.available = False
    
    def speak(self, text: str):
        """Ucapkan teks menggunakan gTTS"""
        if not self.available:
            print(f"TTS not available, would say: {text}")
            return
        
        try:
            self._speaking = True
            
            # Hasilkan ucapan
            tts = self.gTTS(text=text, lang=self.lang, slow=False)
            tts.save(self.temp_file)
            
            # Putar ucapan
            self.pygame.mixer.music.load(self.temp_file)
            self.pygame.mixer.music.play()
            
            # Tunggu hingga selesai
            while self.pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            # Bersihkan
            if os.path.exists(self.temp_file):
                os.remove(self.temp_file)
            
            self._speaking = False
        except Exception as e:
            print(f"❌ TTS error: {e}")
            self._speaking = False
    
    def speak_async(self, text: str):
        """Ucapkan teks (non-memblokir)"""
        thread = threading.Thread(target=self.speak, args=(text,))
        thread.daemon = True
        thread.start()
    
    def is_speaking(self) -> bool:
        """Periksa apakah sedang berbicara"""
        return self._speaking


class PrerecordedTTSEngine(TTSEngine):
    """
    TTS menggunakan file audio yang direkam sebelumnya
    Paling cepat dan paling natural tetapi membutuhkan file audio
    """
    
    def __init__(self, audio_dir: str = 'audio'):
        """
        Inisialisasi mesin rekaman
        
        Args:
            audio_dir: Direktori yang berisi file audio
        """
        try:
            import pygame
            pygame.mixer.init()
            self.pygame = pygame
            self.audio_dir = audio_dir
            self._speaking = False
            self.available = True
            
            # Muat file audio yang tersedia
            self.audio_files = {}
            if os.path.exists(audio_dir):
                for file in os.listdir(audio_dir):
                    if file.endswith(('.mp3', '.wav')):
                        label = os.path.splitext(file)[0].lower()
                        self.audio_files[label] = os.path.join(audio_dir, file)
            
            print(f"✅ Prerecorded TTS initialized ({len(self.audio_files)} files)")
        except Exception as e:
            print(f"❌ Error initializing prerecorded TTS: {e}")
            self.available = False
    
    def speak(self, text: str):
        """Putar audio yang direkam sebelumnya"""
        if not self.available:
            print(f"TTS not available, would say: {text}")
            return
        
        label = text.lower().strip()
        
        if label not in self.audio_files:
            print(f"⚠️  No audio file for: {text}")
            return
        
        try:
            self._speaking = True
            self.pygame.mixer.music.load(self.audio_files[label])
            self.pygame.mixer.music.play()
            
            while self.pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            self._speaking = False
        except Exception as e:
            print(f"❌ TTS error: {e}")
            self._speaking = False
    
    def speak_async(self, text: str):
        """Putar audio (non-memblokir)"""
        thread = threading.Thread(target=self.speak, args=(text,))
        thread.daemon = True
        thread.start()
    
    def is_speaking(self) -> bool:
        """Periksa apakah sedang berbicara"""
        return self._speaking


class ElevenLabsTTSEngine(TTSEngine):
    """
    Mesin TTS menggunakan ElevenLabs API (High Quality)
    Membutuhkan koneksi internet dan API Key valid
    """
    
    def __init__(self, api_key: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM", model_id: str = "eleven_multilingual_v2"):
        """
        Inisialisasi ElevenLabs Engine
        
        Args:
            api_key: API Key ElevenLabs
            voice_id: ID suara (default: Rachel)
            model_id: ID model (default: multilingual v2)
        """
        try:
            from elevenlabs.client import ElevenLabs
            import pygame
            import uuid
            
            self.client = ElevenLabs(api_key=api_key)
            pygame.mixer.init()
            self.pygame = pygame
            self.uuid = uuid
            
            self.voice_id = voice_id
            self.model_id = model_id
            self._speaking = False
            self.available = True
            self.output_dir = "Voice"
            os.makedirs(self.output_dir, exist_ok=True)
            
            print(f"✅ ElevenLabs engine initialized (Voice: {voice_id})")
        except Exception as e:
            print(f"❌ Error initializing ElevenLabs: {e}")
            self.available = False

    def speak(self, text: str):
        """Ucapkan teks menggunakan ElevenLabs"""
        if not self.available:
            print(f"ElevenLabs not available, would say: {text}")
            return

        try:
            self._speaking = True
            
            # Generate audio stream
            audio_stream = self.client.text_to_speech.convert(
                text=text,
                voice_id=self.voice_id,
                model_id=self.model_id
            )
            
            # Save to unique temp file
            filename = os.path.join(self.output_dir, f"speech_{self.uuid.uuid4().hex}.mp3")
            with open(filename, "wb") as f:
                for chunk in audio_stream:
                    f.write(chunk)
            
            # Play audio
            self.pygame.mixer.music.load(filename)
            self.pygame.mixer.music.play()
            
            while self.pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
            self._speaking = False
            
        except Exception as e:
            print(f"❌ ElevenLabs TTS error: {e}")
            self._speaking = False

    def speak_async(self, text: str):
        """Ucapkan teks (non-memblokir)"""
        thread = threading.Thread(target=self.speak, args=(text,))
        thread.daemon = True
        thread.start()

    def is_speaking(self) -> bool:
        return self._speaking


class TTSManager:
    """
    Manajer TTS untuk menangani beberapa mesin dengan fallback
    """
    
    def __init__(
        self,
        primary_engine: str = 'pyttsx3',
        fallback_engine: str = 'print',
        **kwargs
    ):
        """
        Inisialisasi Manajer TTS
        
        Args:
            primary_engine: 'elevenlabs', 'pyttsx3', 'gtts', atau 'prerecorded'
            fallback_engine: 'pyttsx3', 'gtts', 'prerecorded', atau 'print'
            **kwargs: Argumen untuk inisialisasi mesin
        """
        self.engines = {}
        self.current_engine = None
        self.cooldown_time = 2.0  # Detik antara ucapan
        self.last_speak_time = 0
        
        print(f"\n{'='*60}")
        print("INITIALIZING TTS MANAGER")
        print(f"{'='*60}\n")
        
        # Inisialisasi mesin
        self._init_engines(**kwargs)
        
        # Atur mesin utama
        if primary_engine in self.engines:
            self.current_engine = self.engines[primary_engine]
            print(f"✅ Primary engine: {primary_engine}")
        elif fallback_engine in self.engines:
            self.current_engine = self.engines[fallback_engine]
            print(f"⚠️  Primary '{primary_engine}' failed. Using fallback: {fallback_engine}")
        else:
            print("❌ No TTS engine available, using print mode")
    
    def _init_engines(self, **kwargs):
        """Inisialisasi mesin yang tersedia"""
        
        # 1. ElevenLabs
        eleven_args = kwargs.get('elevenlabs_args', {})
        if eleven_args and eleven_args.get('api_key'):
            try:
                engine = ElevenLabsTTSEngine(**eleven_args)
                if engine.available:
                    self.engines['elevenlabs'] = engine
            except Exception as e:
                print(f"⚠️  ElevenLabs not available: {e}")

        # 2. pyttsx3
        try:
            engine = PyttxTTSEngine(**kwargs.get('pyttsx3_args', {}))
            if engine.available:
                self.engines['pyttsx3'] = engine
        except Exception as e:
            print(f"⚠️  pyttsx3 not available: {e}")
        
        # 3. gTTS
        try:
            engine = GTTSTTSEngine(**kwargs.get('gtts_args', {}))
            if engine.available:
                self.engines['gtts'] = engine
        except Exception as e:
            print(f"⚠️  gTTS not available: {e}")
        
        # 4. Prerecorded
        try:
            engine = PrerecordedTTSEngine(**kwargs.get('prerecorded_args', {}))
            if engine.available:
                self.engines['prerecorded'] = engine
        except Exception as e:
            print(f"⚠️  Prerecorded not available: {e}")
    
    def speak(self, text: str, force: bool = False):
        """
        Ucapkan teks dengan perlindungan cooldown
        
        Args:
            text: Teks untuk diucapkan
            force: Paksa bicara meskipun dalam cooldown
        """
        # Periksa cooldown
        current_time = time.time()
        if not force and (current_time - self.last_speak_time) < self.cooldown_time:
            print(f"🔇 Cooldown active, skipping: {text}")
            return
        
        # Periksa jika sudah berbicara
        if self.current_engine and self.current_engine.is_speaking():
            print(f"🔇 Already speaking, skipping: {text}")
            return
        
        # Bicara
        if self.current_engine:
            print(f"🔊 Speaking ({type(self.current_engine).__name__}): {text}")
            self.current_engine.speak_async(text)
            self.last_speak_time = current_time
        else:
            print(f"💬 {text}")
    
    def speak_blocking(self, text: str):
        """Ucapkan teks (memblokir)"""
        if self.current_engine:
            print(f"🔊 Speaking: {text}")
            self.current_engine.speak(text)
        else:
            print(f"💬 {text}")
    
    def is_speaking(self) -> bool:
        """Periksa apakah sedang berbicara"""
        if self.current_engine:
            return self.current_engine.is_speaking()
        return False
    
    def set_cooldown(self, seconds: float):
        """Atur waktu cooldown antara ucapan"""
        self.cooldown_time = seconds
    
    def get_available_engines(self) -> list:
        """Dapatkan daftar mesin yang tersedia"""
        return list(self.engines.keys())


# Contoh penggunaan
if __name__ == '__main__':
    print("🧪 Testing TTS Engines...\n")
    
    # Load env for test
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("ELEVENLABS_API_KEY")
    
    # Inisialisasi manajer
    tts = TTSManager(
        primary_engine='elevenlabs',
        fallback_engine='pyttsx3',
        elevenlabs_args={'api_key': api_key}
    )
    
    print(f"\nAvailable engines: {tts.get_available_engines()}")
    
    # Uji output ucapan
    if 'elevenlabs' in tts.get_available_engines():
        print("\n🧪 Testing ElevenLabs speech output...")
        tts.speak("Halo, ini tes suara ElevenLabs.", force=True)
        time.sleep(5)
    else:
        print("\n🧪 Testing fallback speech output...")
        tts.speak("Halo, fallback.", force=True)
        time.sleep(3)
    
    print("\n✅ TTS tests completed!")
