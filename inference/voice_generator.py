
import pygame
import threading
import os
import time
import uuid
import queue
import pyttsx3 # Fallback
from elevenlabs.client import ElevenLabs # Primary
from dotenv import load_dotenv

class VoiceGenerator:
    """
    Generator Suara Hybrid:
    1. ElevenLabs (High Quality, API Key Required)
    2. pyttsx3 (Offline, Robot)
    """
    def __init__(self, voice: str = "id-ID-ArdiNeural", rate: str = "+0%", volume: str = "+0%"):
        self.output_dir = "Voice"
        os.makedirs(self.output_dir, exist_ok=True)
        load_dotenv()
        
        # Load Config
        from data_collection.utils import load_config
        self.config = load_config()
        self.tts_config = self.config.get("tts", {})
        
        # Init Pygame Mixer
        try:
            pygame.mixer.init()
            print(f"INFO: Voice Generator initialized")
        except Exception as e:
            print(f"ERROR: Failed to init Audio Mixer: {e}")

        # 1. Init ElevenLabs (Primary)
        self.eleven = None
        # Hardcoded ID for 'Rachel' (Standard Voice)
        # Bisa diganti ID lain: 29vD33N1CtxCmqQRPOHJ (Drew), 2EiwWnXFnvU5JabPnv8n (Clyde)
        # "Putra" Voice ID (Indonesian)
        # "Putra" Voice ID (Indonesian)
        self.eleven_voice_id = self.tts_config.get("elevenlabs_voice_id", "RWiGLY9uXI70QL540WNd")
        self.eleven_model = self.tts_config.get("elevenlabs_model", "eleven_multilingual_v2")
        try:
            api_key = os.getenv("ELEVENLABS_API_KEY")
            if not api_key:
                raise RuntimeError("ELEVENLABS_API_KEY is not set")
            self.eleven = ElevenLabs(api_key=api_key)
            print("INFO: ElevenLabs TTS: Ready (Primary)")
        except Exception as e:
            print(f"WARNING: ElevenLabs Init Failed: {e}")

        # 2. Init Fallback Engine (Offline)
        self.fallback_engine = None
        try:
            self.fallback_engine = pyttsx3.init()
            self.fallback_engine.setProperty('rate', 150)
            self.fallback_engine.setProperty('volume', 1.0)
            for voice_opt in self.fallback_engine.getProperty('voices'):
                if 'indonesia' in voice_opt.name.lower():
                    self.fallback_engine.setProperty('voice', voice_opt.id)
                    break
        except Exception as e:
            print(f"WARNING: Fallback TTS init failed: {e}")

        # --- QUEUE SYSTEM ---
        self.queue = queue.Queue()
        self.is_speaking = False
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()

    def speak(self, text: str):
        """Masukkan teks ke antrian"""
        if not text: return
        self.queue.put(text)

    def _process_queue(self):
        """Loop worker"""
        while True:
            text = self.queue.get()
            if text is None: break
            
            self.is_speaking = True
            try:
                self._speak_now(text)
            except Exception as e:
                print(f"ERROR: Worker Error: {e}")
            finally:
                self.is_speaking = False
                self.queue.task_done()

    def _speak_now(self, text: str):
        """Eksekusi bicara dengan 3 layer fallback"""
        filename_only = f"speech_{uuid.uuid4().hex}.mp3"
        unique_filename = os.path.join(self.output_dir, filename_only)
        success = False

        # LAYER 1: ElevenLabs
        if self.eleven:
            try:
                print(f"DEBUG: Generating ElevenLabs: {text[:20]}...")
                audio_gen = self.eleven.text_to_speech.convert(
                    text=text,
                    voice_id=self.eleven_voice_id,
                    model_id=self.eleven_model
                )
                
                # Write iterator to file
                with open(unique_filename, "wb") as f:
                    for chunk in audio_gen:
                        f.write(chunk)
                
                print("DEBUG: Playing ElevenLabs Audio...")
                self._play_file(unique_filename)
                success = True
            except Exception as e:
                # Fallback silently if quota exceeded or error
                print(f"WARNING: ElevenLabs Failed ({e}), switching to Offline...")
                pass
        
        # LAYER 2: Offline Fallback (Jika Layer 1 gagal)
        if not success and self.fallback_engine:
            try:
                self.fallback_engine.say(text)
                self.fallback_engine.runAndWait()
            except Exception as e:
                print(f"ERROR: Offline TTS Failed: {e}")

    def _play_file(self, filename):
        if os.path.exists(filename):
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            self._cleanup_file(filename)

    def _cleanup_file(self, filename: str):
        time.sleep(0.5) 
        try:
            pygame.mixer.music.unload()
            if os.path.exists(filename):
                os.remove(filename)
        except: pass

# Test
if __name__ == "__main__":
    v = VoiceGenerator()
    print("Queuing 3 sentences...")
    v.speak("Satu")
    v.speak("Dua")
    v.speak("Tiga")
    
    while v.is_speaking or not v.queue.empty():
        time.sleep(0.1)
    print("Done")
