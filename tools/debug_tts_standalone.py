
import threading
import os
import time
import uuid
import queue
import pyttsx3
import pygame
from elevenlabs.client import ElevenLabs

# Duplicate class to avoid import errors
class VoiceGenerator:
    def __init__(self):
        self.output_dir = "Voice"
        os.makedirs(self.output_dir, exist_ok=True)
        
        try:
            pygame.mixer.init()
            print("SUCCESS: Pygame Mixer initialized")
        except Exception as e:
            print(f"FAILED to init Audio Mixer: {e}")

        # 1. Init ElevenLabs
        self.eleven = None
        # "Putra" Voice ID (Indonesian)
        self.eleven_voice_id = "RWiGLY9uXI70QL540WNd" 
        self.eleven_model = "eleven_multilingual_v2"
        try:
            api_key = os.getenv("ELEVENLABS_API_KEY")
            if not api_key:
                raise RuntimeError("ELEVENLABS_API_KEY is not set")
            self.eleven = ElevenLabs(api_key=api_key)
            print("SUCCESS: ElevenLabs: Initialized")
        except Exception as e:
            print(f"FAILED: ElevenLabs Init Failed: {e}")

        self.queue = queue.Queue()
        self.is_speaking = False
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()

    def speak(self, text: str):
        self.queue.put(text)

    def _process_queue(self):
        while True:
            text = self.queue.get()
            if text is None: break
            
            self.is_speaking = True
            try:
                self._speak_now(text)
            except Exception as e:
                print(f"FAILED: Worker Error: {e}")
            finally:
                self.is_speaking = False
                self.queue.task_done()

    def _speak_now(self, text: str):
        print(f"Attempting to speak: {text}")
        filename_only = f"speech_{uuid.uuid4().hex}.mp3"
        unique_filename = os.path.join(self.output_dir, filename_only)
        success = False

        if self.eleven:
            try:
                print("calling elevenlabs convert...")
                audio_gen = self.eleven.text_to_speech.convert(
                    text=text,
                    voice_id=self.eleven_voice_id,
                    model_id=self.eleven_model
                )
                
                print("writing file...")
                with open(unique_filename, "wb") as f:
                    for chunk in audio_gen:
                        f.write(chunk)
                
                print("playing file...")
                self._play_file(unique_filename)
                success = True
                print("SUCCESS: ElevenLabs Success")
            except Exception as e:
                print(f"FAILED: ElevenLabs Failed: {e}")
                import traceback
                traceback.print_exc()

    def _play_file(self, filename):
        if os.path.exists(filename):
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            time.sleep(0.5)
            try:
                pygame.mixer.music.unload()
                os.remove(filename)
            except: pass

if __name__ == "__main__":
    v = VoiceGenerator()
    v.speak("Tes satu dua tiga.")
    time.sleep(5)
