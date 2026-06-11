
import sys
import os
sys.path.append(os.getcwd())
from inference.voice_generator import VoiceGenerator
import time

print("--- Testing VoiceGenerator Class ---")
try:
    v = VoiceGenerator()
    print("Calling speak()...")
    v.speak("Halo, ini tes debug.")
    
    # Wait for queue processing (max 10s)
    for _ in range(20):
        if not v.is_speaking and v.queue.empty():
            break
        time.sleep(0.5)
        
    print("Done waiting.")
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
