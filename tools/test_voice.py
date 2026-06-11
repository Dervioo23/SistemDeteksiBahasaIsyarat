from inference.voice_generator import VoiceGenerator
import time

print("Testing Voice Generator...")
try:
    v = VoiceGenerator()
    text = "Halo, ini adalah tes suara Ardi dari sistem penerjemah bahasa isyarat."
    print(f"Speaking: '{text}'")
    v.speak(text)
    
    print("Waiting for audio to finish...")
    time.sleep(10)
    print("Test Complete.")
except Exception as e:
    print(f"Error: {e}")
