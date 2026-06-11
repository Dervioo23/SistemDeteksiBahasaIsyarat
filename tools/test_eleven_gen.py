
from elevenlabs.client import ElevenLabs
import os

api_key = os.getenv("ELEVENLABS_API_KEY")

if not api_key:
    raise RuntimeError("ELEVENLABS_API_KEY is not set. Add it to your environment or .env file.")

try:
    client = ElevenLabs(api_key=api_key)
    print("Trying to generate audio (v2 syntax)...")
    
    # "Rachel" Voice ID: 21m00Tcm4TlvDq8ikWAM
    # Model: eleven_multilingual_v2
    audio_generator = client.text_to_speech.convert(
        text="Halo, ini tes suara Indonesia.",
        voice_id="21m00Tcm4TlvDq8ikWAM",
        model_id="eleven_multilingual_v2"
    )
    
    # Consume generator
    count = 0
    for chunk in audio_generator:
        count += len(chunk)
        
    print(f"SUCCESS: Generated {count} bytes.")

except Exception as e:
    print(f"ERROR: {e}")
