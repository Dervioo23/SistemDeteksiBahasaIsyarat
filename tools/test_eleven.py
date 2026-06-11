
from elevenlabs.client import ElevenLabs
import os

api_key = os.getenv("ELEVENLABS_API_KEY")

if not api_key:
    raise RuntimeError("ELEVENLABS_API_KEY is not set. Add it to your environment or .env file.")

try:
    client = ElevenLabs(api_key=api_key)
    print("Authentication Init...")
    
    print("Fetching Voices...")
    response = client.voices.get_all()
    
    # Handle response structure
    voices = response.voices if hasattr(response, 'voices') else response

    print(f"Found {len(voices)} voices.")
    
    count = 0
    for voice in voices:
        print(f"Name: {voice.name} | ID: {voice.voice_id} | Category: {voice.category}")
        count += 1
        if count >= 5: break
    
    print("SUCCESS: Listed voices.")

except Exception as e:
    print(f"ERROR: {e}")
