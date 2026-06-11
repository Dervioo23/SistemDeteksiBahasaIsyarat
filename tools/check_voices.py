import pyttsx3

try:
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    
    print("\nAVAILABLE VOICES ON THIS SYSTEM:")
    print("="*60)
    
    found_indo = False
    for index, voice in enumerate(voices):
        print(f"Index: {index}")
        print(f"ID: {voice.id}")
        print(f"Name: {voice.name}")
        print("-" * 30)
        
        if "indonesia" in voice.name.lower() or "id-id" in str(voice.languages).lower():
            found_indo = True
            
    print("="*60)
    if found_indo:
        print("INDONESIAN VOICE FOUND!")
    else:
        print("NO INDONESIAN VOICE FOUND.")
        print("System will use default voice.")

except Exception as e:
    print(f"Error: {e}")
