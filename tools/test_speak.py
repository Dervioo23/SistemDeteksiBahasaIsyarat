
import asyncio
import edge_tts
import os

async def test_voice(voice, text, filename):
    print(f"Testing {voice}...")
    try:
        comm = edge_tts.Communicate(text, voice)
        await comm.save(filename)
        print(f"SUCCESS: {filename}")
    except Exception as e:
        print(f"FAILED {voice}: {e}")

async def main():
    await test_voice("id-ID-ArdiNeural", "Tes Ardi", "test_ardi.mp3")
    await test_voice("id-ID-GadisNeural", "Tes Gadis", "test_gadis.mp3")

if __name__ == "__main__":
    asyncio.run(main())
