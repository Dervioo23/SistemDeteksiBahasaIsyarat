
import asyncio
import edge_tts

async def main():
    voices = await edge_tts.list_voices()
    for v in voices:
        if "id-ID" in v["ShortName"]:
            print(f"Name: {v['ShortName']}")
            print(f"Gender: {v['Gender']}")
            print("-" * 20)

if __name__ == "__main__":
    asyncio.run(main())
