import os
import logging
from dotenv import load_dotenv
from groq import Groq

# Setup logging
logger = logging.getLogger(__name__)

# Load env vars
load_dotenv()

class GroqAssistant:
    """
    Wrapper untuk Groq API (Speed LPU)
    Tugas: Menjadi lawan bicara (Chatbot) untuk pengguna Bahasa Isyarat.
    """
    
    def __init__(self, api_key: str = None):
        # Coba ambil dari parameter atau env
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.history = [] # Simpan riwayat percakapan
        
        if not self.api_key:
            logger.warning("GROQ_API_KEY not found! AI features will be disabled.")
            self.client = None
        else:
            try:
                self.client = Groq(api_key=self.api_key)
                logger.info("Groq Client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to init Groq client: {e}")
                self.client = None

    def clear_history(self):
        """Hapus ingatan percakapan"""
        self.history = []
        logger.info("Chat history cleared.")

    def generate_chat_response(self, user_input: str) -> str:
        """
        Menghasilkan jawaban percakapan dari input pengguna.
        Contoh Input: "SAYA LAPAR MAKAN"
        Contoh Output: "Tentu, mari kita cari makanan. Kamu ingin makan apa?"
        """
        if not self.client or not user_input.strip():
            return "Maaf, saya tidak mengerti."

        try:
            # System Prompt untuk Persona Chatbot
            system_prompt = (
                "Kamu adalah asisten yang ramah dan membantu untuk pengguna Bahasa Isyarat. "
                "Input pengguna mungkin berupa kata-kata terputus atau tidak lengkap strukturnya. "
                "Tugasmu adalah memahami maksud pengguna dan MEMBERIKAN JAWABAN yang relevan, sopan, dan natural dalam Bahasa Indonesia. "
                "Jangan memperbaiki kalimat pengguna, tapi JAWABLAH kalimat tersebut layaknya percakapan normal. "
                "PENTING: Jawab dengan SANGAT RINGKAS (maksimal 1-2 kalimat pendek) agar cepat dibacakan oleh suara. Jangan bertele-tele."
                "Jika Saya Berkata halo, Hallo, Hello, Hai, Hi, Balas dengan salam Hallo juga, Apakah ada yang bisa saya bantu?."
                "Jika Saya Berkata terima kasih, Thanks, Thank you, Balas dengan ungkapan yang sama."
                "Jika Saya Berkata siapa namamu, Siapa kamu, Kenalan yuk, Balas dengan memperkenalkan diri."
                "Jika Saya Berkata apa kabar, Bagaimana kabarmu, Balas dengan menanyakan kabar balik."
                "Jika Saya Berkata selamat tinggal, Dadah, Bye, Balas dengan ungkapan perpisahan yang sama."
                "Jika Saya Berkata kamu siapa, Kamu apa, Balas dengan memperkenalkan diri."
                "Jika Saya Berkata kamu bisa apa, Kamu jago apa, Balas dengan menjelaskan kemampuanmu."
                "Jika Saya Berkata kamu dari mana, Kamu asalnya dimana, Balas dengan menjelaskan asal usulmu."
                "Jika Saya Berkata kamu umur berapa, Kamu tua atau muda, Balas dengan menjelaskan usiamu."
                "Jika Saya Berkata kamu buatan siapa, Kamu dibuat oleh siapa, Balas dengan menjelaskan pembuatmu."
                "Jika Saya Berkata kamu suka apa, Kamu hobi apa, Balas dengan menjelaskan hobimu."
                "Jika Saya Berkata kamu makan apa, Kamu suka makan apa, Balas dengan menjelaskan makanan favoritmu."
                "Jika Saya Mengulang Perkataan yang Sama, Jawablah dengan Perkataa yang Sama dengan jawaban sebelumnya."
                "Jika Saya Memberikan Perintah, Jawablah dengan Mengikuti Perintah Tersebut."
            )

            # Bangun pesan dengan riwayat
            messages = [{"role": "system", "content": system_prompt}]
            
            # Tambahkan riwayat (maksimal 10 terakhir agar tidak overload)
            messages.extend(self.history[-10:])
            
            # Tambahkan input pengguna saat ini
            messages.append({"role": "user", "content": user_input})

            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model="llama-3.3-70b-versatile", 
                temperature=0.7, # Sedikit kreatif untuk percakapan
                max_tokens=150,
            )

            result = chat_completion.choices[0].message.content.strip()
            # Bersihkan tanda kutip jika ada
            result = result.replace('"', '').replace("'", "")
            
            # Simpan ke riwayat
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": result})
            
            logger.info(f"Groq Chat: User='{user_input}' -> AI='{result}'")
            return result

        except Exception as e:
            logger.error(f"Groq API Error: {e}")
            print(f"\n❌ GROQ SYSTEM ERROR: {e}") # Print to console directly
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)[:50]}..." # Show partial error on UI

# Singleton instance
groq_assistant = GroqAssistant()
