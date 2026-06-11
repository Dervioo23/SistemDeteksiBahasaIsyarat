import cv2
import numpy as np
import os
import logging
from typing import Optional, Tuple, Dict

from data_collection.landmark_extractor import LandmarkExtractor
from inference.word_detector import WordDetector
from inference.alphabet_detector import AlphabetDetector
from inference.sentence_builder import SentenceBuilder, InputMode
from inference.groq_client import groq_assistant
from inference.utils import setup_camera, run_camera_loop, normalize_landmarks
from data_collection.utils import load_config
from inference.sequence_handler import SequenceHandler
from inference.response_engine import ResponseEngine
from inference.tts_engine import TTSManager

logger = logging.getLogger(__name__)


class HybridDetector:
    """Sistem deteksi hibrida utama"""
    
    def __init__(
        self,
        word_model_path: str,
        alphabet_model_path: str,
        mode: str = 'hybrid'
    ):
        """
        Inisialisasi detektor hibrida
        
        Args:
            word_model_path: Path ke model kata
            alphabet_model_path: Path ke model alfabet
            mode: 'word', 'alphabet', atau 'hybrid'
        """
        self.mode = mode
        
        print("\n" + "="*60)
        print("INITIALIZING HYBRID DETECTION SYSTEM")
        print("="*60)
        
        # Muat konfigurasi
        config = load_config()
        model_cfg = config.get("model", {})
        coll_cfg = config.get("collection", {})

        det_conf = coll_cfg.get("min_detection_confidence", 0.7)
        track_conf = coll_cfg.get("min_tracking_confidence", 0.5)

        # Inisialisasi ekstraktor landmark
        print("\n📍 Initializing landmark extractor...")
        self.extractor = LandmarkExtractor(
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
            max_num_hands=2
        )
        
        # Inisialisasi detektor
        self.word_detector = None
        self.alphabet_detector = None
        
        if mode in ['word', 'hybrid']:
            if os.path.exists(word_model_path):
                print("\n🔤 Initializing word detector...")
                self.word_detector = WordDetector(
                    model_path=word_model_path,
                    sequence_length=coll_cfg.get("sequence_length_word", 45),
                    confidence_threshold=model_cfg.get("confidence_threshold_word", 0.55),
                    cooldown_seconds=3.0
                )
            else:
                print(f"⚠️  Word model not found: {word_model_path}")
        
        if mode in ['alphabet', 'hybrid']:
            if os.path.exists(alphabet_model_path):
                print("\n🔠 Initializing alphabet detector...")
                self.alphabet_detector = AlphabetDetector(
                    model_path=alphabet_model_path,
                    confidence_threshold=model_cfg.get("confidence_threshold_alphabet", 0.5),
                    stability_frames=3,
                    cooldown_seconds=1.5
                )
            else:
                print(f"⚠️  Alphabet model not found: {alphabet_model_path}")
        
        # Inisialisasi penangan urutan
        print("\n📝 Initializing sequence handler...")
        self.sequence_handler = SequenceHandler(
            max_sequence_length=20,
            letter_timeout=3.0,
            word_complete_timeout=2.0
        )
        
        # Inisialisasi Sentence Builder (AI Powered)
        print("\n🧠 Initializing Sentence Builder (Groq AI Ready)...")
        self.sentence_builder = SentenceBuilder()
        
        # Inisialisasi mesin respons
        print("\n💬 Initializing response engine...")
        self.response_engine = ResponseEngine()
        
        # Inisialisasi TTS
        print("\n🔊 Initializing TTS engine...")
        
        # Load environment variables for API keys
        from dotenv import load_dotenv
        load_dotenv()
        
        eleven_api_key = os.getenv("ELEVENLABS_API_KEY")
        primary_engine = 'pyttsx3'
        
        eleven_args = {}
        if eleven_api_key:
            print(f"🔑 Found ElevenLabs API Key: {eleven_api_key[:5]}...")
            primary_engine = 'elevenlabs'
            eleven_args = {
                'api_key': eleven_api_key,
                'voice_id': config.get("tts", {}).get("elevenlabs_voice_id", "RWiGLY9uXI70QL540WNd")
            }
        
        self.tts = TTSManager(
            primary_engine=primary_engine,
            fallback_engine='pyttsx3',
            elevenlabs_args=eleven_args
        )
        
        # Status deteksi
        self.current_mode = 'word' if mode == 'word' else 'alphabet' if mode == 'alphabet' else 'word'
        self.last_spoken_word = None
        self.last_spoken_letter = None
        self.last_spelled_word = None
        
        print("\n" + "="*60)
        print("✅ SYSTEM READY")
        print("="*60)
        print(f"   Mode: {mode}")
        print(f"   Word detector: {'✅' if self.word_detector else '❌'}")
        print(f"   Alphabet detector: {'✅' if self.alphabet_detector else '❌'}")
        print(f"   AI Refinement: Ready (Press 'F' to finish sentence)")
        print("="*60 + "\n")
    
    def process_frame(
        self,
        frame: np.ndarray
    ) -> Tuple[np.ndarray, Optional[str], float]:
        """
        Proses satu frame
        
        Args:
            frame: Frame input
            
        Returns:
            Tuple dari (annotated_frame, detected_label, confidence)
        """
        # Ekstrak landmark
        hands_dict, annotated_frame, num_hands = self.extractor.extract_both_hands(frame)
        
        detected_label = None
        confidence = 0.0
        
        if num_hands > 0:
            # Dapatkan landmark dengan fallback
            hands_data = self.extractor.get_both_hands_with_fallback(hands_dict)
            
            # Ratakan ke array tunggal (126 fitur)
            landmarks = self.extractor.flatten_both_hands(hands_data)
            
            # Konversi ke array numpy jika itu list
            if isinstance(landmarks, list):
                landmarks = np.array(landmarks, dtype=np.float32)
            
            # Normalisasi
            landmarks_normalized = self._normalize_landmarks(landmarks)
            
            # Deteksi berdasarkan mode saat ini
            if self.current_mode == 'word' and self.word_detector:
                detected_label, confidence = self.word_detector.detect(landmarks_normalized)
                
                if detected_label and detected_label != self.last_spoken_word:
                    response = self.response_engine.process_word(detected_label)
                    self.tts.speak(response)
                    self.last_spoken_word = detected_label
                    
                    # Add to Sentence Builder
                    self.sentence_builder.sentence_parts.append(detected_label)
            
            elif self.current_mode == 'alphabet' and self.alphabet_detector:
                detected_label, confidence = self.alphabet_detector.detect(landmarks_normalized)
                
                # Debug: cetak prediksi
                if confidence > 0.0:
                    logger.debug("Alphabet prediction: %s conf=%.3f", detected_label, confidence)
                
                if detected_label and detected_label != 'not_C':
                    # Tambahkan ke urutan
                    added = self.sequence_handler.add_letter(detected_label)
                    if added:
                        response = self.response_engine.process_letter(detected_label)
                        if detected_label != self.last_spoken_letter:
                            self.tts.speak(detected_label)  # Hanya ucapkan huruf
                            self.last_spoken_letter = detected_label
        
        # Periksa apakah kata yang dieja sudah lengkap
        if self.sequence_handler.is_word_complete():
            spelled_word = self.sequence_handler.complete_word()
            if spelled_word:
                response = self.response_engine.process_spelled_word(spelled_word)
                if spelled_word != self.last_spelled_word:
                    self.tts.speak(spelled_word, force=True)  # Ucapkan kata lengkap
                    self.last_spelled_word = spelled_word
                    
                    # Add to Sentence Builder
                    self.sentence_builder.sentence_parts.append(spelled_word)
        
        # Gambar UI
        annotated_frame = self._draw_ui(annotated_frame, num_hands, detected_label, confidence)
        
        return annotated_frame, detected_label, confidence
    
    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Normalisasi landmark"""
        return normalize_landmarks(landmarks)
    
    def _draw_ui(
        self,
        frame: np.ndarray,
        num_hands: int,
        detected_label: Optional[str],
        confidence: float
    ) -> np.ndarray:
        """Gambar overlay UI pada frame"""
        h, w = frame.shape[:2]
        
        # Latar belakang untuk UI
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Mode
        cv2.putText(frame, f"Mode: {self.current_mode.upper()}", 
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Tangan terdeteksi
        cv2.putText(frame, f"Hands: {num_hands}", 
                   (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Deteksi
        if detected_label:
            cv2.putText(frame, f"Detected: {detected_label} ({confidence:.2f})", 
                       (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Urutan saat ini (mode alfabet)
        if self.current_mode == 'alphabet':
            current_word = self.sequence_handler.get_current_word()
            if current_word:
                cv2.rectangle(frame, (10, h-60), (w-10, h-10), (0, 0, 0), -1)
                cv2.putText(frame, f"Spelling: {current_word}", 
                           (20, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                           
        # Draw Sentence Builder
        current_sentence = self.sentence_builder.get_built_sentence(refine=False)
        if current_sentence:
            # Box di bawah (di atas controls)
            cv2.rectangle(frame, (10, h-100), (w-10, h-60), (50, 50, 50), -1)
            cv2.putText(frame, f"Sentence: {current_sentence}", 
                       (20, h-70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Kontrol
        cv2.putText(frame, "Q:Quit | M:Mode | R:Reset", 
                   (w-350, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def switch_mode(self):
        """Beralih antara mode kata dan alfabet"""
        if self.mode == 'hybrid':
            if self.current_mode == 'word':
                self.current_mode = 'alphabet'
                print("🔄 Switched to ALPHABET mode")
            else:
                self.current_mode = 'word'
                print("🔄 Switched to WORD mode")
                self.sequence_handler.reset_sequence()
    
    def reset(self):
        """Reset semua detektor"""
        if self.word_detector:
            self.word_detector.reset_buffer()
        if self.alphabet_detector:
            self.alphabet_detector.reset_stability()
        self.sequence_handler.reset_sequence()
        print("🔄 Detectors reset")
    
    def run(self):
        """Jalankan deteksi waktu nyata"""
        print("\n" + "="*60)
        print("STARTING REAL-TIME DETECTION")
        print("="*60)
        print("\nControls:")
        print("  Q - Quit")
        print("  M - Switch mode (word/alphabet)")
        print("  R - Reset detectors")
        print("  F - Finish & Speak Sentence (AI Refined)")
        print("  C - Clear Sentence")
        print("  Backspace - Delete last word")
        print("="*60 + "\n")
        
        # Inisialisasi kamera
        cap = setup_camera(device_index=0, width=640, height=480, fps=30)

        if not cap.isOpened():
            print("❌ Cannot open camera!")
            print("   Tips: Pastikan kamera terhubung dan tidak dipakai aplikasi lain.")
            logger.error("Camera could not be opened. Check connection or device index.")
            return

        def frame_processor(frame: np.ndarray) -> np.ndarray:
            # Balik untuk efek cermin
            frame = cv2.flip(frame, 1)
            annotated_frame, _, _ = self.process_frame(frame)
            return annotated_frame

        def key_handler(key: int) -> bool:
            if key == ord('q'):
                return False
            elif key == ord('m'):
                self.switch_mode()
            elif key == ord('r'):
                self.reset()
            elif key == ord('f'): # FINISH & CHAT
                raw = self.sentence_builder.get_built_sentence()
                if raw:
                    print(f"\n🗣️  User says: '{raw}'")
                    print(f"🤖 AI is thinking...")
                    
                    # AI CHATBOT RESPONSE
                    answer = groq_assistant.generate_chat_response(raw)
                    print(f"✅ AI Answers: '{answer}'")
                    
                    # Speak the ANSWER usually
                    self.tts.speak(answer, force=True)
                    self.sentence_builder.clear()
            elif key == ord('c'): # CLEAR
                self.sentence_builder.clear()
                groq_assistant.clear_history()
                print("🧹 Sentence & AI History cleared")
            elif key == 8: # BACKSPACE
                self.sentence_builder.backspace()
            return True

        try:
            run_camera_loop(
                cap=cap,
                window_name='Sign Language Detection',
                frame_processor=frame_processor,
                key_handler=key_handler,
            )

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.extractor.close()

            # Cetak statistik
            print("\n" + "="*60)
            print("SESSION STATISTICS")
            print("="*60)

            if self.word_detector:
                word_stats = self.word_detector.get_statistics()
                print(f"\n📊 Word Detection:")
                print(f"   Total: {word_stats['total_detections']}")
                print(f"   Successful: {word_stats['successful_detections']}")

            if self.alphabet_detector:
                alpha_stats = self.alphabet_detector.get_statistics()
                print(f"\n📊 Alphabet Detection:")
                print(f"   Total: {alpha_stats['total_detections']}")
                print(f"   Successful: {alpha_stats['successful_detections']}")

            response_stats = self.response_engine.get_statistics()
            print(f"\n📊 Interactions:")
            print(f"   Total: {response_stats['total_interactions']}")
            print(f"   Words: {response_stats['word_detections']}")
            print(f"   Letters: {response_stats['letter_detections']}")
            print(f"   Spelled: {response_stats['spelled_words']}")

            print("\n" + "="*60)
            print("✅ SESSION COMPLETED")
            print("="*60 + "\n")


def main():
    """Fungsi utama"""
    # Konfigurasi
    config = load_config()
    model_cfg = config.get("model", {})
    word_model_path = model_cfg.get('word_model_path', 'trained_models/word_halo_model_best.keras')
    alphabet_model_path = model_cfg.get('alphabet_model_path', 'trained_models/alphabet_C_model_best.keras')
    
    # Periksa model
    if not os.path.exists(word_model_path):
        print(f"⚠️  Word model not found: {word_model_path}")
    
    if not os.path.exists(alphabet_model_path):
        print(f"⚠️  Alphabet model not found: {alphabet_model_path}")
    
    if not os.path.exists(word_model_path) and not os.path.exists(alphabet_model_path):
        print("\n❌ No models found! Please train models first:")
        print("   python training/train_word_model.py")
        print("   python training/train_alphabet_model.py")
        return
    
    # Inisialisasi detektor
    detector = HybridDetector(
        word_model_path=word_model_path,
        alphabet_model_path=alphabet_model_path,
        mode='hybrid'
    )
    
    # Jalankan
    detector.run()


if __name__ == '__main__':
    main()
