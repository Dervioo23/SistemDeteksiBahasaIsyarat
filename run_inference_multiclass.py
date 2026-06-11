import os
import cv2
import numpy as np
import logging
import threading

from inference.multiclass_detector import MultiClassWordDetector, MultiClassAlphabetDetector, load_multiclass_model
from inference.sentence_builder import SentenceBuilder, InputMode
from inference.groq_client import groq_assistant
from inference.local_brain import LocalBrain
from inference.voice_generator import VoiceGenerator
from data_collection.landmark_extractor import LandmarkExtractor
from data_collection.utils import load_config
from inference.utils import (
    setup_camera,
    run_camera_loop,
    normalize_landmarks,
    get_normalization_method_for_model,
)

logger = logging.getLogger(__name__)

def configure_logging() -> None:
    """Konfigurasi logging untuk inferensi multi-kelas berdasarkan config.json."""
    try:
        config = load_config()
        log_cfg = config.get("logging", {})

        level_name = str(log_cfg.get("level", "INFO")).upper()
        level = getattr(logging, level_name, logging.INFO)
        log_format = log_cfg.get(
            "format",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        log_file = log_cfg.get("file")

        kwargs = {"level": level, "format": log_format}

        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            kwargs["filename"] = log_file

        logging.basicConfig(**kwargs)
    except Exception as exc: 
        logging.basicConfig(level=logging.INFO)
        logger.exception("Failed to configure logging from config.json: %s", exc)


def main():
    """Peluncur utama"""
    
    configure_logging()

    print("\n" + "="*60)
    print("MULTI-CLASS SIGN LANGUAGE TRANSLATOR (OFFLINE + ONLINE VOICE)")
    print("="*60)
    print("\n🎯 Logic Core: Sentence Builder Activated")
    print("🧠 The Brain: Local Dictionary (Kamus JSON)")
    print("🗣️  The Voice: Microsoft Edge TTS (Ardi - Indonesia)")
    print("🛡️  Noise Filter: >0.5s stability required")
    print("🔄 Modes: Word & Spelling Assembly")
    
    # Muat konfigurasi
    config = load_config()
    model_cfg = config.get("model", {})
    coll_cfg = config.get("collection", {})
    
    # Ambang batas kepercayaan
    mc_word_conf_threshold = model_cfg.get("multiclass_confidence_threshold_word", 0.6)
    mc_alphabet_conf_threshold = model_cfg.get("multiclass_confidence_threshold_alphabet", 0.6)
    
    # Path model
    word_model_path = model_cfg.get("multiclass_word_model_path", 'trained_models/multiclass_word_model_final.keras')
    alphabet_model_path = model_cfg.get("multiclass_alphabet_model_path", 'trained_models/multiclass_alphabet_model_final.keras')
    
    # Periksa dan Inisialisasi Detektor
    word_detector = None
    alphabet_detector = None
    word_norm_method = "full"
    alphabet_norm_method = "full"
    
    if os.path.exists(word_model_path):
        try:
            model, class_names = load_multiclass_model(word_model_path)
            word_norm_method = get_normalization_method_for_model(word_model_path, default="full")
            word_detector = MultiClassWordDetector(
                model_path=word_model_path,
                class_names=class_names,
                sequence_length=45,
                confidence_threshold=mc_word_conf_threshold
            )
            print(f"\n✅ Word detector loaded: {len(class_names)} classes")
        except Exception as e:
            print(f"\n❌ Word model error: {e}")

    if os.path.exists(alphabet_model_path):
        try:
            model, class_names = load_multiclass_model(alphabet_model_path)
            alphabet_norm_method = get_normalization_method_for_model(alphabet_model_path, default="full")
            alphabet_detector = MultiClassAlphabetDetector(
                model_path=alphabet_model_path,
                class_names=class_names,
                confidence_threshold=mc_alphabet_conf_threshold,
                stability_frames=5 
            )
            print(f"\n✅ Alphabet detector loaded: {len(class_names)} classes")
        except Exception as e:
            print(f"\n❌ Alphabet model error: {e}")

    # --- LAYER 2: SENTENCE BUILDER ---
    sentence_builder = SentenceBuilder(
        min_stability_duration=0.5,
        word_sequence_timeout=2.0
    )
    
    # Set mode awal
    if word_detector:
        sentence_builder.set_mode(InputMode.WORD)
    else:
        sentence_builder.set_mode(InputMode.SPELLING)

    # --- LAYER 3: LOCAL BRAIN & VOICE ---
    brain = LocalBrain()
    voice = VoiceGenerator() # Inisialisasi Suara
    print(f"✅ Local Brain initialized ({len(brain.responses)} responses)")
    
    response_text = ""

    # Component lain
    extractor = LandmarkExtractor(
        min_detection_confidence=coll_cfg.get("min_detection_confidence", 0.7),
        min_tracking_confidence=coll_cfg.get("min_tracking_confidence", 0.5),
        max_num_hands=2
    )

    cap = setup_camera(device_index=0, width=640, height=480, fps=30)
    if not cap.isOpened():
        print("\n❌ Cannot open camera!")
        return

    print("\n" + "="*60)
    print("CONTROLS")
    print("="*60)
    print("   TAB   - Switch Mode (Word <-> Spelling)")
    print("   SPACE - Commit Spelling")
    print("   8     - Ask Brain (Local Response)")
    print("   ENTER - AI Chat (Groq -> Voice)")
    print("   7     - Manual Input (Type 1 char)")
    print("   BKSP  - Undo / Delete")
    print("   9     - Clear Sentence")
    print("   F     - Toggle Fullscreen")
    print("   0     - Quit")
    print("="*60)
    print("\n🚀 Starting Translator...\n")

    # State untuk input manual
    manual_input_active = False

    def frame_processor(frame: np.ndarray) -> np.ndarray:
        frame = cv2.flip(frame, 1)
        hands_dict, annotated_frame, num_hands = extractor.extract_both_hands(frame)
        h, w = annotated_frame.shape[:2]

        # --- UI LAYOUT ---
        cv2.rectangle(annotated_frame, (0, 0), (w, 80), (30, 30, 30), -1) # Header BG
        
        # 1. DETERMINE SYSTEM STATUS
        current_status = "LISTENING"
        status_color = (0, 255, 0) # Green
        status_icon = "LISTENING" # OpenCV doesn't support emoji well, using text
        
        if voice.is_speaking:
            current_status = "SPEAKING"
            status_color = (255, 100, 0) # Blue-ish
            status_icon = "SPEAKING"
        elif "Sedang berpikir" in response_text:
            current_status = "THINKING"
            status_color = (255, 0, 255) # Magenta
            status_icon = "THINKING"

        # 2. DRAW STATUS BAR (Top Center)
        # Background pill for status
        center_x = w // 2
        cv2.rectangle(annotated_frame, (center_x - 100, 10), (center_x + 100, 50), (60, 60, 60), -1)
        # Status Text
        cv2.putText(annotated_frame, f"{status_icon}", (center_x - 80, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        # Mode Indicator (Top Left)
        mode_str = "WORD" if sentence_builder.current_mode == InputMode.WORD else "SPELLING"
        mode_color = (0, 255, 0) if sentence_builder.current_mode == InputMode.WORD else (0, 200, 255)
        cv2.putText(annotated_frame, f"MODE: {mode_str}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        # Response Panel
        if response_text:
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 80), (w, 170), (20, 20, 60), -1)
            annotated_frame = cv2.addWeighted(overlay, 0.8, annotated_frame, 0.2, 0)
            
            y_pos = 110
            for line in response_text.split('\n'):
                if len(line) > 60:
                    cv2.putText(annotated_frame, line[:60] + "...", (20, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 255), 2)
                else:
                    cv2.putText(annotated_frame, line, (20, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 255), 2)
                y_pos += 30

        # Sentence Panel
        cv2.rectangle(annotated_frame, (0, h-100), (w, h), (30, 30, 30), -1)
        current_sentence = sentence_builder.get_built_sentence() or "..."
        cv2.putText(annotated_frame, current_sentence, (20, h-40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(annotated_frame, "USER SAYS:", (20, h-80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Detection Panel
        if num_hands > 0:
            hands_data = extractor.get_both_hands_with_fallback(hands_dict)
            landmarks = extractor.flatten_both_hands(hands_data)
            if isinstance(landmarks, list):
                landmarks = np.array(landmarks, dtype=np.float32)

            active_detector = None
            norm_method = "full"
            if sentence_builder.current_mode == InputMode.WORD and word_detector:
                active_detector = word_detector
                norm_method = word_norm_method
            elif sentence_builder.current_mode == InputMode.SPELLING and alphabet_detector:
                active_detector = alphabet_detector
                norm_method = alphabet_norm_method
            
            if active_detector:
                landmarks = normalize_landmarks(landmarks, method=norm_method)
                label, conf, metadata = active_detector.detect(landmarks) # Capture metadata
                
                # --- VISUALIZATION PHASE: PROGRESS BAR & TRAFFIC LIGHT ---
                if 'stability_progress' in metadata:
                    progress = metadata['stability_progress']
                    potential_label = metadata.get('potential_label')
                    
                    if 0 < progress < 1.0:
                        # Draw Progress Bar
                        bar_w, bar_h = 200, 20
                        bar_x, bar_y = w - 250, 100
                        
                        # Background
                        cv2.rectangle(annotated_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
                        # Fill
                        fill_w = int(bar_w * progress)
                        cv2.rectangle(annotated_frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 255, 255), -1) # Yellow
                        # Text
                        cv2.putText(annotated_frame, f"Reading: {potential_label}...", (bar_x, bar_y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                if label:
                    # 3. TRAFFIC LIGHT CONFIDENCE VISUALIZATION
                    tl_x, tl_y = w - 60, 50 
                    tl_color = (0, 0, 255) # Default Red
                    display_text = ""
                    
                    if conf > 0.8:
                        tl_color = (0, 255, 0) # Green (High Confidence)
                        display_text = f"{label}"
                        # Draw Solid Green Circle
                        cv2.circle(annotated_frame, (tl_x, tl_y), 20, tl_color, -1)
                    elif conf > 0.5:
                        tl_color = (0, 255, 255) # Yellow (Medium Confidence)
                        display_text = f"Mungkin: {label}?"
                        # Draw Solid Yellow Circle
                        cv2.circle(annotated_frame, (tl_x, tl_y), 20, tl_color, -1)
                    else:
                        tl_color = (0, 0, 255) # Red (Low Confidence)
                        # Draw Empty Red Circle
                        cv2.circle(annotated_frame, (tl_x, tl_y), 20, tl_color, 2)

                    # Draw Label Text next to Traffic Light
                    text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.putText(annotated_frame, display_text, (tl_x - text_size[0] - 30, tl_y + 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, tl_color, 2)

                    committed, content = sentence_builder.process_detection(label, conf)
                    if committed:
                        print(f"✅ Committed: {content}")
                        # Flash Checkmark
                        cv2.circle(annotated_frame, (tl_x, tl_y), 25, (255, 255, 255), 3)

        # --- LOGIC PHASE: AUTO COMMIT CHECK ---
        auto_committed, msg = sentence_builder.check_auto_commit()
        if auto_committed:
            print(f"⏳ {msg}")
            # Visual feedback khusus untuk auto-commit
            cv2.putText(annotated_frame, "TIMEOUT: WORD FINISHED", (w//2 - 150, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # --- INTEGRATION PHASE: AUTO AI TRIGGER ---
            # Ambil kata yang baru saja dicommit
            committed_word = msg.replace("Auto-committed: ", "")
            process_ai_response(committed_word)
            sentence_builder.clear() # Reset setelah dikirim

        # --- VISUALIZATION PHASE: COUNTDOWN TIMER ---
        remaining_time = sentence_builder.get_auto_commit_remaining_time()
        if remaining_time >= 0:
            # Color logic: Yellow default, Red if < 5s
            color = (0, 255, 255) if remaining_time > 5 else (0, 0, 255) 
            text = f"Auto-send: {int(remaining_time)}s"
            
            # Draw near the mode indicator or top center
            cv2.putText(annotated_frame, text, (150, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Optional: Progress circle or bar specifically for timer
            # (Just text is fine for now as per request)

        return annotated_frame

        return annotated_frame

    def process_ai_response(text_input: str):
        """Helper untuk memproses AI di thread terpisah"""
        nonlocal response_text
        if not text_input or text_input == "...":
            print("⚠️ Kalimat kosong")
            return

        print(f"🗣️  User says: '{text_input}'")
        print(f"🤖 AI is thinking...")
        response_text = "AI: Sedang berpikir..." # Feedback visual instan

        def worker():
            nonlocal response_text
            try:
                answer = groq_assistant.generate_chat_response(text_input)
                response_text = f"AI: {answer}"
                print(f"✅ AI Answers: {answer}")
                voice.speak(answer)
            except Exception as e:
                print(f"❌ AI Error: {e}")
                response_text = "AI: Maaf, ada gangguan."
        
        # Jalankan di background thread
        threading.Thread(target=worker, daemon=True).start()

    def key_handler(key: int) -> bool:
        nonlocal response_text, manual_input_active
        
        # --- LOGIKA INPUT MANUAL (TOMBOL 7) ---
        if manual_input_active:
            # Jika sedang mode manual, tombol apapun (kecuali kontrol) jadi huruf
            try:
                char = chr(key).upper()
                if char.isalnum(): # Hanya huruf/angka
                    if sentence_builder.current_mode == InputMode.SPELLING:
                        sentence_builder.current_spelling.append(char) # Tambah ke buffer ejaan
                    else:
                        sentence_builder.sentence_parts.append(char) # Tambah sebagai kata
                    print(f"⌨️  Manual Input: {char}")
            except:
                pass
            manual_input_active = False # Matikan mode manual setelah 1 huruf
            print("⌨️  Manual Mode OFF")
            return True

        # --- LOGIKA KONTROL UTAMA ---
        if key == ord('0'): return False # 0: QUIT
        
        elif key == 9: # TAB (ASCII 9) -> SWITCH MODE (Ganti Shift)
            new_mode = InputMode.SPELLING if sentence_builder.current_mode == InputMode.WORD else InputMode.WORD
            sentence_builder.set_mode(new_mode)
            print(f"🔄 Switched to {new_mode.name}")
                
        elif key == 32: # SPACE
            if sentence_builder.current_mode == InputMode.SPELLING:
                sentence_builder.force_commit_spelling()
                
        elif key == ord('8'): # 8 -> LOCAL BRAIN
            if sentence_builder.current_mode == InputMode.SPELLING:
                sentence_builder.commit_spelling()
            
            query = sentence_builder.get_built_sentence()
            if query and query != "...":
                print(f"🔍 Local Lookup: {query}")
                answer = brain.generate_response(query)
                response_text = f"Bot: {answer}"
                print(f"🤖 Answer: {answer}")
                voice.speak(answer)
            else:
                print("⚠️ Kalimat kosong")
        
        elif key == 13: # ENTER -> AI CHAT (GROQ)
            if sentence_builder.current_mode == InputMode.SPELLING:
                sentence_builder.commit_spelling()
            
            raw = sentence_builder.get_built_sentence()
            # Gunakan fungsi async baru
            process_ai_response(raw)
            sentence_builder.clear()

        elif key == ord('7'): # 7 -> MANUAL INPUT TRIGGER
            manual_input_active = True
            print("\n⌨️  MANUAL INPUT: TEKAN 1 HURUF/ANGKA DI KEYBOARD SEKARANG...")

        elif key == 8: # Backspace
            sentence_builder.backspace()
            
        elif key == ord('9'): # 9 -> CLEAR
            sentence_builder.clear()
            groq_assistant.clear_history() # Clear AI Context
            response_text = ""
            print("🧹 Sentence & AI History cleared")

        elif key == ord('f') or key == ord('F'): # F -> Toggle Fullscreen
            win_name = 'Sign Language Translator'
            current = cv2.getWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN)
            if current == cv2.WINDOW_FULLSCREEN:
                cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            else:
                cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            
        return True

    try:
        run_camera_loop(cap, 'Sign Language Translator', frame_processor, key_handler, fullscreen=True)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        print("\n👋 Goodbye!")

if __name__ == '__main__':
    main()
