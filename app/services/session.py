
import logging
import time
from app.services.inference import inference_service
from inference.sequence_handler import SequenceHandler
from inference.sentence_builder import SentenceBuilder, InputMode
from inference.response_engine import ResponseEngine
from inference.groq_client import groq_assistant
from inference.local_brain import LocalBrain
from app.core.config import settings

logger = logging.getLogger("app.services.session")

class SignLanguageSession:
    """
    Manages state for a single WebSocket connection.
    Handles: Mode switching, Spelling accumulation, Sentence building, and AI Chat.
    Mirrors the functionality of run_inference_multiclass.py for web.
    """
    
    # System status constants
    STATUS_LISTENING = "LISTENING"
    STATUS_SPEAKING = "SPEAKING"
    STATUS_THINKING = "THINKING"
    
    def __init__(self):
        # State components - with improved timeouts to reduce repeated detections
        self.sequence_handler = SequenceHandler(
            max_sequence_length=20,
            letter_timeout=5.0,
            word_complete_timeout=3.0
        )
        self.sentence_builder = SentenceBuilder(
            min_stability_duration=0.5,
            word_sequence_timeout=2.0
        )
        self.response_engine = ResponseEngine()
        self.brain = LocalBrain()  # Local dictionary responses
        
        # Mode: WORD or SPELLING (default to SPELLING since we have alphabet model)
        self.current_mode = InputMode.SPELLING
        self.sentence_builder.set_mode(self.current_mode)
        
        # Status tracking
        self.status = self.STATUS_LISTENING
        self.is_speaking = False
        
        # Committed log - tracks all committed letters/words
        self.committed_log = []
        self.max_log_entries = 20
        
        # Response text for display
        self.response_text = ""
        
        # Local state tracking
        self.last_spoken_letter = None
        self.last_spelled_word = None
        self.last_added_word_time = 0
        
    def process_frame_data(self, image_bytes: bytes):
        """
        Process frame and update state.
        Returns a dict with all UI updates including landmarks for visualization.
        """
        # 1. Run low-level detection
        result = inference_service.process_frame(image_bytes)
        
        if result.get("error"):
            return result

        prediction = result.get("prediction")
        confidence = result.get("confidence", 0.0)
        num_hands = result.get("num_hands", 0)
        landmarks = result.get("landmarks")
        metadata = result.get("metadata", {})
        
        # Build response with all UI data
        response_data = {
            # Detection data
            "prediction": prediction,
            "confidence": confidence,
            "num_hands": num_hands,
            "landmarks": landmarks,  # For frontend visualization
            
            # Status & Mode
            "status": self.status,
            "mode": self.current_mode.name,  # "WORD" or "SPELLING"
            
            # Spelling & Sentence
            "spelled_word": None,
            "sentence": self.sentence_builder.get_built_sentence(),
            
            # Progress & Metadata (for progress bar, traffic light)
            "stability_progress": metadata.get("stability_progress", 0),
            "potential_label": metadata.get("potential_label"),
            
            # Action & Audio
            "action": None,
            "audio_text": None,
            
            # Committed log
            "committed_log": self.committed_log[-10:],  # Last 10 entries
            
            # Response text (AI/Bot response)
            "response_text": self.response_text,
            
            # Auto-commit countdown
            "auto_commit_remaining": self.sentence_builder.get_auto_commit_remaining_time()
        }

        # 2. Update Sequence Handler (Spelling) - only in SPELLING mode
        if self.current_mode == InputMode.SPELLING:
            if prediction and prediction != 'not_C':
                if confidence > 0.7:
                    added = self.sequence_handler.add_letter(prediction)
                    if added:
                        current_word = self.sequence_handler.get_current_word()
                        response_data["spelling_update"] = current_word
                        
                        # Log the committed letter
                        self._add_to_log(f"Letter: {prediction}")
                        
                        if prediction != self.last_spoken_letter:
                            self.last_spoken_letter = prediction
        
        # 3. Process detection through SentenceBuilder (for both modes)
        if prediction and prediction != 'not_C' and confidence > 0.5:
            committed, content = self.sentence_builder.process_detection(prediction, confidence)
            if committed:
                self._add_to_log(f"Committed: {content}")
                response_data["committed"] = content

        # 4. Check Word Completion (Auto-commit)
        current_spelling = self.sequence_handler.get_current_word()
        response_data["spelled_word"] = current_spelling

        if self.sequence_handler.is_word_complete():
            finished_word = self.sequence_handler.complete_word()
            if finished_word:
                processed_word = self.response_engine.process_spelled_word(finished_word)
                self.sentence_builder.add_word(processed_word)
                response_data["sentence"] = self.sentence_builder.get_built_sentence()
                
                self._add_to_log(f"Word: {processed_word}")
                
                response_data["action"] = "speak_word"
                response_data["audio_text"] = self.response_engine.get_response_for_spelled_word(processed_word)

        # 5. Check auto-commit from SentenceBuilder
        auto_committed, msg = self.sentence_builder.check_auto_commit()
        if auto_committed:
            self._add_to_log(f"Auto: {msg}")
            response_data["auto_committed"] = msg
                
        return response_data
    
    def toggle_mode(self):
        """Toggle between WORD and SPELLING mode"""
        if self.current_mode == InputMode.WORD:
            self.current_mode = InputMode.SPELLING
        else:
            self.current_mode = InputMode.WORD
        
        self.sentence_builder.set_mode(self.current_mode)
        self._add_to_log(f"Mode: {self.current_mode.name}")
        
        return {
            "mode": self.current_mode.name,
            "message": f"Switched to {self.current_mode.name} mode"
        }
    
    def commit_spelling(self):
        """Commit current spelling to sentence (Space key)"""
        if self.current_mode == InputMode.SPELLING:
            self.sentence_builder.force_commit_spelling()
            word = self.sequence_handler.complete_word()
            if word:
                self.sentence_builder.add_word(word)
                self._add_to_log(f"Committed: {word}")
            return {"status": "spelling_committed", "sentence": self.sentence_builder.get_built_sentence()}
        return {"status": "not_in_spelling_mode"}
    
    def ask_local_brain(self):
        """Get response from local brain (8 key)"""
        self.status = self.STATUS_THINKING
        
        # Commit any pending spelling
        if self.current_mode == InputMode.SPELLING:
            self.sentence_builder.commit_spelling()
        
        query = self.sentence_builder.get_built_sentence()
        if not query or query == "...":
            return {"status": "empty", "message": "No sentence to process"}
        
        answer = self.brain.generate_response(query)
        self.response_text = f"Bot: {answer}"
        self._add_to_log(f"Bot: {answer[:30]}...")
        
        self.status = self.STATUS_SPEAKING
        
        return {
            "user_message": query,
            "bot_response": answer,
            "action": "speak_response",
            "audio_text": answer,
            "response_text": self.response_text
        }

    def finish_sentence(self):
        """
        Triggered when user clicks 'Finish' or hits ENTER.
        Sends sentence to AI and returns answer.
        """
        self.status = self.STATUS_THINKING
        
        # Commit any pending spelling
        if self.current_mode == InputMode.SPELLING:
            self.sentence_builder.commit_spelling()
        
        raw_sentence = self.sentence_builder.get_built_sentence()
        if not raw_sentence:
            self.status = self.STATUS_LISTENING
            return None
        
        self._add_to_log(f"User: {raw_sentence[:30]}...")
        
        # Clear builder
        self.sentence_builder.clear()
        
        # Get AI Response
        try:
            ai_answer = groq_assistant.generate_chat_response(raw_sentence)
            self.response_text = f"AI: {ai_answer}"
            self._add_to_log(f"AI: {ai_answer[:30]}...")
        except Exception as e:
            ai_answer = "Maaf, ada gangguan koneksi."
            self.response_text = f"AI: {ai_answer}"
            logger.error(f"AI Error: {e}")
        
        self.status = self.STATUS_SPEAKING
        
        return {
            "user_message": raw_sentence,
            "ai_response": ai_answer,
            "action": "speak_response",
            "audio_text": ai_answer,
            "response_text": self.response_text
        }
    
    def backspace(self):
        """Delete last character/word (Backspace key)"""
        self.sentence_builder.backspace()
        self.sequence_handler.remove_last_letter()
        return {
            "status": "deleted",
            "sentence": self.sentence_builder.get_built_sentence(),
            "spelled_word": self.sequence_handler.get_current_word()
        }

    def clear(self):
        """Clear all (9 key)"""
        self.sequence_handler.reset_sequence()
        self.sentence_builder.clear()
        self.response_text = ""
        self.committed_log = []
        self.status = self.STATUS_LISTENING
        groq_assistant.clear_history()
        
        return {
            "status": "cleared",
            "message": "Session cleared"
        }
    
    def set_speaking_done(self):
        """Called when TTS is done speaking"""
        self.status = self.STATUS_LISTENING
        self.is_speaking = False
        return {"status": self.status}
    
    def _add_to_log(self, entry: str):
        """Add entry to committed log"""
        timestamp = time.strftime("%H:%M:%S")
        self.committed_log.append(f"[{timestamp}] {entry}")
        
        # Trim log if too long
        if len(self.committed_log) > self.max_log_entries:
            self.committed_log = self.committed_log[-self.max_log_entries:]
