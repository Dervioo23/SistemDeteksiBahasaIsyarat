
import numpy as np
import cv2
import logging
import os
from pathlib import Path
from app.core.config import settings
from data_collection.landmark_extractor import LandmarkExtractor
from inference.multiclass_detector import MultiClassAlphabetDetector
from inference.utils import normalize_landmarks, get_normalization_method_for_model

logger = logging.getLogger("app.services.inference")

class InferenceService:
    def __init__(self):
        logger.info("Initializing InferenceService...")
        
        # Use same confidence as desktop version (0.7)
        self.extractor = LandmarkExtractor(
            min_detection_confidence=settings.COLLECTION_CONFIG.get("min_detection_confidence", 0.7),
            min_tracking_confidence=settings.COLLECTION_CONFIG.get("min_tracking_confidence", 0.5),
            max_num_hands=2
        )
        
        # Load Alphabet Model
        alphabet_model_path = settings.MODEL_CONFIG.get("multiclass_alphabet_model_path")
        # Fix path generic (relative to root)
        if not os.path.isabs(alphabet_model_path):
             alphabet_model_path = str(Path(os.getcwd()) / alphabet_model_path)
        
        self.alphabet_model_path = alphabet_model_path
        self.alphabet_detector = None
        self.alphabet_norm_method = "full"  # Default

        if os.path.exists(alphabet_model_path):
            logger.info(f"Loading Alphabet Model from {alphabet_model_path}")
            print(f"DEBUG: Loading Alphabet Model from {alphabet_model_path}")
            
            # CRITICAL: Get normalization method from model metadata (same as desktop)
            self.alphabet_norm_method = get_normalization_method_for_model(
                alphabet_model_path, 
                default="full"
            )
            print(f"DEBUG: Using normalization method: {self.alphabet_norm_method}")
            
            self.alphabet_detector = MultiClassAlphabetDetector(
                model_path=alphabet_model_path,
                class_names=settings.VOCABULARY.get("alphabet", []),
                confidence_threshold=settings.MODEL_CONFIG.get("confidence_threshold_alphabet", 0.6),
                stability_frames=5  # CRITICAL: Match desktop version for stable detection
            )
            print(f"DEBUG: Alphabet Model loaded successfully with stability_frames=5")
        else:
            logger.error(f"Alphabet model not found at {alphabet_model_path}")
            print(f"DEBUG: Alphabet model not found at {alphabet_model_path}")

    def process_frame(self, image_bytes: bytes):
        """
        Process raw image bytes, extract landmarks, and run inference.
        Returns landmark data for frontend visualization.
        """
        try:
            # Decode image
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                print("DEBUG: Failed to decode image frame")
                return {"error": "Failed to decode image"}
            
            # Flip frame horizontally to match what user sees (mirrored/selfie view)
            # Desktop does: cv2.flip(frame, 1) before processing
            # Since frontend CSS shows mirrored view, we flip here to match
            frame = cv2.flip(frame, 1)
            print(f"DEBUG: Frame processed, shape: {frame.shape}")
            
            frame_height, frame_width = frame.shape[:2]

            # Extract landmarks
            hands_dict, _, num_hands = self.extractor.extract_both_hands(frame)
            
            result = {
                "num_hands": num_hands,
                "prediction": None,
                "confidence": 0.0,
                "landmarks": None,  # For frontend visualization
                "metadata": {}
            }

            # ALWAYS extract landmarks for visualization when hands are detected
            if num_hands > 0:
                # Extract raw landmark points for visualization FIRST
                # Format: array of {x, y} for each hand (21 points per hand)
                # NOTE: hands_dict uses lowercase keys: 'right', 'left'
                # NOTE: hands_dict values are List[Dict] with 'x', 'y', 'z' keys
                landmark_points = []
                for hand_label in ['right', 'left']:  # FIXED: lowercase keys
                    if hand_label in hands_dict and hands_dict[hand_label] is not None:
                        hand_landmarks = hands_dict[hand_label]  # This is List[Dict]
                        hand_points = []
                        for lm in hand_landmarks:  # FIXED: no .landmark, it's already a list
                            # Convert normalized coordinates to percentage
                            px = lm['x'] * 100  # FIXED: dict access, not attribute
                            py = lm['y'] * 100
                            hand_points.append({"x": px, "y": py})
                        landmark_points.append({
                            "hand": hand_label,
                            "points": hand_points
                        })
                
                result["landmarks"] = landmark_points
                print(f"DEBUG: Landmarks sent: {len(landmark_points)} hands")
                
                # Now do detection if we have the detector
                if self.alphabet_detector:
                    hands_data = self.extractor.get_both_hands_with_fallback(hands_dict)
                    landmarks = self.extractor.flatten_both_hands(hands_data)
                    
                    if isinstance(landmarks, list):
                        landmarks = np.array(landmarks, dtype=np.float32)

                    # CRITICAL: Use same normalization method as desktop version
                    landmarks_normalized = normalize_landmarks(
                        landmarks, 
                        method=self.alphabet_norm_method
                    )
                    
                    label, conf, metadata = self.alphabet_detector.detect(landmarks_normalized)
                    
                    if label:
                        result["prediction"] = label
                        result["confidence"] = float(conf)
                        result["metadata"] = metadata

            return result

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return {"error": str(e)}

# Singleton instance
inference_service = InferenceService()
