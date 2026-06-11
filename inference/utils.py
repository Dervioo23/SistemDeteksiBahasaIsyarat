import cv2
import numpy as np
import logging
import os
import json
from typing import Callable, Optional

from preprocessing.normalize import normalize_landmarks_full

logger = logging.getLogger(__name__)


def setup_camera(
    device_index: int = 0,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    try_additional_indices: bool = True,
) -> cv2.VideoCapture:
    indices = [device_index]
    if try_additional_indices:
        for idx in range(0, 3):
            if idx not in indices:
                indices.append(idx)

    last_cap: Optional[cv2.VideoCapture] = None
    tried = []

    for idx in indices:
        tried.append(idx)
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

        if cap.isOpened():
            if idx != device_index:
                logger.warning("Camera index %d not available, using index %d instead", device_index, idx)
            return cap

        logger.warning("Failed to open camera index %d", idx)
        cap.release()
        last_cap = cap

    logger.error("Unable to open any camera. Tried indices: %s", tried)

    fallback_cap = cv2.VideoCapture(device_index)
    return fallback_cap


def run_camera_loop(
    cap: cv2.VideoCapture,
    window_name: str,
    frame_processor: Callable[[np.ndarray], np.ndarray],
    key_handler: Optional[Callable[[int], bool]] = None,
    fullscreen: bool = False,
) -> None:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed = frame_processor(frame)
        cv2.imshow(window_name, processed)

        key = cv2.waitKey(1) & 0xFF
        if key_handler is not None:
            if not key_handler(key):
                break
        elif key == ord("q"):
            break


def normalize_landmarks(landmarks: np.ndarray, method: str = "full") -> np.ndarray:
    """Helper normalisasi landmark bersama.

    Args:
        landmarks: Array 1D atau 2D yang mewakili fitur landmark.
        method: "full" untuk normalize_landmarks_full (pergelangan tangan + skala),
                "zscore" untuk normalisasi mean/std per-frame.
    """
    if method == "full":
        # Cocokkan perilaku yang ada di HybridDetector: reshape ke (1, 126)
        landmarks_reshaped = landmarks.reshape(1, -1)
        normalized = normalize_landmarks_full(landmarks_reshaped)
        return normalized.flatten()

    if method == "zscore":
        # Normalisasi z-score per-frame yang mendukung representasi landmark
        # 3D (x,y,z) dan 2D (x,y).
        flat = landmarks.reshape(-1).astype(np.float64)
        n = flat.shape[0]

        if n % 3 == 0:
            num_dims = 3
        elif n % 2 == 0:
            num_dims = 2
        else:
            raise ValueError(
                f"Unsupported landmark feature length {n}; expected divisible by 2 or 3 "
                "for 2D or 3D coordinates."
            )

        coords = flat.reshape(-1, num_dims)
        mean = np.mean(coords, axis=0, keepdims=True)
        std = np.std(coords, axis=0, keepdims=True) + 1e-7
        normalized = (coords - mean) / std
        return normalized.astype(np.float32).flatten()

    raise ValueError(f"Unknown normalization method: {method}")


def get_normalization_method_for_model(model_path: str, default: str = "full") -> str:
    """Menyimpulkan metode normalisasi dari metadata JSON model.

    Mencari file JSON saudara bernama *_metadata.json dan membaca bidang
    "normalization_method" jika ada. Kembali ke *default*.
    """
    meta_path = (
        model_path.replace("_final.keras", "_metadata.json")
        .replace("_best.keras", "_metadata.json")
    )

    if not os.path.exists(meta_path):
        logger.warning(
            "Metadata file not found for model %s; using default normalization method '%s'",
            model_path,
            default,
        )
        return default

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Lebih suka bidang tingkat atas, kembali ke konfigurasi bersarang jika ada
        method = meta.get("normalization_method")
        if not method and isinstance(meta.get("config"), dict):
            method = meta["config"].get("normalization_method")

        if not method:
            logger.warning(
                "No normalization_method in metadata %s; using default '%s'",
                meta_path,
                default,
            )
            return default

        logger.info(
            "Using normalization method '%s' inferred from metadata %s",
            method,
            meta_path,
        )
        return str(method)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception(
            "Failed to read normalization_method from %s: %s; using default '%s'",
            meta_path,
            exc,
            default,
        )
        return default
