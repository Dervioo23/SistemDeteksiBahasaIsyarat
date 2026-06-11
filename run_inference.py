import os
import logging

from inference.hybrid_detector import HybridDetector
from data_collection.utils import load_config


logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Konfigurasi logging seluruh aplikasi dari config.json.

    Kembali ke konfigurasi konsol level INFO yang masuk akal jika ada yang gagal.
    """
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
    except Exception as exc:  # pragma: no cover - defensive fallback
        logging.basicConfig(level=logging.INFO)
        logger.exception("Failed to configure logging from config.json: %s", exc)


def main():
    """Peluncur utama"""
    configure_logging()

    print("\n" + "="*60)
    print("SIGN LANGUAGE DETECTION SYSTEM")
    print("="*60)
    print("\n🚀 Starting real-time detection system...")
    
    # Muat konfigurasi
    config = load_config()
    model_cfg = config.get("model", {})
    
    # Path model
    word_model_path = model_cfg.get("word_model_path", "trained_models/word_halo_model_best.keras")
    alphabet_model_path = model_cfg.get("alphabet_model_path", "trained_models/alphabet_C_model_best.keras")
    
    # Periksa apakah model ada
    word_exists = os.path.exists(word_model_path)
    alphabet_exists = os.path.exists(alphabet_model_path)
    
    if not word_exists:
        print(f"\n⚠️  Word model not found: {word_model_path}")
        print("   Train with: python training/train_word_model.py")
    
    if not alphabet_exists:
        print(f"\n⚠️  Alphabet model not found: {alphabet_model_path}")
        print("   Train with: python training/train_alphabet_model.py")
    
    if not word_exists and not alphabet_exists:
        print("\n❌ No trained models found!")
        print("\n📝 Please train models first:")
        print("   1. Preprocess data: python preprocessing/prepare_data.py")
        print("   2. Train word model: python training/train_word_model.py")
        print("   3. Train alphabet model: python training/train_alphabet_model.py")
        print("   4. Run this script again")
        return
    
    # Tentukan mode
    if word_exists and alphabet_exists:
        mode = 'hybrid'
        print("\n✅ Both models found - Running in HYBRID mode")
    elif word_exists:
        mode = 'word'
        print("\n✅ Word model found - Running in WORD mode")
    else:
        mode = 'alphabet'
        print("\n✅ Alphabet model found - Running in ALPHABET mode")
    
    # Inisialisasi dan jalankan detektor
    try:
        detector = HybridDetector(
            word_model_path=word_model_path,
            alphabet_model_path=alphabet_model_path,
            mode=mode
        )
        
        detector.run()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n👋 Goodbye!")


if __name__ == '__main__':
    main()
