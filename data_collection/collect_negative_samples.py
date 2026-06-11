import cv2
import numpy as np
import json
import os
from datetime import datetime

from data_collection.landmark_extractor import LandmarkExtractor
from data_collection.utils import load_config


def collect_negative_samples(
    output_dir: str = 'dataset/words/_background',
    num_samples: int = 30,
    frames_per_sample: int = 30
):
    """
    Kumpulkan sampel latar belakang/negatif (Sampah/Bukan Isyarat)
    
    Args:
        output_dir: Direktori output
        num_samples: Jumlah sampel yang akan dikumpulkan
        frames_per_sample: Frame per sampel
    """
    
    print("\n" + "="*60)
    print("COLLECT BACKGROUND SAMPLES (SAMPAH/BUKAN ISYARAT)")
    print("="*60)
    print(f"\nTarget: {num_samples} samples")
    print(f"Frames per sample: {frames_per_sample}")
    print(f"Output: {output_dir}")
    
    # Buat direktori output
    os.makedirs(output_dir, exist_ok=True)

    # Muat konfigurasi untuk parameter detektor
    config = load_config()
    coll_cfg = config.get('collection', {})
    min_det_conf = coll_cfg.get('min_detection_confidence', 0.7)
    min_track_conf = coll_cfg.get('min_tracking_confidence', 0.5)

    # Inisialisasi ekstraktor menggunakan ambang batas kepercayaan dari config
    print("\nInitializing landmark extractor...")
    extractor = LandmarkExtractor(
        min_detection_confidence=min_det_conf,
        min_tracking_confidence=min_track_conf,
        max_num_hands=2
    )
    
    # Inisialisasi kamera
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("❌ Cannot open camera!")
        return
    
    print("\n" + "="*60)
    print("INSTRUCTIONS - LAKUKAN HAL INI:")
    print("="*60)
    print("\n1. DIAM (Tangan di bawah/di meja)")
    print("2. GARUK HIDUNG / KEPALA")
    print("3. BENARKAN KACAMATA / RAMBUT")
    print("4. GERAKAN ACAK (Bukan isyarat)")
    print("5. MAIN HP (Pura-pura)")
    print("\n⚠️ JANGAN lakukan isyarat 'HALO' atau kata lain!")
    print("\nControls:")
    print("  SPACE - Start recording sample")
    print("  Q - Quit")
    print("="*60 + "\n")
    
    sample_count = 0
    recording = False
    current_sequence = []
    
    try:
        while sample_count < num_samples:
            ret, frame = cap.read()
            if not ret:
                print("❌ Cannot read frame!")
                break
            
            # Balik untuk efek cermin
            frame = cv2.flip(frame, 1)
            
            # Ekstrak landmarks
            hands_dict, annotated_frame, num_hands = extractor.extract_both_hands(frame)
            
            # Gambar UI
            h, w = frame.shape[:2]
            
            # Status bar
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (10, 10), (w-10, 100), (0, 0, 0), -1)
            annotated_frame = cv2.addWeighted(annotated_frame, 0.7, overlay, 0.3, 0)
            
            # Hitungan sampel
            cv2.putText(annotated_frame, f"Samples: {sample_count}/{num_samples}", 
                       (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Tangan terdeteksi
            color = (0, 255, 0) if num_hands == 2 else (0, 165, 255)
            cv2.putText(annotated_frame, f"Hands: {num_hands}", 
                       (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Status perekaman
            if recording:
                progress = len(current_sequence)
                cv2.putText(annotated_frame, f"Recording: {progress}/{frames_per_sample}", 
                           (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Rekam frame (BAHKAN JIKA TIDAK ADA TANGAN - Background butuh frame kosong juga!)
                # Catatan: Logika asli membutuhkan tangan. Untuk background, kita mungkin INGIN frame kosong.
                # Tapi untuk menjaga konsistensi dengan ekstraktor, kita biasanya butuh landmarks.
                # Jika tidak ada tangan, kita rekam landmarks kosong.
                
                hands_data = extractor.get_both_hands_with_fallback(hands_dict)
                landmarks = extractor.flatten_both_hands(hands_data)
                current_sequence.append(landmarks)
                
                # Cek jika selesai
                if len(current_sequence) >= frames_per_sample:
                    # Simpan sampel
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"background_{sample_count+1:03d}_{timestamp}.json"
                    filepath = os.path.join(output_dir, filename)
                    
                    sample_data = {
                        'label': '_background',
                        'timestamp': timestamp,
                        'num_frames': len(current_sequence),
                        'landmarks': current_sequence
                    }
                    
                    with open(filepath, 'w') as f:
                        json.dump(sample_data, f)
                    
                    sample_count += 1
                    print(f"✅ Sample {sample_count}/{num_samples} saved: {filename}")
                    
                    # Reset
                    recording = False
                    current_sequence = []
            else:
                cv2.putText(annotated_frame, "Press SPACE to record", 
                           (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Instruksi
            cv2.putText(annotated_frame, "Do: Idle, Scratch, Random", 
                       (w-350, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(annotated_frame, "SPACE:Record | Q:Quit", 
                       (w-300, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Tampilkan frame
            cv2.imshow('Collect Background Samples', annotated_frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' ') and not recording:
                recording = True
                current_sequence = []
                print(f"\n🎬 Recording sample {sample_count+1}...")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        
        print("\n" + "="*60)
        print("COLLECTION COMPLETED")
        print("="*60)
        print(f"\n✅ Collected: {sample_count} samples")
        print(f"📁 Location: {output_dir}")
        
        if sample_count > 0:
            print("\n📝 Next steps:")
            print("1. Run preprocessing: python run_preprocessing.py")
            print("2. Train model: python run_training.py")
            print("3. Model will now understand 'Silence'!")
        
        print("\n" + "="*60)


if __name__ == '__main__':
    collect_negative_samples(
        output_dir='dataset/words/_background',
        num_samples=30,
        frames_per_sample=30
    )
