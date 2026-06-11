"""
Experimental Word Collection - Test Z coordinate removal
Versi eksperimental dari collect_words.py untuk menguji tanpa koordinat Z
"""
import os
 
from data_collection.collect_words import main as collect_words_main
from data_collection.landmark_extractor import LandmarkExtractor
from data_collection.quality_validator import QualityValidator, VisualFeedback
from data_collection.participant_manager import ParticipantManager
from data_collection.utils import load_config, create_directories
import cv2
import time
import numpy as np


def main():
    """Koleksi eksperimental utama dengan opsi koordinat Z"""
    print("\n" + "="*70)
    print("🧪 EXPERIMENTAL WORD COLLECTION")
    print("="*70)
    print("Testing word collection with different coordinate configurations")
    print("="*70)
    
    # Muat konfigurasi
    config = load_config()
    
    # Minta pengguna untuk mode eksperimental
    print("\n📊 Coordinate Configuration Options:")
    print("  [1] Standard mode (X, Y, Z coordinates) - 126 features")
    print("  [2] Experimental mode (X, Y only) - 84 features")  
    print("  [3] Comparison mode (collect both) - For analysis")
    
    while True:
        choice = input("\n🎯 Select mode (1-3): ").strip()
        if choice in ['1', '2', '3']:
            break
        print("❌ Invalid choice! Please enter 1, 2, or 3")
    
    if choice == '1':
        use_z = True
        experimental = False
        print("\n✅ Standard mode selected (X, Y, Z coordinates)")
    elif choice == '2':
        use_z = False
        experimental = True
        print("\n🧪 Experimental mode selected (X, Y coordinates only)")
    else:
        use_z = True
        experimental = True
        print("\n🔬 Comparison mode selected (will collect both formats)")
    
    # Inisialisasi dengan pengaturan eksperimental
    create_directories(config)
    
    # Manajemen peserta
    pm = ParticipantManager(config['dataset']['participants_csv'])
    participant_data = pm.show_menu()
    
    if not participant_data:
        print("\n❌ Collection cancelled.")
        return
    
    # Inisialisasi ekstraktor eksperimental
    extractor = LandmarkExtractor(
        min_detection_confidence=config['collection']['min_detection_confidence'],
        min_tracking_confidence=config['collection']['min_tracking_confidence'],
        max_num_hands=2,
        use_z_coordinate=use_z,
        experimental_mode=experimental
    )
    
    # Inisialisasi validator kualitas
    validator = QualityValidator(
        min_confidence=0.65,
        min_stability=0.75,
        baseline_dir='data_collection/quality_baselines',
        dynamic_gesture=True
    )
    visual_feedback = VisualFeedback()
    
    # Inisialisasi kamera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['collection']['resolution'][0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['collection']['resolution'][1])
    cap.set(cv2.CAP_PROP_FPS, config['collection']['fps'])
    
    if not cap.isOpened():
        print("❌ Error: Cannot open camera!")
        return
    
    # Tampilkan info eksperimental
    print(f"\n🧪 EXPERIMENTAL CONFIGURATION:")
    print(f"   - Z-coordinate: {'ENABLED' if use_z else 'DISABLED'}")
    print(f"   - Features per frame: {extractor.total_features}")
    print(f"   - Features per hand: {extractor.features_per_hand}")
    print(f"   - Experimental tracking: {'ON' if experimental else 'OFF'}")
    
    if choice == '3':
        print(f"   - Comparison mode: Will analyze Z stability during collection")
    
    # Variabel koleksi
    words = config['vocabulary']['words']
    current_word_idx = 0
    current_word = words[current_word_idx]
    target_samples = config['collection']['samples_per_gesture']
    
    # Pelacakan eksperimental
    z_stability_samples = []
    comparison_data = []
    
    print(f"\n📝 Words to collect: {', '.join(words)}")
    print(f"🎯 Target: {target_samples} samples per word")
    print(f"📏 Sequence length: {config['collection']['sequence_length_word']} frames")
    print("\n🎹 Controls:")
    print("   [SPACE] - Start recording")
    print("   [N] - Next word, [P] - Previous word")
    print("   [Q] - Quit")
    if experimental:
        print("   [A] - Show Z analysis (experimental mode)")
    
    # Loop koleksi
    recording = False
    recorded_landmarks = []
    frame_count = 0
    samples_collected = {word: 0 for word in words}
    
    print(f"\n▶️  Ready to collect: '{current_word.upper()}'")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error reading frame!")
            break
        
        frame = cv2.flip(frame, 1)
        
        # Ekstrak landmark
        hands_dict, annotated_frame, num_hands = extractor.extract_both_hands(frame)
        
        # Eksperimental: Mode perbandingan
        if choice == '3' and num_hands > 0:
            comparison = extractor.compare_with_without_z(frame)
            if comparison:
                comparison_data.append(comparison)
        
        # Validasi kualitas
        quality_passed = False
        quality_scores = {}
        quality_feedback = ""
        
        if hands_dict['right']:
            # Buat array berdasarkan mode koordinat
            if use_z:
                right_lm = np.array([[lm['x'], lm['y'], lm['z']] for lm in hands_dict['right']], dtype=np.float32)
            else:
                right_lm = np.array([[lm['x'], lm['y']] for lm in hands_dict['right']], dtype=np.float32)
            
            quality_passed, quality_scores, quality_feedback = validator.validate_sample(right_lm, current_word, frame)
        
        # Tampilkan info
        h, w, _ = annotated_frame.shape
        
        # Kotak status
        status_height = 200 if recording else 170
        cv2.rectangle(annotated_frame, (10, 10), (w-10, status_height), (0, 0, 0), -1)
        cv2.rectangle(annotated_frame, (10, 10), (w-10, status_height), (255, 255, 255), 2)
        
        # Info teks
        mode_text = f"Mode: {'Standard' if not experimental else 'Experimental'} ({extractor.total_features} features)"
        cv2.putText(annotated_frame, mode_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.putText(annotated_frame, f"Word: {current_word.upper()}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Samples: {samples_collected[current_word]}/{target_samples}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Deteksi tangan
        hand_status = f"Hands: {num_hands} detected"
        cv2.putText(annotated_frame, hand_status, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Skor kualitas
        if quality_scores:
            overall_score = quality_scores.get('overall', 0)
            quality_color = (0, 255, 0) if quality_passed else (0, 165, 255)
            cv2.putText(annotated_frame, f"Quality: {overall_score*100:.1f}%", 
                       (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)
        
        if recording:
            cv2.putText(annotated_frame, f"RECORDING [{frame_count}/{config['collection']['sequence_length_word']}]", 
                       (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Visualisasi landmark yang ditingkatkan
        if hands_dict['right'] or hands_dict['left']:
            if hands_dict['right']:
                annotated_frame = visual_feedback.draw_enhanced_landmarks(
                    annotated_frame, hands_dict['right'], 'right', quality_scores
                )
            if hands_dict['left']:
                annotated_frame = visual_feedback.draw_enhanced_landmarks(
                    annotated_frame, hands_dict['left'], 'left', quality_scores
                )
        
        # Umpan balik kualitas
        if quality_scores:
            annotated_frame = visual_feedback.draw_quality_indicator(
                annotated_frame, quality_scores, quality_feedback, position=(10, h - 180)
            )
        
        cv2.imshow("Experimental Word Collection", annotated_frame)
        
        # Logika perekaman
        if recording and num_hands > 0:
            hands_data = extractor.get_both_hands_with_fallback(hands_dict)
            hands_flat = extractor.flatten_both_hands(hands_data)
            recorded_landmarks.append(hands_flat)
            frame_count += 1
            
            if frame_count >= config['collection']['sequence_length_word']:
                # Simpan sampel (disederhanakan - hanya cetak untuk demonstrasi)
                print(f"✅ Sample {samples_collected[current_word] + 1} recorded with {len(hands_flat)} features per frame")
                samples_collected[current_word] += 1
                
                # Reset
                recording = False
                recorded_landmarks = []
                frame_count = 0
                
                # Periksa jika kata selesai
                if samples_collected[current_word] >= target_samples:
                    print(f"🎉 Word '{current_word}' completed!")
                    if current_word_idx < len(words) - 1:
                        current_word_idx += 1
                        current_word = words[current_word_idx]
                        print(f"▶️  Next word: '{current_word.upper()}'")
                    else:
                        print("🏁 All words completed!")
                        break
        
        # Tangani input keyboard
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Spasi - mulai merekam
            if num_hands > 0 and not recording:
                recording = True
                recorded_landmarks = []
                frame_count = 0
                print(f"🎬 Recording '{current_word}'...")
        
        elif key == ord('n'):  # Kata berikutnya
            if current_word_idx < len(words) - 1:
                current_word_idx += 1
                current_word = words[current_word_idx]
                print(f"▶️  Next word: '{current_word.upper()}'")
        
        elif key == ord('p'):  # Kata sebelumnya
            if current_word_idx > 0:
                current_word_idx -= 1
                current_word = words[current_word_idx]
                print(f"◀️  Previous word: '{current_word.upper()}'")
        
        elif key == ord('a') and experimental:  # Analisis
            z_analysis = extractor.get_z_stability_analysis()
            if z_analysis:
                print(f"\n📊 Z-COORDINATE ANALYSIS:")
                print(f"   - Stability Score: {z_analysis.get('stability_score', 0):.3f}")
                print(f"   - Samples Analyzed: {z_analysis.get('total_samples', 0)}")
        
        elif key == ord('q'):  # Keluar
            break
    
    # Pembersihan dan analisis
    cap.release()
    cv2.destroyAllWindows()
    
    # Tampilkan hasil eksperimental
    if experimental:
        print(f"\n🧪 EXPERIMENTAL RESULTS:")
        print(f"="*50)
        
        if choice == '3' and comparison_data:
            print(f"📊 Comparison Analysis:")
            consistent_detection = sum(1 for comp in comparison_data if comp.get('detection_consistency', False))
            total_comparisons = len(comparison_data)
            consistency_rate = (consistent_detection / total_comparisons) * 100 if total_comparisons > 0 else 0
            
            print(f"   - Detection Consistency: {consistency_rate:.1f}% ({consistent_detection}/{total_comparisons})")
            
            if comparison_data:
                avg_reduction = np.mean([comp.get('feature_reduction_percent', 0) for comp in comparison_data 
                                       if 'feature_reduction_percent' in comp])
                print(f"   - Average Feature Reduction: {avg_reduction:.1f}%")
    
    extractor.close()
    print(f"\n✅ Experimental collection completed!")
    print(f"📊 Total samples collected: {sum(samples_collected.values())}")


if __name__ == "__main__":
    main()
