import cv2
import os
import sys
import json
import numpy as np
from datetime import datetime
from .participant_manager import ParticipantManager
from .landmark_extractor import LandmarkExtractor
from .quality_validator import QualityValidator, VisualFeedback
from .utils import (
    load_config,
    create_directories,
    get_next_sample_number,
    count_existing_samples,
    save_landmarks,
    add_to_manifest,
    print_instructions,
    countdown
)


# Variabel global untuk klik mouse
mouse_click_detected = False

def mouse_callback(event, x, y, flags, param):
    """Menangani event mouse"""
    global mouse_click_detected
    if event == cv2.EVENT_LBUTTONDOWN:
        # Cek apakah klik berada di dalam area tombol
        # Posisi tombol: (w-120, h-60) sampai (w-20, h-20)
        w, h = param['width'], param['height']
        btn_x1, btn_y1 = w - 120, h - 60
        btn_x2, btn_y2 = w - 20, h - 20
        
        if btn_x1 <= x <= btn_x2 and btn_y1 <= y <= btn_y2:
            mouse_click_detected = True

def main():
    """Fungsi utama untuk pengumpulan data abjad"""
    print("\n" + "="*60)
    print("    PENGUMPULAN DATA BAHASA ISYARAT - ABJAD A-Z")
    print("="*60)
    
    # Muat konfigurasi
    config = load_config()
    create_directories(config)
    
    # Inisialisasi manajer partisipan
    pm = ParticipantManager(config['dataset']['participants_csv'])
    
    # Menu manajemen partisipan
    participant_data = pm.show_menu()
    
    if not participant_data:
        print("\n  ❌ Pengumpulan data dibatalkan.")
        return
    
    # Inisialisasi ekstraktor landmark
    extractor = LandmarkExtractor(
        min_detection_confidence=config['collection']['min_detection_confidence'],
        min_tracking_confidence=config['collection']['min_tracking_confidence'],
        max_num_hands=2
    )
    
    # Inisialisasi validator kualitas
    validator = QualityValidator(
        min_confidence=0.7,  # Ambang batas kualitas keseluruhan (70%)
        min_stability=0.85,  # Ambang batas stabilitas (85%)
        baseline_dir='data_collection/quality_baselines'
    )
    visual_feedback = VisualFeedback()
    
    # Inisialisasi kamera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['collection']['resolution'][0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['collection']['resolution'][1])
    cap.set(cv2.CAP_PROP_FPS, config['collection']['fps'])
    
    if not cap.isOpened():
        print("  ❌ Error: Tidak bisa membuka kamera!")
        return
    
    # Siapkan jendela dan callback
    window_name = 'Data Collection - Alphabet'
    cv2.namedWindow(window_name)
    # Kirim dimensi frame ke callback
    cv2.setMouseCallback(window_name, mouse_callback, 
                        {'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                         'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})
    
    print_instructions()
    
    # Ambil abjad
    alphabet = config['vocabulary']['alphabet']
    current_letter_idx = 0
    target_samples = config['collection']['samples_per_gesture']
    
    # Hitung sampel yang ada untuk partisipan ini
    samples_collected = {}
    for letter in alphabet:
        letter_dir = os.path.join(config['dataset']['alphabet_dir'], letter)
        existing_count = count_existing_samples(letter_dir, participant_data['participant_id'], letter)
        samples_collected[letter] = existing_count
    
    capturing = False
    capture_frames = 10  # Ambil 10 frame lalu rata-rata
    captured_landmarks = []
    show_stats = False
    session_stats = {'accepted': 0, 'rejected': 0, 'total_quality': 0.0}
    
    use_timer = True  # Default aktif
    
    print(f"\n  🔤 Abjad yang akan direkam: A-Z (26 huruf)")
    print(f"  🎯 Target: {target_samples} samples per huruf")
    print(f"  📸 Mode: Static pose (10 frames averaged)")
    print(f"  ✨ Quality validation: ENABLED (threshold: 70%)")
    print(f"  🎹 Keyboard: SPACE=capture, T=toggle timer, B=baseline, S=stats, N=next, P=prev, Q=quit")
    
    # Tampilkan progres yang ada
    total_existing = sum(samples_collected.values())
    if total_existing > 0:
        print(f"\n  ♻️  Progress yang sudah ada:")
        letters_with_progress = [f"{letter}:{count}" for letter, count in samples_collected.items() if count > 0]
        # Tampilkan dalam potongan 10
        for i in range(0, len(letters_with_progress), 10):
            chunk = letters_with_progress[i:i+10]
            print(f"     {', '.join(chunk)}")
        print(f"  📊 Total existing: {total_existing} samples\n")
    else:
        print()
    
    current_letter = alphabet[current_letter_idx]
    print(f"\n  ▶️  Huruf saat ini: '{current_letter}'")
    print(f"  📊 Progress: {samples_collected[current_letter]}/{target_samples}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("  ❌ Error membaca frame dari kamera!")
            break
        
        # Cerminkan frame
        frame = cv2.flip(frame, 1)
        
        # Ekstrak landmark untuk kedua tangan
        hands_dict, annotated_frame, num_hands = extractor.extract_both_hands(frame)
        
        # Validasi kualitas jika tangan terdeteksi
        quality_passed = False
        quality_scores = {}
        quality_feedback = ""
        
        active_hand_landmarks = None
        if hands_dict['right']:
            active_hand_landmarks = hands_dict['right']
        elif hands_dict['left']:
            active_hand_landmarks = hands_dict['left']
            
        if active_hand_landmarks:
            # Konversi ke array numpy untuk validasi
            hand_lm = np.array([[lm['x'], lm['y'], lm['z']] for lm in active_hand_landmarks], dtype=np.float32)
            quality_passed, quality_scores, quality_feedback = validator.validate_sample(hand_lm, current_letter, frame)
        
        # Tampilkan info
        h, w, _ = annotated_frame.shape
        
        # Tambahkan umpan balik visual kualitas
        if quality_scores and not show_stats:
            annotated_frame = visual_feedback.draw_quality_indicator(
                annotated_frame, 
                quality_scores,
                quality_feedback,
                position=(10, h - 200)  # Pojok kiri bawah
            )
        
        # Kotak status
        status_height = 180 if quality_scores else 150
        cv2.rectangle(annotated_frame, (10, 10), (w-10, status_height), (0, 0, 0), -1)
        cv2.rectangle(annotated_frame, (10, 10), (w-10, status_height), (255, 255, 255), 2)
        
        # Gambar indikator huruf besar
        cv2.putText(annotated_frame, current_letter, 
                    (w-150, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5)
        
        # Info teks
        cv2.putText(annotated_frame, f"Huruf: {current_letter}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Progress: {samples_collected[current_letter]}/{target_samples}", 
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Status timer
        timer_text = "TIMER: ON (3s)" if use_timer else "TIMER: OFF"
        timer_color = (0, 255, 255) if use_timer else (200, 200, 200)
        cv2.putText(annotated_frame, timer_text, 
                    (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, timer_color, 2)
        
        # --- TAMPILAN GAMBAR REFERENSI ---
        # Muat gambar referensi untuk huruf saat ini
        # Mapping: A=0.png, B=1.png, dll.
        try:
            ref_idx = alphabet.index(current_letter)
            ref_path = os.path.join("reference_images", f"{ref_idx}.png")
            
            if os.path.exists(ref_path):
                ref_img = cv2.imread(ref_path)
                if ref_img is not None:
                    # Ubah ukuran ke ukuran yang wajar (max 150px tinggi atau 150px lebar)
                    max_dim = 150
                    h_ref, w_ref = ref_img.shape[:2]
                    scale = min(max_dim / h_ref, max_dim / w_ref)
                    
                    target_w = int(w_ref * scale)
                    target_h = int(h_ref * scale)
                    
                    ref_img = cv2.resize(ref_img, (target_w, target_h))
                    
                    # Overlay di pojok kanan atas dengan padding
                    bg_pad = 5
                    # Pastikan tidak menimpa teks kiri (asumsi teks memakan ~300px)
                    safe_x = 350 
                    
                    x_offset = w - target_w - 20
                    y_offset = 20
                    
                    if x_offset < safe_x:
                        x_offset = safe_x # Paksa ke kanan jika memungkinkan, atau tumpang tindih sedikit
                    
                    # Gambar kotak latar belakang
                    cv2.rectangle(annotated_frame, 
                                 (x_offset - bg_pad, y_offset - bg_pad), 
                                 (x_offset + target_w + bg_pad, y_offset + target_h + bg_pad), 
                                 (255, 255, 255), -1)
                    
                    # Overlay gambar
                    annotated_frame[y_offset:y_offset+target_h, x_offset:x_offset+target_w] = ref_img
                    
                    # Label
                    cv2.putText(annotated_frame, "REFERENSI", 
                               (x_offset, y_offset + target_h + 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        except Exception as e:
            pass
        # -------------------------------
        
        # Status deteksi tangan
        hand_status = f"Tangan: {num_hands} terdeteksi"
        if num_hands == 2:
            hand_status += " (L+R)"
        elif num_hands == 1:
            if hands_dict['right']:
                hand_status += " (R)"
            else:
                hand_status += " (L)"
        cv2.putText(annotated_frame, hand_status, 
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Tampilan skor kualitas
        if quality_scores:
            overall_score = quality_scores.get('overall', 0)
            quality_color = (0, 255, 0) if quality_passed else (0, 165, 255)
            cv2.putText(annotated_frame, f"Quality: {overall_score*100:.1f}%", 
                        (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)
        
        if capturing:
            cv2.putText(annotated_frame, f"CAPTURE [{len(captured_landmarks)}/{capture_frames}]", 
                        (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            status_text = "READY" if num_hands > 0 else "NO HANDS"
            color = (0, 255, 0) if num_hands > 0 else (0, 0, 255)
            cv2.putText(annotated_frame, status_text, 
                        (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Gambar Tombol Rekam
        # Posisi: Pojok Kanan Bawah (w-120, h-60) sampai (w-20, h-20)
        btn_x1, btn_y1 = w - 120, h - 60
        btn_x2, btn_y2 = w - 20, h - 20
        
        # Status visual tombol
        btn_color = (0, 0, 255) if not capturing else (100, 100, 100) # Merah jika siap, Abu-abu jika sibuk
        cv2.rectangle(annotated_frame, (btn_x1, btn_y1), (btn_x2, btn_y2), btn_color, -1)
        cv2.rectangle(annotated_frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (255, 255, 255), 2)
        
        btn_text = "REC" if not capturing else "..."
        text_size = cv2.getTextSize(btn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = btn_x1 + (btn_x2 - btn_x1 - text_size[0]) // 2
        text_y = btn_y1 + (btn_y2 - btn_y1 + text_size[1]) // 2
        cv2.putText(annotated_frame, btn_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Logika pengambilan gambar
        if capturing:
            if num_hands > 0:
                # Ambil kedua tangan dengan fallback ke nol
                hands_data = extractor.get_both_hands_with_fallback(hands_dict)
                # Ratakan ke array numpy untuk penyimpanan dan rata-rata
                hands_flat = extractor.flatten_both_hands(hands_data)
                captured_landmarks.append(hands_flat)
                
                # Lacak untuk konsistensi (tambahkan ke landmark sebelumnya)
                if hands_dict['right']:
                    right_lm = np.array([[lm['x'], lm['y'], lm['z']] for lm in hands_dict['right']], dtype=np.float32)
                    validator.previous_landmarks.append(right_lm)
                
                if len(captured_landmarks) >= capture_frames:
                    # Rata-rata landmark yang diambil
                    avg_landmarks_flat = np.mean(captured_landmarks, axis=0)
                    
                    # Ubah bentuk untuk mendapatkan tangan yang benar untuk validasi akhir
                    avg_landmarks_reshaped = avg_landmarks_flat.reshape(42, 3)
                    
                    # Tentukan tangan mana yang akan divalidasi
                    # Kita cek apakah slot tangan kanan (0-21) memiliki data non-nol
                    right_hand_data = avg_landmarks_reshaped[:21]
                    left_hand_data = avg_landmarks_reshaped[21:]
                    
                    # Cek sederhana: jika tangan kanan memiliki gerakan signifikan/nilai, gunakan itu. 
                    # Jika tidak, gunakan kiri.
                    if np.any(right_hand_data):
                        validation_hand = right_hand_data
                    else:
                        validation_hand = left_hand_data
                    
                    # Cek kualitas akhir
                    final_passed, final_scores, final_feedback = validator.validate_sample(validation_hand, current_letter, frame)
                    
                    if final_passed:
                        # Simpan data
                        letter_dir = os.path.join(config['dataset']['alphabet_dir'], current_letter)
                        sample_num = get_next_sample_number(
                            letter_dir, 
                            participant_data['participant_id'], 
                            current_letter
                        )
                        
                        filename = f"{participant_data['participant_id']}_{current_letter}_{sample_num:03d}.json"
                        filepath = os.path.join(letter_dir, filename)
                        
                        # Konversi array numpy ke list untuk serialisasi JSON
                        final_landmarks = [avg_landmarks_flat.tolist()]
                        
                        metadata = {
                            'participant_id': participant_data['participant_id'],
                            'session_id': participant_data['session_id'],
                            'label': current_letter,
                            'category': 'alphabet',
                            'frames': 1,
                            'format': 'both_hands',
                            'features_per_frame': 126,
                            'quality_score': final_scores.get('overall', 0),
                            'quality_metrics': final_scores,
                            'acceptance_threshold': 0.7,
                            'was_validated': True
                        }
                        
                        save_landmarks(final_landmarks, filepath, metadata)
                        
                        # Tambahkan ke manifest
                        add_to_manifest(config['dataset']['manifest_csv'], {
                            'file_id': filename.replace('.json', ''),
                            'participant_id': participant_data['participant_id'],
                            'session_id': participant_data['session_id'],
                            'category': 'alphabet',
                            'label': current_letter,
                            'frames': 1,
                            'file_path': filepath,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        samples_collected[current_letter] += 1
                        session_stats['accepted'] += 1
                        session_stats['total_quality'] += final_scores.get('overall', 0)
                        
                        print(f"  ✅ Sample {samples_collected[current_letter]} saved: {filename} (Quality: {final_scores.get('overall', 0)*100:.1f}%)")
                    else:
                        session_stats['rejected'] += 1
                        print(f"  ❌ Sample REJECTED - Quality too low: {final_scores.get('overall', 0)*100:.1f}% (threshold: 70%)")
                        print(f"     Issues: {final_feedback}")
                    
                    # Reset
                    capturing = False
                    captured_landmarks = []
                    
                    # Cek apakah target tercapai
                    if samples_collected[current_letter] >= target_samples:
                        print(f"\n  🎉 Target tercapai untuk huruf '{current_letter}'!")
                        
                        # Pindah ke huruf berikutnya
                        if current_letter_idx < len(alphabet) - 1:
                            current_letter_idx += 1
                            current_letter = alphabet[current_letter_idx]
                            print(f"\n  ▶️  Huruf berikutnya: '{current_letter}'")
                            print(f"  📊 Progress: {samples_collected[current_letter]}/{target_samples}")
                        else:
                            print("\n  🎊 SEMUA HURUF SELESAI!")
                            print("\n  📊 Ringkasan:")
                            total = 0
                            for letter, count in samples_collected.items():
                                print(f"     {letter}: {count} samples")
                                total += count
                            print(f"\n  📈 Total: {total} samples")
                            break
            else:
                print("  ⚠️  Tangan tidak terdeteksi, capture dibatalkan!")
                capturing = False
                captured_landmarks = []
        
        # Tampilkan frame
        cv2.imshow(window_name, annotated_frame)
        
        # Kontrol keyboard
        key = cv2.waitKey(1) & 0xFF
        
        # Cek klik mouse (variabel global)
        global mouse_click_detected
        if mouse_click_detected:
            mouse_click_detected = False # Reset flag
            # Simulasikan logika tekan spasi
            if not capturing and num_hands > 0:
                key = ord(' ') # Pemicu logika capture di bawah
        
        if key == ord('q'):
            print("\n  🛑 Pengumpulan data dihentikan.")
            break
        
        elif key == ord(' ') and not capturing and num_hands > 0:
            if use_timer:
                print(f"\n  📸 Bersiap... (Timer 3 detik)")
                # Gambar hitung mundur di layar alih-alih memblokir konsol
                # Kita perlu menangani hitung mundur secara berbeda jika ingin pembaruan UI
                # Untuk saat ini, tetap gunakan hitung mundur konsol sederhana tapi mungkin tambahkan overlay nanti
                countdown(config['collection']['countdown_seconds'])
            
            print(f"  📸 Capturing huruf '{current_letter}'...")
            capturing = True
            captured_landmarks = []
        
        elif key == ord('n'):
            # Lewati ke huruf berikutnya
            if current_letter_idx < len(alphabet) - 1:
                current_letter_idx += 1
                current_letter = alphabet[current_letter_idx]
                print(f"\n  ⏭️  Skip ke huruf: '{current_letter}'")
                print(f"  📊 Progress: {samples_collected[current_letter]}/{target_samples}")
            else:
                print("\n  ℹ️  Sudah di huruf terakhir!")
        
        elif key == ord('p'):
            # Huruf sebelumnya
            if current_letter_idx > 0:
                current_letter_idx -= 1
                current_letter = alphabet[current_letter_idx]
                validator.previous_landmarks = []  # Reset tracking
                print(f"\n  ⏮️  Kembali ke huruf: '{current_letter}'")
                print(f"  📊 Progress: {samples_collected[current_letter]}/{target_samples}")
            else:
                print("\n  ℹ️  Sudah di huruf pertama!")
        
        elif key == ord('b'):
            # Buat referensi baseline
            baseline_hand = None
            if hands_dict['right']:
                baseline_hand = hands_dict['right']
            elif hands_dict['left']:
                baseline_hand = hands_dict['left']
                
            if baseline_hand:
                hand_lm = np.array([[lm['x'], lm['y'], lm['z']] for lm in baseline_hand], dtype=np.float32)
                # Simpan baseline
                os.makedirs(validator.baseline_dir, exist_ok=True)
                baseline_path = os.path.join(validator.baseline_dir, f"{current_letter}.json")
                with open(baseline_path, 'w') as f:
                    json.dump(hand_lm.tolist(), f)
                # Muat ulang baseline
                validator.baselines = validator._load_baselines()
                print(f"  ✅ Baseline created for '{current_letter}'")
            else:
                print("  ⚠️  No hand detected for baseline!")
        
        elif key == ord('t'):
            # Toggle timer
            use_timer = not use_timer
            status = "ON" if use_timer else "OFF"
            print(f"\n  ⏱️  Timer: {status}")
        
        elif key == ord('s'):
            # Toggle tampilan statistik
            show_stats = not show_stats
            if show_stats:
                print("\n" + "="*60)
                print("  📊 SESSION STATISTICS")
                print("="*60)
                total_attempts = session_stats['accepted'] + session_stats['rejected']
                if total_attempts > 0:
                    session_rate = (session_stats['accepted'] / total_attempts) * 100
                    avg_session_quality = (session_stats['total_quality'] / session_stats['accepted'] * 100) if session_stats['accepted'] > 0 else 0
                    print(f"  Total attempts: {total_attempts}")
                    print(f"  Accepted: {session_stats['accepted']}")
                    print(f"  Rejected: {session_stats['rejected']}")
                    print(f"  Acceptance rate: {session_rate:.1f}%")
                    print(f"  Avg quality: {avg_session_quality:.1f}%")
                else:
                    print(f"  No samples validated yet")
                print("="*60 + "\n")
    
    # Pembersihan
    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    
    # Statistik akhir
    print("\n" + "="*60)
    print("  📊 SESSION SUMMARY")
    print("="*60)
    
    total_samples = sum(samples_collected.values())
    print(f"\n  Total samples collected: {total_samples}")
    print(f"\n  Per-letter breakdown:")
    for letter, count in samples_collected.items():
        if count > 0:
            print(f"    {letter}: {count} samples")
    
    total_attempts = session_stats['accepted'] + session_stats['rejected']
    if total_attempts > 0:
        session_rate = (session_stats['accepted'] / total_attempts) * 100
        avg_quality = (session_stats['total_quality'] / session_stats['accepted'] * 100) if session_stats['accepted'] > 0 else 0
        print(f"\n  Quality Statistics:")
        print(f"    ✅ Accepted: {session_stats['accepted']}")
        print(f"    ❌ Rejected: {session_stats['rejected']}")
        print(f"    📊 Acceptance rate: {session_rate:.1f}%")
        print(f"    ⭐ Average quality: {avg_quality:.1f}%")
    
    print("\n" + "="*60)
    print("  ✅ Data saved successfully!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
