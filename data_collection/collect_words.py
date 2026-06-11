import cv2
import os
import sys
import numpy as np
import time
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
    countdown,
    get_custom_word_input,
    add_word_to_config
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
    """Fungsi utama untuk pengumpulan data kata"""
    print("\n" + "="*60)
    print("    PENGUMPULAN DATA BAHASA ISYARAT - KATA LENGKAP")
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
    
    # Inisialisasi validator kualitas untuk kata (ambang batas lebih rendah untuk gestur dinamis)
    validator = QualityValidator(
        min_confidence=0.65,  # Ambang batas kualitas keseluruhan (65% - lebih longgar untuk kata)
        min_stability=0.75,  # Ambang batas stabilitas (75% - lebih longgar untuk gestur bergerak)
        baseline_dir='data_collection/quality_baselines',
        dynamic_gesture=True  # Aktifkan mode gestur dinamis - bobot stabilitas dikurangi!
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
    window_name = 'Data Collection - Words'
    cv2.namedWindow(window_name)
    # Kirim dimensi frame ke callback
    cv2.setMouseCallback(window_name, mouse_callback, 
                        {'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                         'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})
    
    print_instructions()
    
    # Ambil kosakata
    words = config['vocabulary']['words']
    current_word_idx = 0
    target_samples = config['collection']['samples_per_gesture']
    
    # Hitung sampel yang ada untuk partisipan ini
    samples_collected = {}
    for word in words:
        word_dir = os.path.join(config['dataset']['words_dir'], word)
        existing_count = count_existing_samples(word_dir, participant_data['participant_id'], word)
        samples_collected[word] = existing_count
    
    recording = False
    recorded_landmarks = []
    frame_count = 0
    sequence_length = config['collection']['sequence_length_word']
    show_stats = False
    session_stats = {'accepted': 0, 'rejected': 0, 'total_quality': 0.0, 'frame_qualities': []}
    recording_start_time = None
    frame_quality_scores = []
    
    # Variabel deteksi timeout
    hands_lost_time = None
    TIMEOUT_DURATION = 2.0  # Timeout 2 detik
    last_hands_detected = False
    
    print(f"\n  📝 Kata yang akan direkam: {', '.join(words)}")
    print(f"  🎯 Target: {target_samples} samples per kata")
    print(f"  📏 Sequence length: {sequence_length} frames")
    print(f"  ✨ Quality validation: ENABLED (per-frame, threshold: 65%)")
    print(f"  🎨 Enhanced visualization: ENABLED (color-coded landmarks)")
    print(f"  ⏰ Timeout detection: ENABLED ({TIMEOUT_DURATION}s auto-stop)")
    print(f"  🎹 Keyboard: SPACE=record, S=stats, N=next, P=prev, C=custom, Q=quit")
    
    # Tampilkan progres yang ada
    total_existing = sum(samples_collected.values())
    if total_existing > 0:
        print(f"\n  ♻️  Progress yang sudah ada:")
        for word, count in samples_collected.items():
            if count > 0:
                print(f"     - {word}: {count}/{target_samples} samples")
        print(f"  📊 Total existing: {total_existing} samples\n")
    else:
        print()
    
    current_word = words[current_word_idx]
    print(f"\n  ▶️  Kata saat ini: '{current_word.upper()}'")
    print(f"  📊 Progress: {samples_collected[current_word]}/{target_samples}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("  ❌ Error membaca frame dari kamera!")
            break
        
        # Cerminkan frame
        frame = cv2.flip(frame, 1)
        
        # Ekstrak landmark untuk kedua tangan
        hands_dict, annotated_frame, num_hands = extractor.extract_both_hands(frame)
        
        # Logika deteksi timeout
        current_time = time.time()
        hands_detected = num_hands > 0
        
        # Lacak kejadian tangan hilang/ditemukan
        if hands_detected and not last_hands_detected:
            # Tangan ditemukan lagi
            hands_lost_time = None
        elif not hands_detected and last_hands_detected:
            # Tangan baru saja hilang
            hands_lost_time = current_time
        elif not hands_detected and hands_lost_time is not None:
            # Tangan masih hilang - cek timeout
            time_lost = current_time - hands_lost_time
            if recording and time_lost > TIMEOUT_DURATION:
                print(f"\n  ⏰ TIMEOUT: Tangan hilang selama {time_lost:.1f}s - Menghentikan perekaman")
                # Berhenti merekam otomatis
                recording = False
                recorded_landmarks = []
                frame_count = 0
                recording_start_time = None
                frame_quality_scores = []
                session_stats['rejected'] += 1
        
        last_hands_detected = hands_detected
        
        # Gunakan visualisasi landmark yang ditingkatkan
        if hands_dict['right'] or hands_dict['left']:
            # Gambar landmark yang ditingkatkan dengan pewarnaan kualitas
            if hands_dict['right']:
                annotated_frame = visual_feedback.draw_enhanced_landmarks(
                    annotated_frame, hands_dict['right'], 'right', quality_scores
                )
            if hands_dict['left']:
                annotated_frame = visual_feedback.draw_enhanced_landmarks(
                    annotated_frame, hands_dict['left'], 'left', quality_scores
                )
        
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
            quality_passed, quality_scores, quality_feedback = validator.validate_sample(hand_lm, current_word, frame)
        
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
        
        # Kotak status (lebih tinggi saat merekam atau peringatan timeout)
        timeout_warning = not hands_detected and hands_lost_time is not None
        timeout_remaining = TIMEOUT_DURATION - (current_time - hands_lost_time) if timeout_warning else 0
        
        status_height = 240 if recording else (210 if timeout_warning else (180 if quality_scores else 150))
        cv2.rectangle(annotated_frame, (10, 10), (w-10, status_height), (0, 0, 0), -1)
        cv2.rectangle(annotated_frame, (10, 10), (w-10, status_height), (255, 255, 255), 2)
        
        # Info teks
        cv2.putText(annotated_frame, f"Kata: {current_word.upper()}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Progress: {samples_collected[current_word]}/{target_samples}", 
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
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
        
        if recording:
            # Indikator perekaman
            cv2.putText(annotated_frame, f"REC [{frame_count}/{sequence_length}]", 
                        (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Tampilkan kecepatan (frame per detik)
            if recording_start_time:
                elapsed = time.time() - recording_start_time
                if elapsed > 0:
                    current_fps = frame_count / elapsed
                    target_duration = sequence_length / config['collection']['fps']
                    cv2.putText(annotated_frame, f"Speed: {current_fps:.1f} fps (target: ~{target_duration:.1f}s)", 
                                (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Peringatan timeout saat merekam
            if timeout_warning and timeout_remaining > 0:
                warning_color = (0, 0, 255) if timeout_remaining < 1.0 else (0, 165, 255)
                cv2.putText(annotated_frame, f"⚠️ TIMEOUT in {timeout_remaining:.1f}s", 
                            (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, warning_color, 2)
        else:
            # Status saat tidak merekam
            if timeout_warning and timeout_remaining > 0:
                # Tampilkan hitung mundur timeout
                warning_color = (0, 0, 255) if timeout_remaining < 1.0 else (0, 165, 255)
                cv2.putText(annotated_frame, f"⚠️ Hands lost - Auto-stop in {timeout_remaining:.1f}s", 
                            (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, warning_color, 2)
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
        btn_color = (0, 0, 255) if not recording else (100, 100, 100) # Merah jika siap, Abu-abu jika sibuk
        cv2.rectangle(annotated_frame, (btn_x1, btn_y1), (btn_x2, btn_y2), btn_color, -1)
        cv2.rectangle(annotated_frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (255, 255, 255), 2)
        
        btn_text = "REC" if not recording else "..."
        text_size = cv2.getTextSize(btn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = btn_x1 + (btn_x2 - btn_x1 - text_size[0]) // 2
        text_y = btn_y1 + (btn_y2 - btn_y1 + text_size[1]) // 2
        cv2.putText(annotated_frame, btn_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Logika perekaman
        if recording:
            if num_hands > 0:
                # Ambil kedua tangan dengan fallback ke nol untuk yang tidak terdeteksi
                hands_data = extractor.get_both_hands_with_fallback(hands_dict)
                # Ratakan ke array numpy
                hands_flat = extractor.flatten_both_hands(hands_data)
                recorded_landmarks.append(hands_flat)
                frame_count += 1
                
                # Lacak kualitas per-frame
                if quality_scores:
                    frame_quality_scores.append(quality_scores.get('overall', 0))
                    
                    # Lacak konsistensi untuk tangan mana pun yang aktif
                    active_hand_for_tracking = None
                    if hands_dict['right']:
                        active_hand_for_tracking = hands_dict['right']
                    elif hands_dict['left']:
                        active_hand_for_tracking = hands_dict['left']
                        
                    if active_hand_for_tracking:
                        hand_lm = np.array([[lm['x'], lm['y'], lm['z']] for lm in active_hand_for_tracking], dtype=np.float32)
                        validator.previous_landmarks.append(hand_lm)
                
                if frame_count >= sequence_length:
                    # Hitung rata-rata kualitas di semua frame
                    if frame_quality_scores:
                        avg_quality = np.mean(frame_quality_scores)
                        min_quality = np.min(frame_quality_scores)
                        quality_passed = avg_quality >= 0.65  # 65% threshold for words
                    else:
                        avg_quality = 0.0
                        min_quality = 0.0
                        quality_passed = False
                    
                    # Hitung durasi perekaman
                    recording_duration = time.time() - recording_start_time if recording_start_time else 0
                    target_duration = sequence_length / config['collection']['fps']
                    duration_diff = abs(recording_duration - target_duration)
                    
                    if quality_passed:
                        # Simpan data
                        word_dir = os.path.join(config['dataset']['words_dir'], current_word)
                        sample_num = get_next_sample_number(
                            word_dir, 
                            participant_data['participant_id'], 
                            current_word
                        )
                        
                        filename = f"{participant_data['participant_id']}_{current_word}_{sample_num:03d}.json"
                        filepath = os.path.join(word_dir, filename)
                        
                        # recorded_landmarks sudah berisi list (dari flatten_both_hands)
                        final_landmarks = recorded_landmarks  # Sudah list, tidak perlu konversi
                        
                        metadata = {
                            'participant_id': participant_data['participant_id'],
                            'session_id': participant_data['session_id'],
                            'label': current_word,
                            'category': 'words',
                            'frames': len(recorded_landmarks),
                            'format': 'both_hands',
                            'features_per_frame': 126,
                            'quality_score': float(avg_quality),
                            'min_frame_quality': float(min_quality),
                            'frame_quality_scores': [float(q) for q in frame_quality_scores],
                            'acceptance_threshold': 0.65,
                            'recording_duration': float(recording_duration),
                            'target_duration': float(target_duration),
                            'was_validated': True
                        }
                        
                        save_landmarks(final_landmarks, filepath, metadata)
                        
                        # Tambahkan ke manifest
                        add_to_manifest(config['dataset']['manifest_csv'], {
                            'file_id': filename.replace('.json', ''),
                            'participant_id': participant_data['participant_id'],
                            'session_id': participant_data['session_id'],
                            'category': 'words',
                            'label': current_word,
                            'frames': len(recorded_landmarks),
                            'file_path': filepath,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        samples_collected[current_word] += 1
                        session_stats['accepted'] += 1
                        session_stats['total_quality'] += avg_quality
                        
                        print(f"  ✅ Sample {samples_collected[current_word]} saved: {filename}")
                        print(f"     Quality: {avg_quality*100:.1f}% (min: {min_quality*100:.1f}%), Duration: {recording_duration:.2f}s")
                    else:
                        session_stats['rejected'] += 1
                        print(f"  ❌ Sample REJECTED - Quality too low: {avg_quality*100:.1f}% (threshold: 65%)")
                        print(f"     Min frame quality: {min_quality*100:.1f}%, Duration: {recording_duration:.2f}s")
                    
                    # Reset
                    recording = False
                    recorded_landmarks = []
                    frame_count = 0
                    recording_start_time = None
                    frame_quality_scores = []
                    
                    # Cek apakah target tercapai
                    if samples_collected[current_word] >= target_samples:
                        print(f"\n  🎉 Target tercapai untuk kata '{current_word}'!")
                        
                        # Pindah ke kata berikutnya
                        if current_word_idx < len(words) - 1:
                            current_word_idx += 1
                            current_word = words[current_word_idx]
                            print(f"\n  ▶️  Kata berikutnya: '{current_word.upper()}'")
                            print(f"  📊 Progress: {samples_collected[current_word]}/{target_samples}")
                        else:
                            print("\n  🎊 SEMUA KATA SELESAI!")
                            print("\n  📊 Ringkasan:")
                            for word, count in samples_collected.items():
                                print(f"     {word}: {count} samples")
                            break
            else:
                print("  ⚠️  Tangan tidak terdeteksi, recording dibatalkan!")
                recording = False
                recorded_landmarks = []
                frame_count = 0
                recording_start_time = None
                frame_quality_scores = []
        
        # Tampilkan frame
        cv2.imshow(window_name, annotated_frame)
        
        # Kontrol keyboard
        key = cv2.waitKey(1) & 0xFF
        
        # Cek klik mouse (variabel global)
        global mouse_click_detected
        if mouse_click_detected:
            mouse_click_detected = False # Reset flag
            # Simulasikan logika tekan spasi
            if not recording and num_hands > 0:
                key = ord(' ') # Pemicu logika capture di bawah
        
        if key == ord('q'):
            print("\n  🛑 Pengumpulan data dihentikan.")
            break
        
        elif key == ord(' ') and not recording and num_hands > 0:
            print(f"\n  🎬 Memulai recording kata '{current_word}'...")
            countdown(config['collection']['countdown_seconds'])
            recording = True
            recorded_landmarks = []
            frame_count = 0
            recording_start_time = time.time()
            frame_quality_scores = []
        
        elif key == ord('n'):
            # Lewati ke kata berikutnya
            if current_word_idx < len(words) - 1:
                current_word_idx += 1
                current_word = words[current_word_idx]
                print(f"\n  ⏭️  Skip ke kata: '{current_word.upper()}'")
                print(f"  📊 Progress: {samples_collected[current_word]}/{target_samples}")
            else:
                print("\n  ℹ️  Sudah di kata terakhir!")
        
        elif key == ord('p'):
            # Kata sebelumnya
            if current_word_idx > 0:
                current_word_idx -= 1
                current_word = words[current_word_idx]
                validator.previous_landmarks = []  # Reset tracking
                print(f"\n  ⏮️  Kembali ke kata: '{current_word.upper()}'")
                print(f"  📊 Progress: {samples_collected[current_word]}/{target_samples}")
            else:
                print("\n  ℹ️  Sudah di kata pertama!")
        
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
        
        elif key == ord('c'):
            # Tambah kata kustom
            cv2.destroyAllWindows()  # Tutup jendela sementara untuk input bersih
            
            custom_word, save_to_config = get_custom_word_input()
            
            if custom_word:
                # Cek apakah kata sudah ada di daftar
                if custom_word in words:
                    print(f"\n  ⚠️  Kata '{custom_word}' sudah ada di vocabulary!")
                else:
                    # Tambahkan ke daftar sementara
                    words.append(custom_word)
                    samples_collected[custom_word] = 0
                    
                    # Simpan ke config jika diminta
                    if save_to_config:
                        if add_word_to_config(custom_word):
                            print(f"  ✅ Kata '{custom_word}' berhasil ditambahkan ke config.json!")
                        else:
                            print(f"  ⚠️  Kata '{custom_word}' sudah ada di config atau error saat menyimpan.")
                    else:
                        print(f"  ✅ Kata '{custom_word}' ditambahkan untuk session ini (temporary)")
                    
                    # Ganti ke kata baru
                    current_word_idx = len(words) - 1
                    current_word = words[current_word_idx]
                    print(f"\n  ➡️  Siap collect kata: '{current_word.upper()}'")
                    print(f"  📊 Progress: {samples_collected[current_word]}/{target_samples}")
            else:
                print("\n  ℹ️  Batal menambah kata baru.")
    
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
    print(f"\n  Per-word breakdown:")
    for word, count in samples_collected.items():
        if count > 0:
            print(f"    {word}: {count} samples")
    
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
