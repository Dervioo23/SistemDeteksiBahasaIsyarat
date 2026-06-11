import os
import json
import csv
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse


def load_manifest(manifest_path: str) -> List[Dict]:
    """Muat manifest CSV"""
    if not os.path.exists(manifest_path):
        return []
    
    samples = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)
    
    return samples


def analyze_progress(samples: List[Dict]) -> Dict:
    """Analisis kemajuan pengumpulan"""
    # Kelompokkan berdasarkan peserta
    by_participant = defaultdict(lambda: defaultdict(list))
    
    for sample in samples:
        participant_id = sample['participant_id']
        category = sample['category']
        label = sample['label']
        
        by_participant[participant_id][category].append(label)
    
    # Hitung statistik
    stats = {}
    for participant_id, categories in by_participant.items():
        stats[participant_id] = {
            'alphabet': {},
            'words': {},
            'total_samples': 0
        }
        
        # Hitung alfabet
        if 'alphabet' in categories:
            alphabet_counts = defaultdict(int)
            for letter in categories['alphabet']:
                alphabet_counts[letter] += 1
            stats[participant_id]['alphabet'] = dict(alphabet_counts)
        
        # Hitung kata
        if 'words' in categories:
            word_counts = defaultdict(int)
            for word in categories['words']:
                word_counts[word] += 1
            stats[participant_id]['words'] = dict(word_counts)
        
        # Total sampel
        stats[participant_id]['total_samples'] = (
            len(categories.get('alphabet', [])) + 
            len(categories.get('words', []))
        )
    
    return stats


def load_config(config_path: str = 'config.json') -> Dict:
    """Muat konfigurasi kosakata"""
    if not os.path.exists(config_path):
        return {
            'words': ['halo', 'terimakasih', 'tolong', 'maaf', 'permisi']
        }
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config.get('vocabulary', {})


def get_target_samples() -> int:
    """Dapatkan target sampel dari konfigurasi"""
    config_path = 'config.json'
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return config.get('collection', {}).get('samples_per_gesture', 15)
    return 15


def print_summary(stats: Dict, target_samples: int = 15):
    """Cetak ringkasan semua peserta"""
    print("\n" + "="*70)
    print("  📊 DATA COLLECTION PROGRESS - ALL PARTICIPANTS")
    print("="*70)
    
    if not stats:
        print("\n  ⚠️  No data collected yet!")
        return
    
    for participant_id in sorted(stats.keys()):
        participant_stats = stats[participant_id]
        
        print(f"\n  👤 PARTICIPANT: {participant_id}")
        print("  " + "-"*66)
        
        # Kemajuan alfabet
        alphabet_data = participant_stats['alphabet']
        if alphabet_data:
            total_alphabet = sum(alphabet_data.values())
            unique_letters = len(alphabet_data)
            print(f"  📝 ALPHABET: {total_alphabet} samples | {unique_letters} unique letters")
            
            # Tampilkan huruf teratas yang dikumpulkan
            top_letters = sorted(alphabet_data.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"     Top: ", end="")
            for letter, count in top_letters:
                status = "✅" if count >= target_samples else "🔄"
                print(f"{letter}={count}{status} ", end="")
            print()
        else:
            print(f"  📝 ALPHABET: No data yet")
        
        # Kemajuan kata
        words_data = participant_stats['words']
        if words_data:
            total_words = sum(words_data.values())
            unique_words = len(words_data)
            print(f"  💬 WORDS: {total_words} samples | {unique_words} unique words")
            
            # Tampilkan semua kata
            print(f"     ", end="")
            for word, count in sorted(words_data.items()):
                status = "✅" if count >= target_samples else "🔄"
                print(f"{word}={count}{status} ", end="")
            print()
        else:
            print(f"  💬 WORDS: No data yet")
        
        print(f"  📊 TOTAL: {participant_stats['total_samples']} samples")


def print_detailed(stats: Dict, participant_id: str, target_samples: int = 15):
    """Cetak kemajuan terperinci untuk peserta tertentu"""
    if participant_id not in stats:
        print(f"\n  ⚠️  No data found for participant '{participant_id}'")
        return
    
    participant_stats = stats[participant_id]
    
    print("\n" + "="*70)
    print(f"  📊 DETAILED PROGRESS - PARTICIPANT {participant_id}")
    print("="*70)
    
    # Rincian alfabet
    alphabet_data = participant_stats['alphabet']
    if alphabet_data:
        print(f"\n  📝 ALPHABET GESTURES:")
        print("  " + "-"*66)
        
        # Semua huruf A-Z
        all_letters = [chr(i) for i in range(ord('A'), ord('Z')+1)]
        
        incomplete = []
        complete = []
        not_started = []
        
        for letter in all_letters:
            count = alphabet_data.get(letter, 0)
            if count == 0:
                not_started.append(letter)
            elif count < target_samples:
                incomplete.append((letter, count))
            else:
                complete.append((letter, count))
        
        # Tampilkan yang selesai
        if complete:
            print(f"\n  ✅ COMPLETE ({len(complete)}):")
            for i in range(0, len(complete), 10):
                batch = complete[i:i+10]
                print(f"     ", end="")
                for letter, count in batch:
                    print(f"{letter}:{count} ", end="")
                print()
        
        # Tampilkan yang belum selesai
        if incomplete:
            print(f"\n  🔄 IN PROGRESS ({len(incomplete)}):")
            for i in range(0, len(incomplete), 8):
                batch = incomplete[i:i+8]
                print(f"     ", end="")
                for letter, count in batch:
                    remaining = target_samples - count
                    print(f"{letter}:{count}/{target_samples}(-{remaining}) ", end="")
                print()
        
        # Tampilkan yang belum dimulai
        if not_started:
            print(f"\n  ⭕ NOT STARTED ({len(not_started)}):")
            print(f"     {', '.join(not_started)}")
        
        # Ringkasan
        total_alphabet = sum(alphabet_data.values())
        progress_pct = (len(complete) / 26) * 100
        print(f"\n  📊 Alphabet Summary:")
        print(f"     Total samples: {total_alphabet}")
        print(f"     Progress: {len(complete)}/26 letters ({progress_pct:.1f}%)")
        print(f"     Remaining: {len(incomplete) + len(not_started)} letters")
    else:
        print(f"\n  📝 ALPHABET: No data collected yet")
        print(f"     Next: Start with letter 'A'")
    
    # Rincian kata
    words_data = participant_stats['words']
    vocab = load_config()
    target_words = vocab.get('words', ['halo', 'terimakasih'])
    
    if words_data or target_words:
        print(f"\n  💬 WORD GESTURES:")
        print("  " + "-"*66)
        
        complete_words = []
        incomplete_words = []
        not_started_words = []
        
        for word in target_words:
            count = words_data.get(word, 0)
            if count == 0:
                not_started_words.append(word)
            elif count < target_samples:
                incomplete_words.append((word, count))
            else:
                complete_words.append((word, count))
        
        # Tampilkan yang selesai
        if complete_words:
            print(f"\n  ✅ COMPLETE ({len(complete_words)}):")
            for word, count in complete_words:
                print(f"     {word}: {count}/{target_samples}")
        
        # Tampilkan yang belum selesai
        if incomplete_words:
            print(f"\n  🔄 IN PROGRESS ({len(incomplete_words)}):")
            for word, count in incomplete_words:
                remaining = target_samples - count
                print(f"     {word}: {count}/{target_samples} (need {remaining} more)")
        
        # Tampilkan yang belum dimulai
        if not_started_words:
            print(f"\n  ⭕ NOT STARTED ({len(not_started_words)}):")
            for word in not_started_words:
                print(f"     {word}: 0/{target_samples}")
        
        # Ringkasan
        if words_data:
            total_words = sum(words_data.values())
            progress_pct = (len(complete_words) / len(target_words)) * 100 if target_words else 0
            print(f"\n  📊 Words Summary:")
            print(f"     Total samples: {total_words}")
            print(f"     Progress: {len(complete_words)}/{len(target_words)} words ({progress_pct:.1f}%)")
            print(f"     Remaining: {len(incomplete_words) + len(not_started_words)} words")
    
    print(f"\n  📊 TOTAL SAMPLES: {participant_stats['total_samples']}")
    print("="*70)


def print_next_steps(stats: Dict, participant_id: str, target_samples: int = 15):
    """Sarankan langkah selanjutnya untuk pengumpulan data"""
    if participant_id not in stats:
        print(f"\n  💡 NEXT STEPS for NEW participant '{participant_id}':")
        print(f"     1. Run: python run_collect_alphabet.py")
        print(f"     2. Start with letter 'A'")
        print(f"     3. Collect {target_samples} samples per letter")
        return
    
    participant_stats = stats[participant_id]
    
    print(f"\n  💡 NEXT STEPS for '{participant_id}':")
    
    # Periksa alfabet
    alphabet_data = participant_stats['alphabet']
    all_letters = [chr(i) for i in range(ord('A'), ord('Z')+1)]
    
    next_alphabet = None
    for letter in all_letters:
        count = alphabet_data.get(letter, 0)
        if count < target_samples:
            next_alphabet = (letter, count)
            break
    
    if next_alphabet:
        letter, count = next_alphabet
        remaining = target_samples - count
        if count == 0:
            print(f"     📝 Start collecting letter '{letter}' (0/{target_samples})")
        else:
            print(f"     📝 Continue letter '{letter}' ({count}/{target_samples}, need {remaining} more)")
        print(f"        → python run_collect_alphabet.py")
    else:
        print(f"     ✅ All alphabet letters complete!")
    
    # Periksa kata
    words_data = participant_stats['words']
    vocab = load_config()
    target_words = vocab.get('words', ['halo', 'terimakasih'])
    
    next_word = None
    for word in target_words:
        count = words_data.get(word, 0)
        if count < target_samples:
            next_word = (word, count)
            break
    
    if next_word:
        word, count = next_word
        remaining = target_samples - count
        if count == 0:
            print(f"     💬 Start collecting word '{word}' (0/{target_samples})")
        else:
            print(f"     💬 Continue word '{word}' ({count}/{target_samples}, need {remaining} more)")
        print(f"        → python run_collect_words.py")
    else:
        if target_words:
            print(f"     ✅ All words complete!")


def main():
    parser = argparse.ArgumentParser(description='View data collection progress')
    parser.add_argument('--participant', '-p', type=str, help='Show detailed progress for specific participant')
    parser.add_argument('--detailed', '-d', action='store_true', help='Show detailed breakdown')
    parser.add_argument('--all', '-a', action='store_true', help='Show all participants detailed')
    
    args = parser.parse_args()
    
    # Muat data
    manifest_path = 'dataset/manifest.csv'
    samples = load_manifest(manifest_path)
    stats = analyze_progress(samples)
    target_samples = get_target_samples()
    
    if not samples:
        print("\n" + "="*70)
        print("  📊 DATA COLLECTION PROGRESS")
        print("="*70)
        print("\n  ⚠️  No data collected yet!")
        print("\n  💡 Start collecting:")
        print("     Alphabet: python run_collect_alphabet.py")
        print("     Words:    python run_collect_words.py")
        print("="*70 + "\n")
        return
    
    # Tampilkan hasil
    if args.participant:
        # Rincian untuk peserta tertentu
        print_detailed(stats, args.participant, target_samples)
        print_next_steps(stats, args.participant, target_samples)
    elif args.all:
        # Rincian untuk semua peserta
        for participant_id in sorted(stats.keys()):
            print_detailed(stats, participant_id, target_samples)
            print()
    elif args.detailed:
        # Ringkasan + rincian
        print_summary(stats, target_samples)
        print("\n  💡 TIP: Use --participant 001 for detailed breakdown")
    else:
        # Hanya ringkasan
        print_summary(stats, target_samples)
        print("\n  💡 TIP: Use --participant 001 for detailed breakdown")
    
    print()


if __name__ == '__main__':
    main()
