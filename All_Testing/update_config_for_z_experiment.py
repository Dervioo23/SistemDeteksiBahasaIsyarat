"""
Configuration Updater untuk Z-Coordinate Experiment
Script untuk mengupdate config.json dengan opsi Z coordinate
"""
import json
import os
import sys
from typing import Dict


def load_config(config_path: str = "config.json") -> Dict:
    """Load konfigurasi dari file JSON"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_config(config: Dict, config_path: str = "config.json") -> None:
    """Save konfigurasi ke file JSON"""
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def update_config_for_z_experiment(remove_z_coordinate: bool = False):
    """
    Update config.json untuk Z coordinate experiment
    
    Args:
        remove_z_coordinate: True untuk menghilangkan Z coordinate (84 features),
                           False untuk menggunakan Z coordinate (126 features)
    """
    
    print("\n" + "="*60)
    print("⚙️  Z-COORDINATE EXPERIMENT CONFIG UPDATER")
    print("="*60)
    
    # Muat konfigurasi saat ini
    try:
        config = load_config()
        print("✅ Current config loaded successfully")
    except FileNotFoundError:
        print("❌ config.json not found!")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing config.json: {e}")
        return False
    
    # Tampilkan pengaturan saat ini
    print(f"\n📊 CURRENT MODEL SETTINGS:")
    current_word_shape = config.get('model', {}).get('word_input_shape', [45, 126])
    current_alphabet_shape = config.get('model', {}).get('alphabet_input_shape', [126])
    current_total_features = config.get('model', {}).get('total_features', 126)
    current_features_per_hand = config.get('model', {}).get('features_per_hand', 63)
    
    print(f"   - Word input shape: {current_word_shape}")
    print(f"   - Alphabet input shape: {current_alphabet_shape}")
    print(f"   - Total features: {current_total_features}")
    print(f"   - Features per hand: {current_features_per_hand}")
    
    # Hitung dimensi baru
    if remove_z_coordinate:
        new_total_features = 84  # 21 landmarks × 2 coordinates × 2 hands
        new_features_per_hand = 42  # 21 landmarks × 2 coordinates
        new_word_input_shape = [current_word_shape[0], 84]  # Keep sequence length, change features
        new_alphabet_input_shape = [84]
        mode_name = "WITHOUT Z-coordinate"
        coordinate_description = "X, Y only"
    else:
        new_total_features = 126  # 21 landmarks × 3 coordinates × 2 hands  
        new_features_per_hand = 63  # 21 landmarks × 3 coordinates
        new_word_input_shape = [current_word_shape[0], 126]
        new_alphabet_input_shape = [126]
        mode_name = "WITH Z-coordinate"
        coordinate_description = "X, Y, Z"
    
    print(f"\n🎯 PROPOSED CHANGES ({mode_name}):")
    print(f"   - Coordinates used: {coordinate_description}")
    print(f"   - Word input shape: {current_word_shape} → {new_word_input_shape}")
    print(f"   - Alphabet input shape: {current_alphabet_shape} → {new_alphabet_input_shape}")
    print(f"   - Total features: {current_total_features} → {new_total_features}")
    print(f"   - Features per hand: {current_features_per_hand} → {new_features_per_hand}")
    
    # Hitung dampak
    feature_reduction = current_total_features - new_total_features
    if feature_reduction != 0:
        reduction_percent = abs(feature_reduction / current_total_features) * 100
        if feature_reduction > 0:
            print(f"   - Feature reduction: {feature_reduction} features ({reduction_percent:.1f}% smaller)")
        else:
            print(f"   - Feature increase: {abs(feature_reduction)} features ({reduction_percent:.1f}% larger)")
    else:
        print(f"   - No dimensional changes")
    
    # Konfirmasi perubahan
    print(f"\n⚠️  IMPACT ASSESSMENT:")
    if remove_z_coordinate:
        print(f"   ✅ Reduced memory usage (33% less features)")
        print(f"   ✅ Faster processing (less computation)")
        print(f"   ✅ Potentially more stable (no depth noise)")
        print(f"   ⚠️  Models need retraining with new dimensions")
        print(f"   ⚠️  Existing trained models incompatible")
    else:
        print(f"   ✅ Full 3D information preserved")
        print(f"   ✅ Compatible with existing models")
        print(f"   ⚠️  Higher memory usage")
        print(f"   ⚠️  Potential depth instability")
    
    # Minta konfirmasi
    while True:
        confirm = input(f"\n❓ Apply these changes to config.json? (y/n): ").strip().lower()
        if confirm in ['y', 'yes']:
            break
        elif confirm in ['n', 'no']:
            print("❌ Configuration update cancelled.")
            return False
        else:
            print("❌ Please enter 'y' or 'n'")
    
    # Cadangkan konfigurasi asli
    backup_path = "config_backup.json"
    try:
        save_config(config, backup_path)
        print(f"💾 Backup saved to {backup_path}")
    except Exception as e:
        print(f"⚠️  Warning: Could not create backup: {e}")
    
    # Perbarui konfigurasi
    config['model']['word_input_shape'] = new_word_input_shape
    config['model']['alphabet_input_shape'] = new_alphabet_input_shape
    config['model']['total_features'] = new_total_features
    config['model']['features_per_hand'] = new_features_per_hand
    
    # Tambahkan pengaturan eksperimental
    if 'experimental' not in config:
        config['experimental'] = {}
    
    config['experimental']['use_z_coordinate'] = not remove_z_coordinate
    config['experimental']['coordinates_used'] = coordinate_description
    config['experimental']['feature_reduction_applied'] = remove_z_coordinate
    config['experimental']['original_features'] = 126
    config['experimental']['current_features'] = new_total_features
    
    # Tambahkan komentar/catatan
    config['experimental']['notes'] = [
        f"Configuration updated for Z-coordinate experiment",
        f"Mode: {mode_name}",
        f"Models will need retraining if dimensions changed",
        f"Use update_config_for_z_experiment.py to switch modes"
    ]
    
    # Simpan konfigurasi yang diperbarui
    try:
        save_config(config)
        print(f"✅ Configuration updated successfully!")
        print(f"📝 Experimental settings added to config['experimental'] section")
        
        print(f"\n📋 NEXT STEPS:")
        print(f"   1. Test data collection with: python run_collect_words_experimental.py")
        print(f"   2. Run Z-stability analysis: python test_z_coordinate_experiment.py")
        if feature_reduction != 0:
            print(f"   3. Retrain models with new input dimensions:")
            print(f"      - Word model: input shape {new_word_input_shape}")
            print(f"      - Alphabet model: input shape {new_alphabet_input_shape}")
        print(f"   4. Compare model performance between configurations")
        print(f"   5. To revert: python update_config_for_z_experiment.py --revert")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving config: {e}")
        return False


def revert_config():
    """Kembalikan konfigurasi dari cadangan"""
    backup_path = "config_backup.json"
    
    if not os.path.exists(backup_path):
        print(f"❌ Backup file {backup_path} not found!")
        return False
    
    try:
        backup_config = load_config(backup_path)
        save_config(backup_config)
        print(f"✅ Configuration reverted from backup successfully!")
        return True
    except Exception as e:
        print(f"❌ Error reverting config: {e}")
        return False


def show_current_config():
    """Tampilkan pengaturan terkait koordinat Z saat ini"""
    try:
        config = load_config()
        
        print(f"\n📊 CURRENT Z-COORDINATE CONFIGURATION:")
        print(f"="*50)
        
        # Pengaturan model
        model_config = config.get('model', {})
        print(f"Model Settings:")
        print(f"   - Word input shape: {model_config.get('word_input_shape', 'Not set')}")
        print(f"   - Alphabet input shape: {model_config.get('alphabet_input_shape', 'Not set')}")
        print(f"   - Total features: {model_config.get('total_features', 'Not set')}")
        print(f"   - Features per hand: {model_config.get('features_per_hand', 'Not set')}")
        
        # Pengaturan eksperimental
        exp_config = config.get('experimental', {})
        if exp_config:
            print(f"\nExperimental Settings:")
            print(f"   - Use Z coordinate: {exp_config.get('use_z_coordinate', 'Not set')}")
            print(f"   - Coordinates used: {exp_config.get('coordinates_used', 'Not set')}")
            print(f"   - Current features: {exp_config.get('current_features', 'Not set')}")
            
            notes = exp_config.get('notes', [])
            if notes:
                print(f"   - Notes:")
                for note in notes:
                    print(f"     • {note}")
        else:
            print(f"\nNo experimental settings found")
            
    except Exception as e:
        print(f"❌ Error reading config: {e}")


def main():
    """Fungsi utama"""
    if len(sys.argv) > 1:
        if sys.argv[1] == '--revert':
            revert_config()
        elif sys.argv[1] == '--show':
            show_current_config()
        elif sys.argv[1] == '--help':
            print(f"Usage:")
            print(f"  python {sys.argv[0]}              # Interactive mode")
            print(f"  python {sys.argv[0]} --revert     # Revert from backup")
            print(f"  python {sys.argv[0]} --show       # Show current config")
            print(f"  python {sys.argv[0]} --help       # Show this help")
        else:
            print(f"❌ Unknown argument: {sys.argv[1]}")
            print(f"Use --help for usage information")
    else:
        # Mode interaktif
        show_current_config()
        
        print(f"\n🎯 Z-COORDINATE EXPERIMENT OPTIONS:")
        print(f"   [1] Remove Z coordinate (126 → 84 features)")
        print(f"   [2] Keep Z coordinate (126 features)")
        print(f"   [3] Show current config only")
        print(f"   [4] Revert from backup")
        
        while True:
            choice = input(f"\nSelect option (1-4): ").strip()
            
            if choice == '1':
                update_config_for_z_experiment(remove_z_coordinate=True)
                break
            elif choice == '2':
                update_config_for_z_experiment(remove_z_coordinate=False)
                break
            elif choice == '3':
                print(f"✅ Configuration displayed above.")
                break
            elif choice == '4':
                revert_config()
                break
            else:
                print(f"❌ Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
