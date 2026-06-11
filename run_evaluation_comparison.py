import os
import subprocess
import sys

def run_comparison():
    """Menjalankan evaluasi komparasi model Alphabet secara otomatis"""
    
    # Path model yang akan dibandingkan
    baseline_model = "trained_models/alphabet_baseline_model_final.keras"
    proposed_model = "trained_models/multiclass_alphabet_model_final.keras"
    data_dir = "preprocessed_data/alphabet"
    
    print("\n" + "="*60)
    print("SISTEM EVALUASI KOMPARASI MODEL ALPHABET")
    print("="*60)
    
    # Periksa keberadaan file
    missing_files = []
    if not os.path.exists(baseline_model): missing_files.append(baseline_model)
    if not os.path.exists(proposed_model): missing_files.append(proposed_model)
    if not os.path.exists(data_dir): missing_files.append(data_dir)
    
    if missing_files:
        print("\n[ERROR] File atau direktori berikut tidak ditemukan:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPastikan Anda sudah melatih model baseline dan default terlebih dahulu.")
        return

    # Siapkan environment agar mendukung karakter UTF-8 di terminal Windows
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    
    # Perintah untuk menjalankan skrip komparasi
    cmd = [
        sys.executable, "-m", "training.compare_models",
        "--models", baseline_model, proposed_model,
        "--data_dir", data_dir
    ]
    
    print(f"\n🚀 Menjalankan evaluasi pada data: {data_dir}...")
    print("-" * 60)
    
    try:
        # Jalankan proses
        result = subprocess.run(cmd, env=env, text=True)
        
        if result.returncode == 0:
            print("\n" + "="*60)
            print("✅ EVALUASI SELESAI!")
            print(f"Hasil lengkap tersimpan di: evaluation_results/model_comparison.md")
            print("="*60)
        else:
            print(f"\n❌ Terjadi kesalahan saat menjalankan evaluasi (Exit code: {result.returncode})")
            
    except Exception as e:
        print(f"\n❌ Gagal menjalankan skrip: {e}")

if __name__ == "__main__":
    run_comparison()
