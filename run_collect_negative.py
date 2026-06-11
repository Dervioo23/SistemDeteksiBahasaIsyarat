from data_collection.collect_negative_samples import collect_negative_samples

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  🚀 PENGUMPULAN SAMPEL LATAR BELAKANG / NEGATIF")
    print("  ✨ Ajarkan AI untuk memahami 'Keheningan' dan 'Kebisingan'")
    print("="*60)
    print("\n  💡 Tips Cepat:")
    print("     • Santai! Jadilah diri sendiri.")
    print("     • JANGAN membuat gestur bahasa isyarat tertentu.")
    print("     • Sertakan: Menggaruk, membetulkan kacamata, mengistirahatkan tangan.")
    print("     • Sertakan: Bingkai kosong (keluar dari kamera).")
    print("\n  🎯 Tujuan: Buat kelas 'Sampah' yang kuat untuk mengurangi positif palsu!")
    print("="*60 + "\n")
    
    collect_negative_samples(
        output_dir='dataset/words/_background',
        num_samples=30,
        frames_per_sample=30
    )
