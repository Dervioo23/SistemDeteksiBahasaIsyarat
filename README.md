# Sistem Deteksi Bahasa Isyarat

Sistem Deteksi Bahasa Isyarat adalah aplikasi deteksi bahasa isyarat berbasis Python yang menggabungkan MediaPipe, OpenCV, TensorFlow/Keras, FastAPI, WebSocket, dan text-to-speech. Proyek ini mendukung pengumpulan data gestur, preprocessing, training model, evaluasi, dan inferensi real-time melalui kamera.

## Fitur Utama

- Deteksi alfabet bahasa isyarat A-Z.
- Deteksi kata berbasis sequence, seperti `halo`, `terimakasih`, `namasaya`, `apakabar`, `selamatpagi`, dan `dervio`.
- Mode inferensi hybrid untuk berpindah antara deteksi kata dan alfabet.
- Web app FastAPI dengan streaming kamera melalui WebSocket.
- Pipeline dataset: collection, preprocessing, training, evaluation, dan inference.
- Dukungan TTS offline melalui `pyttsx3`, serta opsi ElevenLabs jika API key tersedia.
- Integrasi AI opsional melalui Groq untuk penyusunan atau penyempurnaan kalimat.

## Struktur Proyek

```text
.
├── app/                  # FastAPI app, WebSocket, static files, template UI
├── data_collection/      # Pengumpulan landmark gestur dari kamera
├── dataset/              # Dataset raw landmark alfabet dan kata
├── inference/            # Detector, sentence builder, TTS, dan AI bridge
├── models/               # Arsitektur model CNN/BiLSTM dan utilities
├── preprocessing/        # Normalisasi, split data, dan persiapan dataset
├── training/             # Training dan evaluasi model
├── trained_models/       # Model hasil training dan metadatanya
├── All_Testing/          # Script pengujian dan diagnostik
├── main.py               # Entry point web app
├── config.json           # Konfigurasi proyek
└── requirements.txt      # Dependensi Python
```

## Kebutuhan Sistem

- Python 3.10 atau lebih baru.
- Webcam aktif.
- Windows direkomendasikan untuk fitur TTS `pyttsx3`/`pywin32` yang ada di dependency.
- Ruang penyimpanan cukup untuk dataset, model, dan file hasil training.

## Instalasi

Clone repository, masuk ke folder proyek, lalu buat virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Salin konfigurasi environment:

```bash
copy .env.example .env
```

Isi `.env` jika fitur AI/TTS online ingin dipakai:

```env
GROQ_API_KEY=isi_api_key_groq_anda
ELEVENLABS_API_KEY=isi_api_key_elevenlabs_anda
```

Jangan commit file `.env`. File itu berisi rahasia, bukan dekorasi.

## Menjalankan Web App

```bash
python main.py
```

Buka browser ke:

```text
http://localhost:8000
```

Web app akan menggunakan kamera, mengirim frame ke backend melalui WebSocket, lalu menampilkan hasil deteksi.

## Menjalankan Inferensi Desktop

Untuk inferensi utama:

```bash
python run_inference.py
```

Untuk inferensi multiclass/hybrid:

```bash
python run_inference_multiclass.py
```

Pastikan path model di `config.json` sesuai dengan file yang tersedia di `trained_models/`.

## Pengumpulan Dataset

Pengumpulan alfabet:

```bash
python run_collect_alphabet.py
```

Pengumpulan kata:

```bash
python run_collect_words.py
```

Pengumpulan negative samples:

```bash
python run_collect_negative.py
```

Dataset disimpan di folder `dataset/` dan dicatat melalui manifest/participant metadata.

## Preprocessing

```bash
python run_preprocessing.py
```

Pipeline preprocessing menyiapkan data training, validation, dan test berdasarkan konfigurasi di `config.json`.

## Training Model

```bash
python run_training.py
```

Output training akan tersimpan di `trained_models/`, termasuk model `.keras`, metadata, grafik history, dan confusion matrix.

## Evaluasi

```bash
python run_evaluation_comparison.py
```

Hasil evaluasi dapat digunakan untuk membandingkan performa model baseline dan multiclass.

## Konfigurasi

Konfigurasi utama ada di `config.json`, termasuk:

- lokasi dataset;
- jumlah frame sequence;
- threshold confidence;
- path model;
- vocabulary kata dan alfabet;
- parameter UI;
- pilihan TTS.

API key tidak disimpan di `config.json`. Gunakan `.env` untuk `GROQ_API_KEY` dan `ELEVENLABS_API_KEY`.

## Catatan Keamanan

Repository ini sengaja mengabaikan file seperti `.env`, cache Python, log runtime, TensorBoard event files, dan audio TTS hasil generate. File semacam itu mudah membuat repo kotor, berat, dan berisiko membocorkan data sensitif.

Jika API key pernah masuk commit lama, rotate key di provider terkait. Menghapus dari file saat ini tidak otomatis membuat key lama aman.

## Lisensi

Belum ada lisensi eksplisit. Tambahkan file `LICENSE` sebelum repository dipakai untuk distribusi publik atau kolaborasi formal.
