# Rencana Implementasi: Hyperparameter Tuning Analisis Sentimen (Lengkap Sesuai Rubrik)

Rencana ini telah dirombak dan disempurnakan berdasarkan pembacaan analitis mendalam terhadap seluruh poin pada lembar tugas `tugas-hyperparameter-tuning-analisis-sentimen.pdf`. Tidak ada satupun persyaratan dosen yang terlewat dalam rencana eksekusi ini.

## 1. Spesifikasi Eksperimen & Dataset (Memenuhi Poin 3)
*   **Dataset:** Indonesian Mobile Banking App Reviews Dataset (`dataset_perbankan.csv`).
*   **Sumber:** Kaggle (Publik).
*   **Karakteristik:** 12.000 data teks berbahasa Indonesia dengan 3 kelas label (Positif, Negatif, Netral).
*   **Framework:** TensorFlow / Keras (Memenuhi Poin 4).
*   **Arsitektur Model:** Eksperimen akan membangun fungsi fleksibel yang bisa memanggil **RNN, LSTM, dan GRU** untuk perbandingan komprehensif.

## 2. Protokol Eksperimen & Validasi (Memenuhi Poin 5 - KRUSIAL)
Dosen Anda sangat menekankan bahwa eksperimen harus valid, adil, dan bisa direproduksi. Oleh karena itu, kode akan diimplementasikan dengan protokol ketat:

1.  **Pemisahan Data (Mencegah *Data Leakage*):**
    *   Dataset akan dibagi menjadi 3 bagian: **Data Latih (Train)** 70%, **Data Validasi (Val)** 15%, dan **Data Uji (Test)** 15%.
    *   *Sesuai perintah dosen:* Optuna hanya akan menggunakan **Data Validasi** untuk mencari hyperparameter terbaik. **Data Uji** akan disembunyikan dan hanya dipakai satu kali di akhir untuk evaluasi final.
2.  **Anggaran Eksperimen (*Trial Budget*):**
    *   Optuna akan dibatasi sebanyak **30 trials per arsitektur model** untuk menyeimbangkan antara eksplorasi ekstensif dan batas waktu komputasi.
3.  **Reproduksibilitas (*Random Seed*):**
    *   Kode akan mengunci *seed* (misal: `SEED = 42`) pada Numpy, Python Random, dan TensorFlow di sel paling atas agar hasil tuning selalu sama setiap kali di-*run* ulang.

## 3. Ruang Pencarian Hyperparameter (Memenuhi Poin 5)
Menggunakan algoritma **Optuna (TPE)** dengan 4 kombinasi parameter wajib dan strategis:

1.  **Learning Rate** -> Log-uniform `[1e-4, 1e-2]` *(Wajib analisis)*
2.  **Batch Size** -> Pilihan `[16, 32, 64]` *(Wajib analisis)*
3.  **Jumlah Hidden Units** -> Pilihan `[32, 64, 128]` *(Wajib analisis)*
4.  **Dropout Rate** -> Uniform `[0.1, 0.5]` *(Mencegah overfitting teks)*
*Parameter Statis:* Jumlah Layer (1), Optimizer (Adam), Epoch (Keras Early Stopping).

## 4. Evaluasi & Analisis Hasil (Memenuhi Poin 6)
Kode *Jupyter Notebook* akan dirancang untuk secara otomatis memunculkan luaran evaluasi yang diwajibkan:
*   Fungsi otomatis pencetak metrik: **Accuracy, Precision, Recall, F1-score**.
*   Fungsi visualisasi **Confusion Matrix** menggunakan *Seaborn/Matplotlib*.
*   Kode khusus untuk men-*generate* **Tabel Hasil Eksperimen** yang mematuhi **Format Tabel 2** dari dokumen dosen.

## 5. Struktur Output Kode (Menyelaraskan dengan Poin 8 - Struktur Laporan)
Agar Anda sangat mudah memindahkan hasil dari kode ke dalam Laporan PDF Anda, *Jupyter Notebook* akan saya buat dengan struktur *Heading* Markdown yang sama persis dengan yang diminta dosen:

*   **Bab 1:** Pendahuluan & Import Library (Setting Random Seed)
*   **Bab 2:** Deskripsi Dataset (Membaca CSV, Visualisasi Distribusi Label)
*   **Bab 3:** Preprocessing Data (Cleansing, Label Encoding, Tokenization, Padding, Train/Val/Test Split)
*   **Bab 4:** Arsitektur Model (Definisi fungsi pembuatan RNN, LSTM, GRU)
*   **Bab 5:** Skenario Hyperparameter Tuning (Sistem Optuna & TPE)
*   **Bab 6:** Hasil Evaluasi (Print Tabel 2, Metrik, Confusion Matrix)
*   *(Bab 7, 8, 9 terkait Analisis dan Kesimpulan akan Anda tulis sendiri di PDF berdasarkan grafik luaran kode).*

## 6. Struktur Proyek dan Manajemen File
Proyek ini telah diorganisasikan dengan standar *Data Science* profesional agar kode, data, dan laporan tersusun rapi. Berikut adalah fungsi dari masing-masing folder yang telah dibuat di direktori Anda:

*   `app/` : Untuk menyimpan kode antarmuka aplikasi atau deployment model (opsional/tidak diwajibkan untuk tugas ini).
*   `config/` : Untuk file konfigurasi seperti parameter Optuna atau path direktori agar tidak di-hardcode.
*   `data/` : Folder utama untuk manajemen dataset.
    *   `data/raw/` : Tempat data mentah asli yang tidak boleh diubah (Di sinilah file `dataset_perbankan.csv` diletakkan).
    *   `data/processed/` : Tempat menyimpan data hasil *cleansing* atau *tokenized* yang siap dimasukkan ke model.
    *   `data/external/` : Data tambahan dari luar jika ada.
*   `docs/` : Tempat menyimpan dokumen referensi seperti lembar tugas (`tugas-hyperparameter-tuning-analisis-sentimen.pdf`) dan tempat Anda meletakkan file PDF Laporan Akhir nantinya.
*   `images/output/` : Direktori khusus untuk menyimpan hasil visualisasi (*Confusion Matrix*, grafik Optuna) yang nantinya akan di-copy ke PDF laporan.
*   `logs/` : File catatan (log) untuk memantau proses *training* dan pergerakan akurasi Optuna trials.
*   `models/` : Direktori untuk menyimpan bobot (*weights*) model terbaik yang sudah selesai dilatih (misal dalam format `.h5` atau `.keras`).
*   `notebooks/` : **Folder utama tempat eksekusi**. Di sinilah file *Jupyter Notebook* (`.ipynb`) eksperimen *hyperparameter tuning* akan kita buat dan dijalankan.
*   `src/` : Folder untuk menyimpan *script* Python `.py` pendukung yang dapat dipanggil oleh notebook (misal fungsi pembersih teks).
*   `test/` : Untuk *script unit testing* kualitas kode (opsional).
*   `README.md` : Dokumentasi utama proyek.

> [!IMPORTANT]
> **User Review Required**
> Rencana Implementasi telah diperbarui dengan Bab 6 yang menjelaskan Struktur Proyek dan posisi file-file Anda saat ini.
> Silakan tinjau kembali, dan klik **Proceed** jika dokumentasi ini sudah sempurna agar kita bisa masuk ke tahap penulisan kode di folder `notebooks/`.
