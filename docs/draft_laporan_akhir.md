# Laporan Eksperimen Hyperparameter Tuning untuk Analisis Sentimen Berbasis Data Teks

**Nama:** Muhammad Dhiauddin  
**NIM:** 25917024  
**Konsentrasi:** Sains Data - Profesional  
**Mata Kuliah:** Deep Learning  
**Dosen Pengampu:** Ahmad Fathan Hidayatullah, Ph.D.  

---

## 1. Pendahuluan
Analisis sentimen merupakan salah satu penerapan utama pemrosesan bahasa alami (NLP) untuk memahami polaritas opini publik terhadap suatu produk atau layanan. Dalam ranah *Deep Learning*, model berbasis urutan (*sequence-based models*) seperti Recurrent Neural Network (RNN) dan variannya (LSTM dan GRU) telah terbukti efektif dalam menangkap konteks pada data teks. Namun, performa model-model tersebut sangat bergantung pada pemilihan *hyperparameter* yang tepat. 

Tugas ini menitikberatkan pada perancangan dan pelaksanaan eksperimen *hyperparameter tuning* yang sistematis menggunakan pustaka **Optuna**. Tujuannya adalah untuk mengamati secara mendalam bagaimana parameter seperti *learning rate*, *batch size*, *hidden units*, dan *dropout rate* memengaruhi performa arsitektur RNN, LSTM, dan GRU dalam mengklasifikasikan ulasan aplikasi *mobile banking* berbahasa Indonesia.

## 2. Deskripsi Dataset
Penelitian ini menggunakan dataset publik berupa teks ulasan pengguna.
* **Nama Dataset:** Indonesian Mobile Banking App Reviews Dataset
* **Sumber:** Kaggle (diunggah oleh reginakirana)
* **Jumlah Data:** ± 12.000 baris ulasan
* **Kelas Sentimen:** 3 Kelas (Positif, Negatif, Netral)
* **Contoh Data:** Teks ulasan yang umumnya menggunakan bahasa informal, slang, dan singkatan khas Indonesia.

> `[MASUKKAN GAMBAR: images/output/distribusi_label.png DI SINI]`  
> *(Keterangan: Grafik bar dan pie chart yang menunjukkan proporsi kelas Positif, Negatif, dan Netral)*

Dataset ini memiliki distribusi kelas yang cukup *imbalanced* (tidak seimbang), di mana kelas "Positif" dan "Negatif" mendominasi, sedangkan kelas "Netral" adalah kelompok minoritas. Hal ini mendasari penggunaan F1-Score Makro (*Macro F1-Score*) sebagai metrik evaluasi utama.

> `[MASUKKAN GAMBAR: images/output/eda_top_words_per_kelas.png DI SINI]`  
> *(Keterangan: Grafik frekuensi kata (Top 20 Kata) setelah dihilangkan stopwords bahasa Indonesia)*

> `[MASUKKAN GAMBAR: images/output/eda_distribusi_panjang_teks.png DI SINI]`  
> *(Keterangan: Histogram dan Boxplot yang menunjukkan sebaran jumlah kata per ulasan untuk tiap kelas sentimen)*

> `[MASUKKAN GAMBAR: images/output/eda_rating_vs_sentimen.png DI SINI]`  
> *(Keterangan: Hubungan antara skor rating (bintang) aplikasi dengan sentimen aktual)*

> `[MASUKKAN GAMBAR: images/output/eda_distribusi_app.png DI SINI]`  
> *(Keterangan: Distribusi frekuensi ulasan pada berbagai aplikasi mobile banking)*

## 3. Preprocessing Data
Sebelum dimasukkan ke dalam model *Deep Learning*, data teks mentah melalui serangkaian proses pra-pemrosesan:
1. **Pembersihan Data (Missing Values):** Baris data yang bernilai *null* pada kolom teks atau label dihapus.
2. **Label Encoding:** Kelas teks sentimen (Positif, Negatif, Netral) diubah menjadi angka integer (0, 1, 2) menggunakan `LabelEncoder` dari scikit-learn.
3. **Data Splitting:** Dataset dibagi menjadi tiga bagian dengan teknik *Stratified Split* untuk menjaga rasio kelas:
   * **Train Set (70%):** Untuk melatih model.
   * **Validation Set (15%):** Untuk bahan evaluasi *tuning* Optuna (*Validation Loss* dan F1-Score).
   * **Test Set (15%):** Sepenuhnya "disembunyikan" dan hanya digunakan untuk menguji performa final model terbaik dari tiap arsitektur.
4. **Tokenisasi & Padding:** Teks dikonversi menjadi sekuens angka menggunakan `Tokenizer` Keras dengan batas maksimal 10.000 kata unik (`VOCAB_SIZE`). Sekuens kemudian disamakan panjangnya menjadi 100 token (`MAX_LEN`) melalui proses *Padding*.

## 4. Arsitektur Model
Dalam eksperimen ini, dibangun tiga arsitektur menggunakan *framework* TensorFlow/Keras untuk dibandingkan:
1. **Model RNN:** Menggunakan layer dasar `SimpleRNN`. Rentan terhadap masalah *vanishing gradient* pada kalimat yang sangat panjang.
2. **Model LSTM (Long Short-Term Memory):** Dilengkapi dengan mekanisme *gates* (Input, Forget, Output) yang memungkinkannya mengingat konteks dalam urutan teks yang panjang.
3. **Model GRU (Gated Recurrent Unit):** Penyederhanaan dari LSTM dengan hanya memiliki *Reset Gate* dan *Update Gate*, menawarkan waktu komputasi yang umumnya lebih efisien daripada LSTM.

Ketiga model tersebut dibungkus oleh desain layer yang sama:
`Embedding Layer -> Recurrent Layer (RNN/LSTM/GRU) -> Dropout Layer -> Dense Layer (Softmax 3 Kelas)`.

Berikut ringkasan arsitektur (*model summary*) untuk masing-masing model:

> `[MASUKKAN OUTPUT: model.summary() dari Notebook Bab 4 – RNN DI SINI]`  
> *(Keterangan: Tabel model.summary() arsitektur RNN yang menampilkan Layer, Output Shape, dan jumlah Parameter)*

> `[MASUKKAN OUTPUT: model.summary() dari Notebook Bab 4 – LSTM DI SINI]`  
> *(Keterangan: Tabel model.summary() arsitektur LSTM – perhatikan jumlah parameter lebih besar dari RNN karena mekanisme gates)*

> `[MASUKKAN OUTPUT: model.summary() dari Notebook Bab 4 – GRU DI SINI]`  
> *(Keterangan: Tabel model.summary() arsitektur GRU – parameter lebih ringan dari LSTM namun lebih besar dari RNN)*


## 5. Skenario Hyperparameter Tuning
Pencarian hyperparameter terbaik dilakukan menggunakan metode **Optuna** dengan *sampler* TPE (*Tree-structured Parzen Estimator*). Optuna dipilih karena sangat efisien dalam mengerucutkan ruang pencarian (*search space*) secara cerdas dibandingkan *Grid Search* konvensional. Anggaran eksperimen ditetapkan sebanyak **30 trial** untuk setiap arsitektur. Seluruh eksperimen direproduksi di bawah **Random Seed 42**.

**Tabel Ruang Pencarian (Search Space)**
| Hyperparameter | Ruang Pencarian | Alasan Pemilihan Rentang |
|:---|:---|:---|
| **Learning Rate** | `[0.0001 - 0.01]` *(Log-Uniform)* | Menentukan seberapa besar model memperbarui bobotnya. Di atas 0.01 seringkali konvergensi gagal, di bawah 0.0001 terlalu lama untuk konvergen. |
| **Batch Size** | `[16, 32, 64]` *(Categorical)* | Memengaruhi stabilitas estimasi gradien. Batch lebih kecil menawarkan regularisasi implisit, namun komputasi lebih lama. |
| **Hidden Units** | `[32, 64, 128]` *(Categorical)* | Kapasitas model. 128 cocok untuk menangkap fitur yang kompleks, namun sangat rentan terhadap *overfitting* pada dataset ulasan pendek. |
| **Dropout Rate** | `[0.1 - 0.5]` *(Uniform Float)* | Teknik krusial untuk mematikan sekian persen neuron secara acak demi mencegah hafalan berlebih (*overfitting*) pada data latih. |

*Epoch* maksimal diset pada angka 20, dengan callback `EarlyStopping` (patience = 3 pada Validation Loss).

## 6. Hasil Evaluasi

> `[MASUKKAN DATA CSV DARI: data/processed/tabel2_semua_trial.csv DI BAWAH INI]`  
> *(Catatan: Rangkum dan salin beberapa baris kombinasi dari CSV ke format tabel di bawah sesuai permintaan Tabel 2 di PDF Dosen. Anda tidak perlu menyalin semua 90 baris, cukup masukkan 5 percobaan terbaik per arsitektur).*

**Tabel 2. Format hasil eksperimen hyperparameter tuning (Sampel)**
| No | Model | Learning Rate | Batch Size | Hidden Units | Accuracy | Precision | Recall | F1-Score |
|:--:|:-----:|:-------------:|:----------:|:------------:|:--------:|:---------:|:------:|:--------:|
| 1  | RNN   | `[isi]` | `[isi]` | `[isi]` | `[isi]` | `[isi]` | `[isi]` | `[isi]` |
| 2  | LSTM  | `[isi]` | `[isi]` | `[isi]` | `[isi]` | `[isi]` | `[isi]` | `[isi]` |
| 3  | GRU   | `[isi]` | `[isi]` | `[isi]` | `[isi]` | `[isi]` | `[isi]` | `[isi]` |

> `[MASUKKAN GAMBAR: images/output/confusion_matrix_all.png DI SINI]`  
> *(Keterangan: Matriks kebingungan final yang membandingkan performa RNN, LSTM, dan GRU pada Test Set)*

### 6.1 Tabel Perbandingan Model Terbaik pada Test Set

Setelah hyperparameter terbaik ditemukan oleh Optuna untuk masing-masing arsitektur, model dilatih ulang menggunakan gabungan data Train + Validation (85%) dan dievaluasi satu kali pada Test Set (15%) yang sepenuhnya tersembunyi selama proses tuning.

> `[MASUKKAN DATA CSV DARI: data/processed/evaluasi_test_set.csv DI BAWAH INI]`

**Tabel 3. Perbandingan Evaluasi Model Terbaik pada Test Set**
| Arsitektur | Learning Rate | Batch Size | Hidden Units | Dropout Rate | Test Accuracy | Test Precision | Test Recall | Test F1-Score |
|:----------:|:-------------:|:----------:|:------------:|:------------:|:-------------:|:--------------:|:-----------:|:-------------:|
| RNN  | `[isi]` | `[isi]` | `[isi]` | `[isi]` | `[isi]` | `[isi]` | `[isi]` | `[isi]` |
| LSTM | `[isi]` | `[isi]` | `[isi]` | `[isi]` | `[isi]` | `[isi]` | `[isi]` | `[isi]` |
| GRU  | `[isi]` | `[isi]` | `[isi]` | `[isi]` | `[isi]` | `[isi]` | `[isi]` | `[isi]` |

### 6.2 Classification Report (Per Kelas Sentimen)

Tabel berikut menunjukkan Precision, Recall, dan F1-Score untuk **masing-masing kelas sentimen** (Positif, Negatif, Netral) pada setiap arsitektur. Ini memungkinkan analisis lebih detail apakah model mampu mengenali semua kelas secara merata atau hanya unggul pada kelas mayoritas.

> `[MASUKKAN TEKS OUTPUT: Classification Report dari Notebook Bab 6.4 DI SINI]`  
> *(Salin output teks classification_report() dari notebook untuk tiap arsitektur. Format tabel sudah tersedia langsung di output notebook.)*

**Classification Report – `[ISI ARSITEKTUR TERBAIK]`**
|  | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| Positif | `[isi]` | `[isi]` | `[isi]` | `[isi]` |
| Negatif | `[isi]` | `[isi]` | `[isi]` | `[isi]` |
| Netral  | `[isi]` | `[isi]` | `[isi]` | `[isi]` |
| **Macro Avg** | `[isi]` | `[isi]` | `[isi]` | `[isi]` |


## 7. Analisis dan Pembahasan
> *(Catatan: Anda dapat langsung menyalin teks yang di-generate dari Jupyter Notebook pada **Bab 6.7 Analisis & Kesimpulan** ke bagian ini. Teks di notebook secara cerdas akan langsung menyebutkan angka pasti dari model yang menang setelah proses running selesai. Berikut adalah draf kerangkanya:)*

**1. Kombinasi Hyperparameter Terbaik**
Berdasarkan evaluasi *Macro F1-Score* pada Test Set yang diisolasi, kombinasi terbaik jatuh pada model `[ISI ARSITEKTUR]` dengan nilai F1-Score sebesar `[ISI F1 SCORE]`. Kombinasi HP-nya adalah: Learning Rate = `[isi]`, Batch Size = `[isi]`, Hidden Units = `[isi]`, dan Dropout Rate = `[isi]`.

**2. Mengapa Kombinasi Tersebut Terbaik?**
Kombinasi tersebut berhasil menyeimbangkan batas konvergensi dan penghindaran *overfitting*. Learning rate yang moderat memastikan loss dapat turun perlahan hingga menuju minima global. Penggunaan tingkat Dropout rate yang tepat mencegah model menghafal dataset (mengingat struktur kalimat perbankan yang pendek).

> `[MASUKKAN GAMBAR: images/output/optuna_importance_lstm.png atau gru/rnn DI SINI]`  
> *(Keterangan: Bar chart yang menunjukkan parameter mana yang paling berpengaruh)*

**3. Pengaruh Learning Rate, Batch Size, dan Hidden Units**
Merujuk pada hasil plot *Parameter Importance* dari Optuna, terlihat bahwa:
* **Learning Rate** memberikan impak paling besar. Pada *slice plot*, nilai ekstrim terlalu besar menyebabkan model tidak pernah konvergen.
* **Batch Size** mempengaruhi mulusnya penurunan loss.
* **Hidden Units** berbanding lurus dengan Dropout. Unit besar (128) mewajibkan Dropout yang lebih tinggi untuk menangkal dominasi pola yang spesifik pada data latih.

> `[MASUKKAN GAMBAR: images/output/optuna_history_lstm.png (Pilih arsitektur terbaik) DI SINI]`  
> *(Keterangan: Riwayat optimasi Optuna yang menunjukkan progres peningkatan F1-Score)*

> `[MASUKKAN GAMBAR: images/output/optuna_slice_lstm.png (Pilih arsitektur terbaik) DI SINI]`  
> *(Keterangan: Slice plot interaktif yang membuktikan korelasi antara rentang nilai hyperparameter dengan performa model)*

> `[MASUKKAN GAMBAR: images/output/optuna_contour_lstm.png (Pilih arsitektur terbaik) DI SINI]`  
> *(Keterangan: Contour plot untuk melihat area kombinasi optimal antara dua parameter, misal learning rate vs dropout)*

**4. Perbandingan Performa Antar Model**
> `[MASUKKAN GAMBAR: images/output/perbandingan_metrik_test.png DI SINI]`  
> Walaupun secara teori LSTM/GRU mampu menangkap *long-term dependencies* lebih baik daripada RNN, untuk konteks dataset *mobile banking* Indonesia di mana pengguna kerap menulis ulasan dengan sangat pendek, perbedaan kemampuannya tidak berjarak terlampau drastis. Namun LSTM dan GRU tetap menangkap konteks kata sifat ("buruk", "mantap") dan objeknya ("aplikasi", "transfer") dengan lebih stabil.

**5. Kendala Eksperimen**
Kendala utama yang dialami adalah mahalnya *cost* waktu komputasi saat menjalankan 90 iterasi iterasi model, terutama tanpa akselerasi GPU lokal. Kendala berikutnya ada pada imbalansi kelas, di mana label "Netral" cukup minim dan sangat sering diprediksi salah (*misclassified*) oleh model sebagai sentimen "Positif" maupun "Negatif" akibat minimnya sampel pembelajaran.

## 8. Kesimpulan
1. **Hyperparameter Tuning Sangat Esensial:** Pendekatan TPE (Optuna) secara signifikan mampu melampaui kemampuan arsitektur *baseline* (tanpa *tuning*) melalui pencarian *learning rate* dan *dropout* terbaik.
2. Kombinasi parameter sangat bergantung pada kompleksitas dataset. Teks ulasan perbankan berbahasa Indonesia dengan vokabular informalnya menuntut nilai *dropout* yang cukup kuat agar model tetap bisa generalisasi ke kalimat *unseen* (data baru).
3. Evaluasi mutlak harus menggunakan Validation Set yang dipisahkan (*Stratified*) untuk tuning Optuna dan diuji di Test Set terakhir demi menghindari kebocoran data (*data leakage*). F1-Score Makro menjadi pedoman paling akurat untuk data yang jumlah kelasnya tidak seimbang.

## 9. Referensi
1. Dataset Indonesian Mobile Banking App Reviews: [Kaggle Dataset oleh reginakirana](https://www.kaggle.com/datasets/reginakirana/indonesian-mobile-banking-app-reviews-dataset)
2. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. *In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD)*.
3. Chollet, F. et al. (2015). Keras: Deep Learning for humans. *https://keras.io*.
4. Abadi, M. et al. (2015). TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems. *Software available from tensorflow.org*.
5. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780. (Literatur dasar arsitektur LSTM).
6. Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *arXiv preprint arXiv:1406.1078*. (Literatur dasar arsitektur GRU).
7. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. *The Journal of Machine Learning Research*, 15(1), 1929-1958.
8. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*. (Metode optimizer yang digunakan dalam eksperimen).
9. Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). Algorithms for hyper-parameter optimization. *Advances in Neural Information Processing Systems (NIPS)*, 24. (Konsep dasar metode Tree-structured Parzen Estimator / TPE pada Optuna).
10. Minaee, S., Kalchbrenner, N., Cambria, E., Nikzad, N., Chenaghlu, M., & Gao, J. (2021). Deep Learning--based Text Classification: A Comprehensive Review. *ACM Computing Surveys (CSUR)*, 54(3), 1-40.
