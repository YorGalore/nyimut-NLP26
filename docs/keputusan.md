# Keputusan Proyek

Dokumen ini mencatat keputusan metodologis dan teknis yang diambil selama pengembangan proyek **Prediksi Peristiwa Geopolitik Global: Analisis Dampak Terhadap Nilai Tukar**. Setiap keputusan disertai alasan pemilihan serta alternatif yang dipertimbangkan agar proses penelitian dapat ditelusuri dan diaudit.

---

## 1. Pemilihan Sumber Data

### Keputusan
Menggunakan **CNBC** sebagai satu-satunya sumber berita geopolitik dan **JISDOR Bank Indonesia** sebagai sumber nilai tukar.

### Alasan
- CNBC menyediakan berita internasional yang relevan dengan peristiwa geopolitik dan memiliki keterkaitan dengan ekonomi serta pasar keuangan.
- Struktur artikel CNBC relatif konsisten sehingga mempermudah proses scraping dan preprocessing.
- JISDOR merupakan kurs referensi harian USD/IDR yang diterbitkan langsung oleh Bank Indonesia.
- Penggunaan satu sumber berita mengurangi kebutuhan penyelarasan skema dari beberapa sumber.
- Ketentuan tugas menetapkan Bank Indonesia sebagai sumber data nilai tukar.

### Alternatif yang Dipertimbangkan
Menggabungkan beberapa sumber berita untuk memperluas cakupan data.

### Alasan Tidak Dipilih
Penggabungan beberapa sumber akan menambah kompleksitas preprocessing, penyamaan format, deduplikasi lintas sumber, serta penyelarasan timestamp.

---

## 2. Pemilihan Metode Akuisisi CNBC

### Keputusan
Menggunakan **Playwright** untuk mengambil indeks artikel CNBC dari halaman archive.

### Alasan
Halaman archive CNBC menggunakan JavaScript untuk memuat dan merender elemen artikel. Playwright memungkinkan halaman dijalankan seperti browser sehingga elemen artikel dan tautannya dapat diidentifikasi.

### Alternatif yang Dipertimbangkan
Menggunakan HTTP request langsung atau melakukan scraping berdasarkan kategori politik.

### Alasan Tidak Dipilih
- Request langsung tidak selalu memperoleh elemen yang dirender secara dinamis.
- Halaman kategori politik CNBC tidak menyediakan seluruh artikel hingga periode yang dibutuhkan, terutama untuk artikel lama.
- Halaman archive memberikan pendekatan berbasis tanggal yang lebih sesuai dengan rentang waktu penelitian.

---

## 3. Pemilihan Strategi Filtering Berita

### Keputusan
Melakukan filtering dalam dua tahap:

1. Filtering berdasarkan judul.
2. Filtering berdasarkan konten artikel setelah full scraping.

### Alasan
Jumlah awal artikel sangat besar, yaitu **96.558 artikel**, sehingga mengambil isi penuh seluruh artikel sejak awal tidak efisien.

Filtering judul menggunakan lexicon geopolitik dapat mengurangi jumlah artikel yang perlu di-*fetch* sebelum proses scraping penuh.

### Alternatif yang Dipertimbangkan
Melakukan scraping isi penuh terhadap seluruh artikel kemudian melakukan klasifikasi relevansi.

### Alasan Tidak Dipilih
Pendekatan tersebut membutuhkan jauh lebih banyak request, waktu scraping, dan sumber daya komputasi.

---

## 4. Pemilihan Lexicon-Based Filtering

### Keputusan
Menggunakan **keyword/lexicon geopolitik** untuk tahap awal filtering judul.

### Alasan
- Sederhana dan mudah diaudit.
- Tidak membutuhkan model machine learning tambahan.
- Cepat diterapkan pada puluhan ribu artikel.
- Keyword dapat disesuaikan dengan ruang lingkup penelitian.

### Validasi
Metode diuji menggunakan audit manual terhadap **50 sampel acak** dan menghasilkan presisi sebesar **86%**.

### Catatan
Filtering berbasis lexicon digunakan sebagai tahap penyaringan awal, bukan sebagai klaim bahwa seluruh artikel yang lolos pasti relevan.

---

## 5. Pemilihan Sumber Utama Ekstraksi Artikel

### Keputusan
Menggunakan **JSON-LD sebagai sumber utama** untuk ekstraksi metadata dan isi artikel.

Urutan ekstraksi yang digunakan:

1. JSON-LD
2. Meta tag
3. CSS selector sebagai fallback

### Alasan
JSON-LD merupakan data terstruktur yang lebih stabil terhadap perubahan tampilan halaman dibandingkan selector CSS.

### Alasan Menggunakan Fallback
Struktur artikel CNBC dapat berbeda antara artikel lama dan baru. Oleh karena itu, meta tag dan CSS selector tetap disediakan untuk menangani artikel yang tidak lengkap pada lapisan sebelumnya.

---

## 6. Validasi CSS Selector

### Keputusan
Sebuah kandidat container artikel dari CSS selector hanya dianggap valid apabila mengandung minimal **3 elemen paragraf**.

### Alasan
Beberapa elemen halaman seperti widget "artikel terkait" dapat memiliki struktur yang menyerupai isi artikel.

Ambang minimal tiga paragraf digunakan untuk mengurangi risiko mengambil konten yang bukan merupakan badan artikel.

---

## 7. Penggunaan Checkpoint pada Scraping

### Keputusan
Menggunakan sistem checkpoint selama proses scraping dan menyimpan progres setiap **200 artikel**.

### Alasan
Scraping artikel merupakan proses yang panjang dan dapat terhenti karena error, koneksi, atau penghentian manual.

Checkpoint memungkinkan:
- proses dihentikan tanpa kehilangan seluruh progres;
- proses dilanjutkan tanpa mengulang artikel yang sudah selesai;
- hasil scraping dipulihkan dari checkpoint sebelumnya.

### Keputusan Tambahan
URL yang sudah selesai diproses dibaca kembali dari seluruh file checkpoint untuk menentukan artikel yang masih harus diproses.

---

## 8. Pemisahan Folder Checkpoint Produksi dan Validasi

### Keputusan
Eksperimen validasi menggunakan folder checkpoint yang terpisah dari proses produksi.

### Alasan
Pada akhir proses, seluruh file checkpoint dalam folder yang dirujuk akan digabungkan. Jika validasi menggunakan folder produksi, hasil scraping sebelumnya dapat ikut masuk ke dataset validasi.

### Parameter Isolasi
Tiga parameter disediakan:

- `--index-file`
- `--out-csv`
- `--ckpt-dir`

Keputusan ini memastikan eksperimen validasi dapat dijalankan tanpa mencemari hasil produksi.

---

## 9. Tidak Menggunakan `--limit-articles` untuk Validasi

### Keputusan
Parameter `--limit-articles` tidak digunakan untuk mengambil sampel validasi.

### Alasan
Parameter tersebut mengurutkan artikel berdasarkan `title_score` dan mengambil artikel dengan skor tertinggi.

Cara tersebut dapat menyebabkan sampel validasi terlalu banyak berisi artikel yang memiliki keyword geopolitik pada judul, sehingga tidak merepresentasikan populasi artikel secara keseluruhan.

### Keputusan Sampling Validasi
Sampel validasi menggunakan **random sampling terstratifikasi berdasarkan tahun**.

---

## 10. Stratified Sampling Berdasarkan Tahun

### Keputusan
Validasi konten menggunakan **210 artikel**, yaitu:

- 35 artikel per tahun
- 6 tahun
- 2021–2026
- `random_state=42`

### Alasan
Distribusi artikel dapat berbeda antar tahun. Stratifikasi memastikan setiap tahun dalam rentang penelitian tetap terwakili.

### Hasil
Sampling terstratifikasi menghasilkan tingkat keberhasilan fetch sebesar **93,4%**.

---

## 11. Penanganan Artikel Berbayar

### Keputusan
Artikel CNBC Pro dan Investing Club dikeluarkan dari korpus.

### Alasan
Konten berbayar tidak dapat diperoleh secara konsisten melalui scraping biasa.

URL artikel berbayar juga tidak selalu dapat dibedakan secara langsung dari artikel reguler.

### Sinyal yang Digunakan
Metadata `section` digunakan dengan kondisi yang mendeteksi:

- `pro:`
- `investing club`

Sinyal tersebut digunakan sebagai indikator utama artikel berbayar.

---

## 12. Penanganan Live Blog

### Keputusan
Artikel **live blog tidak dibuang** dari korpus.

### Alasan
Live blog tetap dapat mengandung informasi geopolitik yang relevan.

Selain itu, pipeline preprocessing menyediakan `description` sebagai fallback ketika `body` tidak tersedia. Dengan demikian, artikel live blog masih dapat menyediakan informasi teks yang dapat diproses.

### Indikator Live Blog
Deteksi menggunakan indikator seperti:

- tanda `;` pada judul;
- frasa seperti `"live updates"`.

---

## 13. Penanganan Zona Waktu

### Keputusan
Konversi zona waktu hanya dilakukan pada timestamp berita CNBC.

`published_utc` dikonversi menjadi `published_wib` menggunakan zona waktu:

`Asia/Jakarta`

### Alasan
Timestamp CNBC tersedia dalam UTC, sedangkan JISDOR berkaitan dengan waktu operasional pasar Indonesia.

JISDOR tidak dikonversi dari UTC karena data tersebut merepresentasikan tanggal perdagangan BI, bukan timestamp berita yang perlu dikonversikan.

---

## 14. Penentuan Cutoff Berita terhadap JISDOR

### Keputusan
Menggunakan **08:00 WIB** sebagai cutoff pembentukan target.

### Aturan
- Berita sebelum **08:00 WIB** → dapat dikaitkan dengan fixing pada hari tersebut.
- Berita pada atau setelah **08:00 WIB** → digulirkan ke fixing hari kerja BI berikutnya.

### Alasan
JISDOR dihitung berdasarkan transaksi antarbank pada sekitar **08:00–09:45 WIB** dan dipublikasikan sekitar pukul 10:00 WIB.

Cutoff 08:00 dipilih agar berita yang muncul setelah awal jendela pembentukan fixing tidak dianggap sebagai informasi yang memengaruhi harga yang sudah mulai terbentuk.

### Alternatif yang Tidak Dipilih
Menggunakan pukul 10:00 WIB sebagai cutoff.

### Alasan Tidak Dipilih
Berita pada pukul 09:30 WIB, misalnya, memang terbit sebelum JISDOR dipublikasikan, tetapi informasi tersebut sudah muncul ketika transaksi yang membentuk fixing sedang berlangsung.

---

## 15. Penanganan Berita di Luar Hari Perdagangan

### Keputusan
Menggunakan **roll-forward** ke hari kerja BI berikutnya.

### Contoh
Berita yang diterbitkan pada Sabtu atau Minggu akan dikaitkan dengan fixing hari kerja berikutnya.

### Alasan
Informasi tidak seharusnya dikaitkan dengan fixing yang sudah terjadi pada masa lalu.

### Alternatif yang Dipertimbangkan

#### Roll-backward
Tidak digunakan karena dapat menyebabkan informasi masa depan dikaitkan dengan harga yang sudah terbentuk.

#### Membuang berita akhir pekan
Tidak digunakan karena dapat menghilangkan informasi geopolitik penting yang terjadi ketika pasar sedang tutup.

---

## 16. Penentuan Kalender Hari Kerja

### Keputusan
Kalender hari kerja diturunkan langsung dari data JISDOR.

### Alasan
JISDOR hanya memiliki data pada hari ketika fixing tersedia. Dengan demikian, hari tanpa entri JISDOR dapat digunakan untuk mengidentifikasi hari non-trading tanpa bergantung pada library kalender eksternal.

---

## 17. Mempertahankan Hari Tanpa Berita

### Keputusan
Hari kerja yang tidak memiliki berita geopolitik **tetap dipertahankan** dalam `aligned_daily.csv`.

### Representasi
- `n_news = 0`
- `text_norm_concat = ""`

### Alasan
Tidak adanya berita merupakan kondisi yang memiliki makna dalam deret waktu.

Menghapus hari tanpa berita dapat menyebabkan:
- kontinuitas deret waktu terganggu;
- observasi kurs hilang;
- model hanya melihat hari yang memiliki berita.

---

## 18. Normalisasi Volume Berita

### Keputusan
Tidak menggunakan `n_news` mentah sebagai satu-satunya ukuran volume berita.

Variabel tambahan:

`news_density = n_news / window_hours`

digunakan untuk mempertimbangkan perbedaan panjang jendela waktu.

### Alasan
Jendela berita untuk hari Senin lebih panjang karena mencakup periode akhir pekan.

Rata-rata:

- Senin: sekitar **62,85 jam**
- Hari kerja lainnya: sekitar **30 jam**

Namun jumlah berita tidak meningkat dua kali lipat. Hal ini menunjukkan bahwa jumlah berita mentah tidak dapat dibandingkan langsung antar hari tanpa memperhitungkan panjang jendela.

### Implikasi
`n_news / window_hours` dapat digunakan sebagai variabel kontrol dalam pemodelan atau analisis deskriptif.

---

## 19. Jalur Preprocessing TF-IDF dan BERT

### Keputusan
Membuat dua jalur preprocessing yang berbeda:

1. **TF-IDF**
2. **FinBERT/BERT**

### Alasan
Kedua jenis representasi teks memiliki kebutuhan preprocessing yang berbeda.

TF-IDF bekerja pada token yang telah dinormalisasi, sedangkan tokenizer BERT membutuhkan bentuk teks yang relatif alami agar proses subword tokenization tetap bekerja sebagaimana mestinya.

---

## 20. Preprocessing TF-IDF

### Keputusan
Pada jalur TF-IDF:

- melakukan lowercase;
- menghapus sebagian besar tanda baca;
- mempertahankan `.`, `$`, dan `%`;
- menggunakan custom stopword;
- mempertahankan kata negasi;
- tidak melakukan stemming;
- tidak melakukan lemmatization;
- melakukan deduplikasi;
- menghapus data yang bukan artikel valid.

### Alasan

#### Mempertahankan `$` dan `%`
Simbol tersebut dapat menunjukkan informasi ekonomi, misalnya:

- `oil surges 8%`
- `$100 billion in sanctions`

#### Mempertahankan `.`
Diperlukan untuk:
- angka desimal seperti `3.5`;
- singkatan seperti `U.S.`

#### Mempertahankan Negasi
Kata seperti:

- `not`
- `no`
- `never`
- `without`

dipertahankan karena penghapusannya dapat mengubah makna kalimat.

#### Tidak Melakukan Stemming
Stemming dapat menghasilkan bentuk kata yang tidak bermakna pada nama negara, organisasi, atau istilah geopolitik.

---

## 21. Preprocessing BERT

### Keputusan
Melakukan preprocessing seminimal mungkin pada teks yang akan diberikan kepada FinBERT.

### Keputusan
Tidak melakukan:

- lowercase manual;
- penghapusan tanda baca secara agresif;
- stopword removal;
- stemming;
- lemmatization.

### Alasan
Tokenizer BERT dirancang untuk memproses teks dalam bentuk yang relatif alami. Normalisasi agresif dapat mengubah representasi teks yang diharapkan oleh model pretrained.

---

## 22. Penghapusan Boilerplate pada BERT

### Keputusan
Menghapus boilerplate CNBC yang telah diuji sebelumnya, tetapi tidak melakukan penghapusan teks secara agresif.

### Contoh
Boilerplate yang dapat dihapus adalah bagian berulang seperti promosi atau footer tertentu.

### Alasan
Teks boilerplate tidak merepresentasikan informasi geopolitik dan dapat menjadi noise bagi model.

### Keputusan Tambahan
Pola boilerplate harus diuji terhadap keseluruhan korpus sebelum digunakan agar kalimat berita asli tidak ikut terhapus.

---

## 23. Penanganan Artikel Tanpa Body

### Keputusan
Jika body artikel tidak tersedia, menggunakan `description` sebagai fallback.

### Alasan
Sebagian artikel, terutama artikel dengan format live blog, tidak memiliki body yang dapat diekstraksi secara normal.

Sebanyak **454 artikel** menggunakan description sebagai fallback.

---

## 24. Penanganan Panjang Teks BERT

### Keputusan
Artikel yang melebihi kapasitas BERT diproses menggunakan **chunking 512 token dengan overlap 50 token**.

### Alasan
BERT memiliki batas panjang input sekitar 512 token.

Rata-rata artikel setelah preprocessing sekitar **710 token**, dan sekitar **73,4% artikel** melebihi batas tersebut.

### Alasan Menggunakan Overlap
Overlap 50 token digunakan untuk mengurangi risiko informasi penting terpotong tepat pada batas antar-chunk.

---

## 25. Pemrosesan Data Kurs JISDOR

### Keputusan
Data JISDOR dibersihkan dan diubah menjadi format kronologis dari tanggal terlama ke tanggal terbaru.

### Alasan
Data raw JISDOR tersusun dari tanggal terbaru ke tanggal terlama, sedangkan perhitungan perubahan nilai tukar membutuhkan informasi hari sebelumnya.

---

## 26. Parsing Format Tanggal JISDOR

### Keputusan
Format tanggal JISDOR dibaca secara eksplisit sebagai `M/D/YYYY` dan dikonversi menjadi `YYYY-MM-DD`.

### Alasan
Parsing eksplisit menghindari ambiguitas antara format bulan/hari dan hari/bulan.

Format standar `YYYY-MM-DD` juga memudahkan pengurutan berdasarkan waktu.

---

## 27. Parsing Nilai Kurs

### Keputusan
Parser kurs dibuat untuk menangani format angka Indonesia maupun Inggris.

### Contoh
- `15.250,00`
- `15,250.00`

### Alasan
Perbedaan pemisah ribuan dan desimal dapat menyebabkan nilai kurs terbaca secara salah apabila format tidak ditangani secara eksplisit.

---

## 28. Penghapusan Kolom NO JISDOR

### Keputusan
Kolom `NO` dihapus setelah data dibaca.

### Alasan
Kolom tersebut hanya merupakan nomor urut dan tidak memiliki informasi ekonomi yang digunakan dalam pemodelan.

---

## 29. Pembentukan Target Kurs

### Keputusan
Perubahan kurs dibandingkan dengan **hari kerja BI sebelumnya**, bukan tanggal kalender sebelumnya.

### Alasan
JISDOR tidak tersedia pada:
- Sabtu;
- Minggu;
- hari libur;
- cuti bersama.

Dengan menggunakan hari kerja BI sebelumnya, perubahan kurs Senin dibandingkan dengan Jumat sebelumnya, bukan Minggu.

---

## 30. Menyimpan Variabel Kontrol Alignment

### Keputusan
Menyimpan variabel kontrol berikut pada `aligned_daily.csv`:

- `is_offhours`
- `hours_to_fixing`
- `window_hours`
- `n_news`
- `target_date_lag1`

### Alasan
Variabel tersebut memungkinkan proses temporal alignment diperiksa dan digunakan dalam analisis robustness.

Keputusan ini mencegah efek dari perbedaan panjang jendela waktu tersembunyi di dalam dataset.

---

## 31. Mempertahankan Varian Lag-1

### Keputusan
`target_date_lag1` disimpan sebagai varian target untuk robustness check.

### Alasan
Variabel ini memungkinkan pengujian alternatif terhadap aturan alignment utama tanpa perlu mengulang seluruh proses akuisisi dan preprocessing.

---

## 32. Struktur Dataset Akhir

### Keputusan
Hasil temporal alignment disimpan dalam:

`aligned_daily.csv`

### Alasan
File ini menjadi titik pertemuan antara:

- representasi berita geopolitik;
- informasi tanggal;
- jumlah berita;
- informasi jendela waktu;
- nilai tukar;
- target perubahan kurs.

Dengan demikian, tahap pemodelan dapat dilakukan menggunakan dataset yang sudah terstruktur secara temporal.

---

## 33. Target Prediksi: Kelas Volatilitas

### Keputusan
Target model adalah `volatility_class` (`low` / `medium` / `high`), yaitu kelas dari `realized_vol`:

`realized_vol(t) = std(log_return[t-4], ..., log_return[t])`

atau simpangan baku perubahan kurs (log return) selama 5 hari kerja BI terakhir, termasuk hari t (`VOLATILITY_WINDOW = 5`).

### Alasan
- Pertanyaan penelitian berfokus pada **fluktuasi** nilai tukar, bukan arah naik/turun. Volatilitas lebih relevan untuk menangkap dampak ketidakpastian geopolitik.
- Jendela 5 hari kerja (±1 minggu) meredam derau harian satu hari.

### Konsekuensi yang Perlu Dipahami
Jendela hari t dan hari t-1 berbagi 4 dari 5 return yang sama, sehingga kelas volatilitas sangat persisten dari hari ke hari. Akibatnya baseline naive persistence (Keputusan 36) sudah kuat, dan fitur harga perlu dirancang mengikuti struktur ini (Keputusan 37).

### Alternatif yang Dipertimbangkan
- Arah perubahan kurs (`direction`: up/flat/down dengan `FLAT_THRESHOLD`).
- Volatilitas ke depan yang tidak tumpang tindih dengan data kemarin (mis. |r[t]| atau std r[t..t+4]).

### Alasan Tidak Dipilih
Kolom `direction` tetap dihitung di `preprocess_kurs.py`, tetapi tidak dipakai sebagai target karena pertanyaan penelitian adalah fluktuasi. Target volatilitas ke depan tidak dipilih untuk menjaga definisi target yang sudah ditetapkan; keterbatasan persistensinya dicatat sebagai bagian dari interpretasi hasil.

---

## 34. Ambang Kelas Volatilitas (Tertile dari Data Train)

### Keputusan
`realized_vol` dibagi menjadi 3 kelas menggunakan **tertile** (`pd.qcut(..., 3)`) yang dihitung **hanya dari 70% hari kerja tertua**. Ambang yang diperoleh:

- `low`: realized_vol < 0,207%
- `medium`: 0,207% – 0,312%
- `high`: > 0,312%

Ambang yang sama kemudian diterapkan (`pd.cut`) ke seluruh deret, termasuk valid dan test.

### Alasan
- **Kelas seimbang:** tertile membagi data train menjadi tiga kelompok berukuran sama, sehingga model tidak bisa mendapat akurasi tinggi hanya dengan menebak satu kelas.
- **Relatif terhadap perilaku rupiah:** "high" berarti lebih bergejolak dari kebiasaan USD/IDR selama periode train, bukan ambang absolut yang dipilih manual.
- **Mencegah kebocoran:** jika ambang dihitung dari seluruh data, ambang tersebut ikut "mengetahui" sebaran volatilitas periode test.

### Konsekuensi
Distribusi kelas di valid dan test tidak lagi seimbang (test: low 51, medium 65, high 51). Hal ini wajar karena kondisi pasar berubah antar periode.

---

## 35. Split Kronologis 70/15/15 pada Hari Berberita

### Keputusan
- Pelabelan kelas dan fitur lag dihitung pada **kalender hari kerja lengkap**, baru kemudian difilter ke hari yang memiliki berita (`n_news > 0`).
- Data dibagi secara kronologis: 70% train, 15% valid, 15% test.
- Ketiga skrip training (`train_baseline_ts.py`, `train_tfidf.py`, `train_combined.py`) memakai **himpunan hari yang identik**:
  - Train: 772 hari (16-09-2021 s.d. 25-03-2025)
  - Valid: 165 hari (26-03-2025 s.d. 09-12-2025)
  - Test: 167 hari (11-12-2025 s.d. 01-09-2026)

### Alasan
- Split acak akan membuat model belajar dari masa depan untuk memprediksi masa lalu.
- Valid dipakai untuk memilih hyperparameter; test hanya dipakai **satu kali** di akhir.
- Fitur lag dihitung sebelum filter agar "hari sebelumnya" selalu berarti hari kerja BI sebelumnya, bukan hari berberita sebelumnya.
- Himpunan hari yang identik diperlukan agar akurasi antar model dapat dibandingkan secara langsung.

---

## 36. Baseline Naive Persistence

### Keputusan
Baseline utama adalah **naive persistence**: kelas volatilitas hari t diprediksi sama dengan kelas hari t-1 (`lag1_volatility_class`). Baseline ini tidak dilatih dan tidak memiliki hyperparameter.

### Alasan
Karena target sangat persisten (Keputusan 33), naive persistence adalah tolok ukur minimum yang bermakna. Model yang tidak dapat melampaui naive tidak memberikan nilai tambah.

### Alternatif yang Tidak Dipilih
Dummy classifier (selalu menebak kelas mayoritas train). Baseline ini dihapus karena terlalu lemah (akurasi test 0,305, setara menebak acak) dan tidak relevan untuk pertanyaan penelitian.

---

## 37. Fitur Harga (`PRICE_FEATURES`)

### Keputusan
Fitur harga didefinisikan satu kali di `dataset_split.py` dan dipakai **sama persis** oleh XGBoost tanpa NLP dan XGBoost + TF-IDF + LM:

| Fitur | Isi |
|---|---|
| `lag1_log_return` s.d. `lag4_log_return` | Return 1–4 hari kerja sebelumnya |
| `known4_mean`, `known4_std` | Rata-rata dan simpangan baku dari r[t-4..t-1] |
| `vol_if_r0` | `realized_vol(t)` jika r[t] = 0 |
| `vol_if_rtyp_pos`, `vol_if_rtyp_neg` | `realized_vol(t)` jika r[t] = ± median \|r\| historis (expanding, di-shift 1 hari) |
| `lag1_volatility_class_enc` | Kelas kemarin (low = 0, medium = 1, high = 2) |

### Alasan
Empat dari lima return dalam jendela target sudah diketahui sebelum hari t (Keputusan 33). Fitur di atas memberikan informasi tersebut secara langsung, sehingga model cukup menilai apakah pergerakan hari t (dan informasi berita) cukup besar untuk memindahkan kelas.

Semua fitur hanya menggunakan data **sebelum hari t**. Hal ini telah diuji: mengubah return hari t dan sesudahnya tidak mengubah nilai fitur hari t.

### Alternatif yang Dipertimbangkan
Fitur awal: `lag1–3_log_return`, `lag1_realized_vol`, `lag1_volatility_class_enc`.

### Alasan Tidak Dipilih
- `lag4_log_return` tidak tersedia, padahal r[t-4] masih termasuk dalam jendela target.
- `lag1_realized_vol` memuat r[t-5], yaitu return yang sudah keluar dari jendela target.
- Dengan fitur awal, XGBoost tanpa NLP hanya mencapai akurasi test 0,689, lebih rendah dari naive (0,713).

---

## 38. Tipe Numerik untuk Fitur Kelas Kemarin

### Keputusan
`lag1_volatility_class_enc` disimpan sebagai **float** (`.astype(float)`).

### Alasan
Tanpa konversi, kolom tersebut mewarisi tipe `category` dari `pd.cut`. XGBoost membacanya sebagai kategori tanpa urutan jika diberi DataFrame (XGBoost tanpa NLP dan ablasi), tetapi sebagai angka berurutan jika diberi array numpy (model gabungan). Akibatnya fitur yang sama diperlakukan berbeda antar model yang dibandingkan. Pada versi XGBoost lama, kondisi ini juga dapat menimbulkan error.

---

## 39. Normalisasi Fitur Sentimen LM (Rata-rata per Artikel)

### Keputusan
Fitur sentimen Loughran-McDonald yang dipakai model:

- `mean_lm_polarity`
- `lm_positive_mean = lm_positive_sum / n_news`
- `lm_negative_mean = lm_negative_sum / n_news`

Fitur ini didefinisikan sekali sebagai `LM_FEATURES` di `dataset_split.py`.

### Alasan
- **Konsisten dengan Keputusan 18.** Jumlah berita mentah tidak dapat dibandingkan langsung antar hari. Fitur berbasis jumlah (`lm_*_sum`) memiliki masalah yang sama karena nilainya ikut membesar seiring jumlah artikel.
- **Fitur sentimen seharusnya mengukur nada, bukan volume.** Volume berita sudah terwakili terpisah melalui `n_news`.
- **Temuan EDA:** jumlah artikel per hari tidak stabil antar periode (rata-rata 3,9 pada train dan 8,0 pada valid), diduga akibat perbedaan cakupan scraping. Sementara itu, jumlah kata negatif per artikel relatif stabil (±21–27). Pergeseran ini sudah terlihat pada train dan valid, tanpa perlu melihat data test.

### Alternatif yang Dipertimbangkan
`lm_positive_sum` dan `lm_negative_sum` (jumlah per hari).

### Catatan Transparansi
Kedua versi telah dibandingkan pada model utama. Versi rata-rata: akurasi valid 0,812 dan test 0,760. Versi jumlah: akurasi valid 0,806 dan test 0,766. Selisih di test hanya satu hari dan justru sedikit berpihak pada versi jumlah, sehingga keputusan ini tidak mengubah kesimpulan dan tidak didasarkan pada hasil test; versi rata-rata dipilih karena alasan di atas (dan akurasi valid-nya juga lebih tinggi).

---

## 40. Representasi TF-IDF pada Model Gabungan

### Keputusan
- TF-IDF: `min_df=5`, `max_df=0.8`, unigram + bigram, maksimum 5.000 fitur, di-fit dari data train saja.
- Untuk model gabungan, TF-IDF dikompres menjadi 20 komponen dengan `TruncatedSVD` yang juga di-fit dari data train saja.

### Alasan
Tanpa kompresi, 5.000 kolom TF-IDF akan mendominasi pemilihan split pada pohon XGBoost, sehingga fitur harga dan LM yang jumlahnya jauh lebih sedikit hampir tidak pernah terpakai. Model TF-IDF saja (`train_tfidf.py`) tetap memakai 5.000 kolom penuh karena tidak digabung dengan fitur lain.

---

## 41. XGBoost dengan Tuning Hyperparameter yang Seragam

### Keputusan
Semua model XGBoost di-tuning menggunakan fungsi dan grid yang sama (`tune_xgb` dan `XGB_GRID` di `dataset_split.py`):

- `n_estimators`: 100, 200, 400
- `max_depth`: 2, 3, 4
- `learning_rate`: 0,03; 0,05; 0,1

Total 27 kombinasi. Setiap kombinasi dilatih pada train dan dinilai dengan akurasi valid. Kombinasi terbaik dievaluasi pada test **satu kali**. Parameter lain memakai default XGBoost dengan `random_state=42`.

### Alasan
- Sebelumnya hanya model gabungan yang di-tuning, sedangkan model pembanding memakai parameter tetap. Perbandingan seperti itu tidak adil karena model NLP mendapat kesempatan optimasi lebih banyak.
- Dengan grid yang sama, selisih akurasi antar model mencerminkan kontribusi fitur, bukan perbedaan upaya tuning.

### Alternatif yang Tidak Dipilih
Logistic Regression sebagai algoritma pembanding pada model teks saja. Algoritma ini dihapus karena perbandingan antar algoritma bukan bagian dari pertanyaan penelitian.

---

## 42. Model yang Dibandingkan

### Keputusan
**Perbandingan utama** (`train_combined.py`):

1. Naive persistence (baseline)
2. XGBoost tanpa NLP (`PRICE_FEATURES`)
3. XGBoost + TF-IDF + LM (model utama)

**Pendukung:**

- XGBoost teks saja: TF-IDF saja dan LM saja, tanpa fitur harga (`train_tfidf.py`).
- Ablasi: XGBoost harga + LM tanpa TF-IDF (`train_combined.py`).

### Alasan
- Selisih (3) terhadap (2) mengukur kontribusi berita geopolitik di atas histori harga, yang merupakan inti pertanyaan penelitian.
- Model teks saja menguji apakah berita dapat memprediksi volatilitas **tanpa** histori harga.
- Ablasi memisahkan kontribusi LM dari TF-IDF.

---

## 43. Metrik Evaluasi dan Uji Signifikansi

### Keputusan
- Metrik utama: **akurasi** pada test set. Metrik pendukung: macro F1, classification report, dan confusion matrix.
- Signifikansi selisih antar model diuji dengan **uji McNemar exact** (α = 0,05) pada test set. Uji ini sudah dijalankan otomatis di `train_combined.py`.

### Alasan
- Kelas pada data train seimbang, sehingga akurasi mudah diinterpretasikan. Macro F1 ditambahkan karena distribusi kelas test tidak seimbang.
- Uji McNemar sesuai untuk membandingkan dua classifier pada data uji yang sama: yang dihitung hanya hari-hari ketika kedua model memberikan hasil berbeda.

---

## 44. Hasil Evaluasi Akhir dan Keterbatasan

### Hasil (test set, 167 hari)

| Model | Akurasi valid | Akurasi test | Macro F1 test |
|---|---|---|---|
| Naive persistence | 0,727 | 0,713 | 0,72 |
| XGBoost teks saja (TF-IDF) | 0,382 | 0,287 | 0,21 |
| XGBoost teks saja (LM) | 0,370 | 0,305 | 0,31 |
| XGBoost tanpa NLP | 0,806 | 0,731 | 0,74 |
| Ablasi: harga + LM | 0,818 | 0,737 | – |
| **XGBoost + TF-IDF + LM** | **0,812** | **0,760** | **0,77** |

Uji McNemar (hari "A saja benar" / "B saja benar"):

- XGBoost tanpa NLP vs naive: 20 / 17, p = 0,743
- XGBoost + TF-IDF + LM vs XGBoost tanpa NLP: 8 / 3, p = 0,227
- XGBoost + TF-IDF + LM vs naive: 23 / 15, p = 0,256

### Interpretasi
- Histori harga merupakan sumber informasi utama. Total feature importance model utama: harga 0,69, TF-IDF 0,26, LM 0,05.
- Berita geopolitik menambah akurasi sebesar +3,0 poin di atas model harga saja, terutama pada kelas `medium`. Namun peningkatan ini **tidak signifikan secara statistik** pada α = 0,05.
- Berita **tidak mampu memprediksi sendirian**. Model teks saja berada pada tingkat tebakan acak, dan model TF-IDF saja cenderung menghafal kosakata periode tertentu (misalnya "invasion", "annexation crimea") alih-alih pola yang dapat digeneralisasi.
- TF-IDF (topik berita) lebih informatif dibanding nada sentimen LM.

### Keterbatasan
- Test set kecil (167 hari): satu hari setara 0,6 poin akurasi.
- Evaluasi hanya pada satu periode test.
- Jumlah artikel per hari meningkat pada 2025–2026, kemungkinan akibat cakupan scraping.
- Fitur harga (Keputusan 37) dirancang pada tahap eksplorasi ketika hasil test versi awal sudah terlihat. Rancangannya diturunkan dari definisi target, bukan dari coba-coba, tetapi hal ini tetap dicatat sebagai keterbatasan.

---

## 45. Pemeriksaan Pergeseran Distribusi Fitur Antar Periode

### Keputusan

Memasukkan pemeriksaan distribusi fitur antar periode sebagai bagian dari **Exploratory Data Analysis (EDA)**. Perbandingan train dan valid digunakan untuk mengidentifikasi pergeseran sebelum evaluasi akhir pada test.

### Bukti dari EDA

Berdasarkan ringkasan EDA, rata-rata jumlah artikel per hari berubah dari sekitar **3,9 pada train** menjadi **8,0 pada valid**. Perbedaan ini menunjukkan perubahan rata-rata volume berita antar periode, sehingga indikasi pergeseran sudah dapat diketahui tanpa melihat test.

Perubahan cakupan scraping merupakan salah satu kemungkinan penyebabnya. Namun, penyebab tersebut belum dapat dipastikan hanya dari rata-rata jumlah artikel; perubahan intensitas pemberitaan dan komposisi periode juga perlu dipertimbangkan.

### Cakupan Pemeriksaan

- Jumlah artikel per tahun dan per hari untuk menggambarkan cakupan hasil scraping dari waktu ke waktu. Perbandingan antar tahun perlu memperhatikan apakah tahun tersebut tercakup penuh atau hanya sebagian.
- Distribusi `n_news` antar periode, termasuk median, kuartil, dan proporsi hari tanpa berita, agar analisis tidak hanya bergantung pada rata-rata.
- Sebaran fitur teks, misalnya `lm_negative_sum`, pada train dan valid untuk memeriksa apakah perubahan volume berita turut memengaruhi fitur agregat.

Pemeriksaan ini termasuk **distribution shift pada fitur**. Istilah *covariate shift* secara lebih khusus mengasumsikan bahwa distribusi fitur berubah sementara hubungan bersyarat antara fitur dan target tetap; pemeriksaan sebaran fitur saja belum membuktikan asumsi tersebut.

### Implikasi

Fitur berbentuk jumlah, seperti `lm_negative_sum`, dapat meningkat karena lebih banyak artikel atau teks yang terkumpul. Karena itu, kenaikannya tidak langsung diartikan sebagai peningkatan intensitas sentimen negatif. Analisis perlu mempertimbangkan volume berita dan panjang teks, serta membandingkan fitur agregat dengan ukuran yang dinormalisasi bila sesuai.

Normalisasi `news_density` pada keputusan 18 memperhitungkan panjang jendela waktu, tetapi tidak otomatis mengatasi perubahan cakupan scraping antar periode.

### Batas Penggunaan Data

Test disimpan untuk evaluasi akhir dan tidak digunakan untuk menentukan fitur, normalisasi, atau parameter model. Transformasi yang mempelajari parameter dari data, seperti scaler dan TF-IDF, di-fit hanya pada train, kemudian diterapkan pada valid dan test.

---

## 46. Finalisasi Fitur Topik LDA

### Keputusan
`data/processed/topic_modelling/features/selected_model_daily_topic_features.csv` dibuat ulang dari model **LDA** dengan memanggil `extract_daily_feature_files(loaded, "lda")` di `src/topic_modelling/evaluate.py`, tanpa menjalankan `main()`.

- Hasil: 1.202 baris × 37 kolom (kolom `model` = `lda`).
- Nilainya sama dengan `lda_daily_topic_features.csv` (selisih numerik maksimum 4,4 × 10⁻¹⁶; hanya berbeda format penulisan float).
- `selected_model.json` tidak berubah (`selected_model` = `lda`, `selection_type` = `manual_final_selection`).
- Lima file lain di folder `features/` tidak diubah.

### Alasan
- Versi sebelumnya berisi fitur **Top2Vec** (1.202 × 141, kolom `model` = `top2vec`), sehingga tidak konsisten dengan `selected_model.json` yang menetapkan LDA sebagai model final.
- `main()` tidak dijalankan karena `gensim` tidak terpasang di lingkungan saat ini, dan `main()` akan menimpa `selection_type` di `selected_model.json`.

### Alternatif yang Tidak Dipilih
- Menjalankan ulang `main()`: ditolak karena alasan di atas.
- Mengubah `selected_model.json` menjadi Top2Vec: ditolak karena LDA sudah ditetapkan sebagai model topik final.

### Konsekuensi
- Tidak ada skrip di `src/` yang membaca file ini. Evaluasi early, late, dan feature fusion membaca `data/processed/topic_modelling/<model>/daily_topic_features.csv` untuk keempat model topik (LDA, BERTopic, DTM, Top2Vec) sebagai **varian fitur**. Regenerasi ini tidak mengubah hasil fusion mana pun.
- Karena itu, Model C (Keputusan 47) dapat memakai varian topik selain LDA.

---

## 47. Aturan Seleksi Model C

### Keputusan
**Model C = `early_fusion_1` dengan embedding jalur BERT baru** (`run_id` `early_fusion_1__new`):

| Blok | Sumber | Kolom |
|---|---|---|
| Harga | `PRICE_FEATURES` (Keputusan 37) | 10 |
| Embedding | `finbert_fulltext`, `data/processed/bert_path/finbert_fulltext_features.csv` | 768 |
| Topik | Top2Vec | 204 |
| NER | `daily_entity_features.csv` | 54 |
| **Total** | early fusion (concat), XGBoost | **1.036** |

Macro-F1 walk-forward CV: **0,7744 ± 0,0337** (akurasi CV 0,7885).

**Aturan final (dibekukan):** calon Model C hanya berasal dari pipeline yang valid, yaitu embedding transformer jalur BERT baru atau embedding statis. Embedding transformer versi lama (dari `text_norm_concat`) tidak boleh menjadi Model C dan hanya dipakai sebagai pembanding. Di antara calon yang sah, dipilih rata-rata macro-F1 CV tertinggi, dengan tie-break akurasi CV.

**Protokol CV** (sama untuk semua tahap): 937 hari train + valid kanonis (16-09-2021 s.d. 09-12-2025), `TimeSeriesSplit(5)` dengan expanding window (train fold pertama 157 hari, tiap fold evaluasi 156 hari). Konfigurasi kandidat tetap. Imputer, scaler, classifier, serta OOF dan meta-LR late fusion di-fit hanya di bagian train fold. Test tidak dimuat.

**Finalis:** 13 kandidat dari file validasi yang sudah ada:
- top-5 early fusion (dari 240 kandidat);
- top-5 late fusion (dari 3.888 kandidat);
- top-3 feature fusion temporal E5–E8 (dari 48).

Peringkat memakai macro-F1 validasi. Semua 13 kandidat lolos filter "≤ 1 hari benar di bawah #1 arsitektur" dan dievaluasi dengan CV.

### Riwayat Aturan
1. **Aturan awal** (`src/select_model_c.py`): CV tertinggi, lalu kandidat dengan rata-rata ≥ (terbaik − 1 SD antar-fold) dianggap setara, dan dipilih yang `n_features`-nya paling sedikit. Kandidat terbaik CV adalah `early_fusion_5` (0,7773 ± 0,0320), sehingga ambangnya 0,7453. Sebanyak 12 dari 13 kandidat lolos ambang. Hasilnya `late_fusion_3`: kurs + Word2Vec fulltext, meta-LR, 112 fitur, 0,7624 ± 0,0176.
2. **Perubahan pertama** (`src/select_model_c_bert_path.py`): embedding transformer lama dibuat dari `text_norm_concat` (teks jalur TF-IDF), sehingga menyimpang dari Keputusan 21–24. Embedding diekstrak ulang dengan jalur BERT (`src/bert_text.py`, `src/extract_bert_embeddings.py`):
   - teks: judul + body, dengan description hanya sebagai fallback bila body kosong;
   - chunk 512 token dengan overlap 50 (Keputusan 24);
   - mean pooling token konten, rata-rata antar-chunk, lalu rata-rata antar-artikel per `target_date`;
   - model float32.

   Ekstraksi ulang dilakukan untuk top-2 early dan top-2 late menurut CV. Aturan diganti menjadi CV tertinggi tanpa aturan 1 SD, dengan versi lama dan baru sama-sama menjadi calon. Hasilnya `early_fusion_5__old` (0,7773).
3. **Perubahan kedua (final)** (`src/finalize_model_c.py`): aturan pipeline valid di atas. Perubahan ini dilakukan **setelah melihat hasil CV, tetapi sebelum test dibuka**. Instruksi pada perubahan pertama tidak tegas soal status versi lama; aturan final menegaskan maksud semula ekstraksi ulang, yaitu mengganti embedding yang cacat.

### Temuan Eksploratif: Embedding Lama vs Jalur BERT (CV, 937 hari)

| Kandidat | Embedding | Fitur | Macro-F1 lama | Macro-F1 baru | Δ baru − lama |
|---|---|---|---|---|---|
| early_fusion_5 (E2 kurs + emb + LDA) | RoBERTa | 828 | 0,7773 ± 0,0320 | 0,7591 ± 0,0461 | −0,0182 |
| early_fusion_1 (E4 kurs + emb + Top2Vec + NER) | FinBERT | 1.036 | 0,7683 ± 0,0443 | 0,7744 ± 0,0337 | +0,0062 |
| late_fusion_1 (E1 kurs + emb) | Qwen2.5-0.5B | 906 | 0,7628 ± 0,0187 | 0,7621 ± 0,0236 | −0,0008 |
| late_fusion_3 (E1 kurs + emb, meta-LR) | Word2Vec (statis) | 112 | 0,7624 ± 0,0176 | – | – |

Sumber: `data/processed/ablation/model_c_cv_bert_path.csv`. Tabel ini disimpan apa adanya untuk analisis eksploratif dan tidak dipakai untuk seleksi.

### Alasan
- Tujuan ekstraksi ulang adalah mengganti embedding transformer yang cacat. Versi lama hanya pembanding untuk mengukur dampak perbaikan preprocessing.
- CV walk-forward di 937 hari dipakai sebagai pengganti skor validasi tunggal, karena finalis dipilih dari ribuan kandidat di 165 hari validasi yang sama sehingga skor validasi maksimumnya bias ke atas.

### Alternatif yang Tidak Dipilih
- **`early_fusion_5` dengan RoBERTa lama** (CV tertinggi, 0,7773 ± 0,0320): ditolak karena embedding-nya berasal dari pipeline yang cacat. Selisihnya dengan Model C hanya 0,0029, jauh di bawah SD antar-fold kedua model (0,0320 dan 0,0337), sehingga secara CV keduanya tidak dapat dibedakan.
- **`late_fusion_3`** (hasil aturan awal): tidak dipakai setelah aturan diganti pada perubahan pertama.

### Konsekuensi
- Model C memakai **Top2Vec, bukan LDA** (model topik final Keputusan 46), karena keempat model topik diuji sebagai varian fitur dalam pencarian early/late fusion. Varian topik mengikuti skor validasi/CV, bukan pilihan model topik final.
- Aturan seleksi berubah dua kali setelah hasil CV terlihat. Test belum dibuka pada kedua perubahan, tetapi seleksi ini tidak sepenuhnya bebas dari pengaruh hasil CV.
- Tim sudah pernah melihat hasil test early/late/feature fusion (`finalists_test.csv`) sebelum seleksi ini. Skrip seleksi tidak membacanya, tetapi pembentukan pool kandidat tidak sepenuhnya buta test.
- Validasi 165 hari ikut menjadi bagian fold CV terakhir, sehingga skor fold 5 tidak independen dari pemilihan finalis.
- Hanya top-2 kandidat per arsitektur yang diekstrak ulang. Kandidat lain dan embedding headline tetap memakai embedding lama.
- Pertanyaan terbuka: Keputusan 23 menyebut 454 artikel tanpa body, sedangkan `src/bert_text.py` menemukan 343 (341 memakai description sebagai fallback, 2 hanya judul) dari 6.013 artikel. Selisih ini dicatat di `data/processed/bert_path/metadata.json` dan belum diselidiki.

---

## 48. Protokol dan Hasil Ablasi

### Keputusan
Ablasi akhir dijalankan dengan `src/ablation.py`. Semua baris dievaluasi di test **sekali, dalam satu kali jalan**, setelah seluruh tuning dan seleksi selesai.

**Protokol seragam:**
- Split kanonis 772 / 165 / 167 hari (Keputusan 35). Hari dan label early fusion di-assert identik dengan `build_dataset()`.
- Semua model XGBoost memakai `tune_xgb()` + `XGB_GRID` (27 kombinasi, Keputusan 41): fit di train, pilih dengan akurasi valid, `random_state=42`.
- Loader early fusion: `build_source_matrices()` (merge kiri pada tanggal; hari tanpa fitur → 0).

**Baris:**

| Baris | Isi |
|---|---|
| naive | kelas kemarin (Keputusan 36) |
| A | XGBoost harga (`PRICE_FEATURES`) |
| B | XGBoost harga + LM + TF-IDF (SVD 20); TF-IDF/SVD di-fit di train, logika disalin dari `train_combined.py` |
| C1 | harga + FinBERT jalur BERT |
| C2 | C1 + Top2Vec |
| C3 | C1 + NER |
| C | C1 + Top2Vec + NER (Model C, Keputusan 47) |
| C_optimized | fitur C dengan grid diperluas (lihat bawah) |
| EF5_old | pembanding sekunder, **bukan Model C**: harga + RoBERTa lama + LDA, protokol tuning sama dengan C |

**C_optimized:**
- Grid: `XGB_GRID` × `subsample` {0,7; 1,0} × `colsample_bytree` {0,3; 0,6; 1,0} × `min_child_weight` {1; 5} × `reg_lambda` {1; 10} = 648 kombinasi. Grid ini mencakup grid seragam pada nilai default.
- Seleksi dengan walk-forward CV 5 fold di 937 hari train + valid, memakai fold yang sama dengan seleksi Model C (rata-rata macro-F1, tie-break akurasi).
- Fit akhir di 772 hari train, sama dengan baris lain.
- Hasil: CV macro-F1 terbaik 0,7843 ± 0,0308 (grid seragam terbaik di CV: 0,7776), dengan parameter `n_estimators` 100, `max_depth` 2, `learning_rate` 0,1, `colsample_bytree` 0,6, `subsample` 1,0, `min_child_weight` 1, `reg_lambda` 1.

**Statistik:** McNemar exact dan moving-block bootstrap CI 95% (blok 10 hari, 2.000 resample, seed 42, indeks yang sama untuk semua baris). Delta pairwise = model kedua − model pertama.

### Hasil (test 167 hari: low 51, medium 65, high 51)

| Baris | Fitur | Akurasi valid | Akurasi test (hari benar) | CI 95% akurasi | Macro-F1 test | McNemar vs naive (model saja / naive saja, p) |
|---|---|---|---|---|---|---|
| naive | 1 | 0,727 | 0,713 (119) | 0,629–0,766 | 0,715 | – |
| A | 10 | 0,806 | 0,731 (122) | 0,641–0,790 | 0,737 | 20 / 17, p = 0,743 |
| B | 33 | 0,812 | 0,766 (128) | 0,694–0,820 | 0,772 | 23 / 14, p = 0,188 |
| C1 | 778 | 0,788 | 0,760 (127) | 0,695–0,808 | 0,765 | 22 / 14, p = 0,243 |
| C2 | 982 | 0,800 | 0,754 (126) | 0,683–0,808 | 0,760 | 22 / 15, p = 0,324 |
| C3 | 832 | 0,788 | 0,760 (127) | 0,695–0,808 | 0,765 | 22 / 14, p = 0,243 |
| **C** | 1.036 | 0,800 | **0,772 (129)** | 0,695–0,832 | 0,777 | 24 / 14, p = 0,143 |
| C_optimized | 1.036 | 0,782 | 0,772 (129) | 0,689–0,832 | 0,777 | 24 / 14, p = 0,143 |
| EF5_old | 828 | 0,800 | 0,749 (125) | 0,665–0,802 | 0,753 | 21 / 15, p = 0,405 |

**Pairwise:**

| Perbandingan | Δ akurasi | CI 95% Δ akurasi | Δ macro-F1 | CI 95% Δ macro-F1 | Hanya kedua benar / hanya pertama benar | p McNemar |
|---|---|---|---|---|---|---|
| A vs B | +0,036 | −0,006 s.d. 0,084 | +0,035 | −0,011 s.d. 0,078 | 9 / 3 | 0,146 |
| B vs C | +0,006 | −0,042 s.d. 0,054 | +0,004 | −0,042 s.d. 0,057 | 8 / 7 | 1,000 |
| A vs C | +0,042 | −0,018 s.d. 0,108 | +0,040 | −0,020 s.d. 0,106 | 14 / 7 | 0,189 |
| EF5_old vs C | +0,024 | −0,006 s.d. 0,066 | +0,024 | −0,006 s.d. 0,071 | 7 / 3 | 0,344 |

Sumber: `data/processed/ablation/ablation_matrix.csv`, `ablation_pairwise.csv`, `ablation_test_predictions.csv`, `ablation_c_optimized_cv.csv`, `ablation_metadata.json`, `c_vs_c_optimized_summary.json`.

### Interpretasi
- Model C memiliki akurasi test tertinggi (0,772), tetapi **tidak ada selisih yang signifikan** pada α = 0,05. Semua CI delta memuat nol. C hanya unggul satu hari atas B (8 / 7 hari berbeda).
- Arah EF5_old → C (+4 hari) sejalan dengan dugaan bahwa perbaikan preprocessing teks membantu, tetapi CI-nya memuat nol. Perbandingan ini juga mengubah embedding (RoBERTa → FinBERT) dan topik (LDA → Top2Vec) sekaligus, jadi bukan efek preprocessing saja.
- **C3 memberi prediksi yang sama persis dengan C1 di semua 167 hari.** Pada hyperparameter terpilih (100 pohon, depth 2, lr 0,03), tidak ada split pohon yang memakai fitur NER (importance blok NER = 0).
- **C_optimized dan C adalah dua model berbeda** dengan akurasi test kebetulan sama:
  - Probabilitas prediksi tidak identik: selisih absolut maksimum 0,274 (2026-08-18), rata-rata 0,034, dan 25 dari 167 hari memiliki selisih > 0,1.
  - Label berbeda di 9 hari: C saja benar 4 hari, C_optimized saja benar 4 hari, keduanya salah 1 hari.
  - Tuning yang lebih luas menaikkan CV (0,7843 vs 0,7776), tetapi tidak menaikkan akurasi test.
- Perbandingan yang adil dengan A dan B memakai baris **C (grid seragam)**, bukan C_optimized.
- **Importance per blok tidak dipakai sebagai ukuran kontribusi.** Importance per blok adalah jumlah `feature_importances_` XGBoost atas kolom-kolom blok, dan nilainya bias terhadap jumlah kolom. Sebagai contoh, blok embedding (768 kolom) mendapat 0,793 di C, sedangkan harga (10 kolom) 0,126. Kontribusi sumber dinilai dari selisih antar baris ablasi.

### Verifikasi
- **A = 0,731**: cocok dengan Keputusan 44.
- **B = 0,766**, bukan 0,760 seperti di Keputusan 44 (selisih satu hari; akurasi valid tetap 0,812). `train_combined.py` asli tanpa edit juga menghasilkan 0,766 di lingkungan saat ini (pandas 3.1.1, numpy 2.5.5, scikit-learn 1.9.1, xgboost 3.4.1), dengan McNemar B vs A 9 / 3, p = 0,146. Jadi angka 0,760 di Keputusan 44 berasal dari versi library tim sebelumnya yang tidak tercatat. Keputusan 44 tidak diubah; angka di keputusan ini memakai lingkungan saat ini.
- **Determinisme:** `ablation.py` dijalankan dua kali (kedua ke `data/processed/ablation/rerun_check/`). `ablation_pairwise.csv`, `ablation_test_predictions.csv`, dan `ablation_c_optimized_cv.csv` identik byte. `ablation_matrix.csv` dan metadata hanya berbeda di kolom waktu.
- **Kebocoran:**
  - Fitur harga/LM hari ≤ t tidak berubah saat return hari t dan sesudahnya diacak, diuji di 4 titik potong (Keputusan 37). Pada dua titik di periode test, `realized_vol` juga diacak.
  - `lag1–3_log_return` sama dengan `log_return.shift(k)`.
  - TF-IDF/SVD identik walau teks valid+test diganti.
  - 0 dari 6.013 artikel terbit setelah cutoff 08.00 WIB hari targetnya, dan `target_date` jalur BERT sama dengan `news_tfidf_clean.csv`.
  - Tidak ada kolom target di file fitur teks.

### Alternatif yang Tidak Dipilih
- **C_optimized sebagai Model C pembanding utama**: tidak dipilih karena upaya tuningnya lebih besar daripada A dan B (Keputusan 41).
- **Memasang versi library lama untuk mereproduksi 0,760**: tidak dipilih karena versi yang dipakai tim tidak tercatat.

### Keterbatasan
- Test kecil (167 hari; satu hari = 0,6 poin akurasi) dan hanya satu periode.
- Top2Vec dan embedding statis dilatih secara transduktif di seluruh korpus oleh tim. Ini bukan kebocoran label, tetapi representasinya ikut melihat teks periode test.
- Seleksi Model C memakai aturan yang diubah dua kali setelah CV terlihat (Keputusan 47).
