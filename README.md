# Credit Scoring BMT — Data Preparation

Proyek ini menyiapkan data untuk model Machine Learning credit scoring pada aplikasi digital BMT (Baitul Maal wa Tamwil). Model akan memprediksi kelayakan pemohon kredit: **LAYAK** atau **TIDAK LAYAK**.

---

## Dataset

File CSV tidak di-push ke repository karena ukurannya terlalu besar. Download manual melalui Google Drive lalu letakkan sesuai folder path di bawah.

| File | Keterangan | Link |
|---|---|---|
| `application_train.csv` | Dataset mentah sumber | [Download](https://drive.google.com/drive/folders/17P3W0ooRG2YR3oVnYo4wTlz2gTDhV4Jf?usp=sharing) |
| `train_ready.csv` | Data final siap training ML | [Download](https://drive.google.com/drive/folders/1xKCx-Abzta_CsMKrcMLrw8yl0Dqo7PlE?usp=sharing) |

**Folder path:** Letakkan file sesuai struktur di bawah:

```
home-credit-default-risk/
├── application_train.csv
└── InfiniteLearning/
    └── train_ready.csv
```

---

## Struktur Proyek

```
.
├── application_train_processing.ipynb       # Notebook utama — seluruh proses data
├── home-credit-default-risk/
│   ├── application_train.csv                # Dataset mentah (sumber, download dari Drive)
│   └── InfiniteLearning/
│       ├── train.csv                        # Data setelah filter kolom
│       └── train_ready.csv                  # Data final siap training ML
└── README.md
```

---

## Alur Proses

### [1] Data Collection
- Sumber: dataset Home Credit Default Risk (real-world)
- 307.511 baris, 122 kolom awal
- Label: `TARGET` — 0 = tidak gagal bayar, 1 = gagal bayar

### [2] Data Preprocessing
- Filter ke 23 kolom yang relevan sesuai spesifikasi produk
- Mengisi nilai kosong dengan median (numerik) dan `Unknown` (teks)
- Menangani anomali `DAYS_EMPLOYED` (nilai 365243 → diganti 0 + flag baru)
- Menghapus 16 baris duplikat
- Encoding: Label Encoding (binary) + One-Hot Encoding (multi-kelas)

### [3] EDA — Exploratory Data Analysis
- Analisis univariat: distribusi pendapatan, kredit, cicilan, usia
- Analisis outlier dengan metode IQR
- Setiap analisis dilengkapi kesimpulan dalam bahasa awam

### [4] Visualisasi Data
- Distribusi jenis pendapatan, pekerjaan, kepemilikan aset, status pernikahan
- Grafik histogram, bar chart, boxplot dengan tabel statistik

### [5] Finalisasi Data
- Feature engineering: `FLAG_TIDAK_BEKERJA`
- Cek class imbalance (rasio 11.4:1)
- Output: `train_ready.csv`

---

## Data Final

| Keterangan | Nilai |
|---|---|
| File | `InfiniteLearning/train_ready.csv` |
| Jumlah baris | 307.495 |
| Jumlah fitur | 115 + 1 TARGET |
| Missing value | 0 |
| Duplikat | 0 |
| Kolom teks | 0 (semua sudah numerik) |

---

## Cara Pakai

```python
import pandas as pd

df = pd.read_csv('home-credit-default-risk/InfiniteLearning/train_ready.csv')

X = df.drop(columns=['TARGET'])
y = df['TARGET']

# Langsung masuk model
model.fit(X, y)
```

> **Catatan:** Data memiliki ketidakseimbangan kelas (91% vs 8%).
> Gunakan `class_weight='balanced'` atau `scale_pos_weight=11` saat training.
