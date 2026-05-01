# Credit Scoring — Home Credit Default Risk
## Infinite Learning · Tugas Akhir Machine Learning

Proyek ini membangun model Machine Learning credit scoring end-to-end dari dataset Home Credit Default Risk.  
Model memprediksi kelayakan pemohon kredit: **LAYAK (0)** atau **TIDAK LAYAK (1)**.

---

## Hasil Model Terbaik

| Metrik | Nilai | Standar Bank |
|---|---|---|
| AUC-ROC | **0.7641** | ≥ 0.75 ⚠️ borderline |
| Gini Coefficient | **0.5281** | ≥ 0.50 ✅ |
| F1 Score | **0.3159** | — |
| Threshold Optimal | **0.680** | via PR curve |
| Model | **LightGBM** | — |

---

## Struktur Proyek

```
.
├── credit_scoring_model.ipynb          # Notebook utama — 6 model, sklearn Pipeline
├── best_model_abel.pkl                 # Model terbaik tersimpan (LightGBM)
├── hasil_prediksi_abel.csv             # Hasil prediksi 48.744 nasabah test
├── rangkuman_perbandingan_model.md     # Perbandingan lengkap semua model
├── home-credit-default-risk/
│   ├── application_train.csv           # Data training (307.511 baris)
│   └── application_test.csv            # Data testing (48.744 baris)
└── README.md
```

---

## Dataset

File CSV tidak di-push ke repository karena ukurannya terlalu besar.  
Download manual melalui link berikut lalu letakkan sesuai struktur folder di atas.

| File | Ukuran | Link |
|---|---|---|
| `application_train.csv` | ~166 MB | [Download](https://drive.google.com/drive/folders/17P3W0ooRG2YR3oVnYo4wTlz2gTDhV4Jf?usp=sharing) |
| `application_test.csv` | ~26 MB | [Download](https://drive.google.com/drive/folders/17P3W0ooRG2YR3oVnYo4wTlz2gTDhV4Jf?usp=sharing) |

---

## Notebook Utama — credit_scoring_model.ipynb

### Fitur yang Digunakan (26 fitur)

| Kategori | Fitur |
|---|---|
| Demografi | CODE_GENDER, NAME_INCOME_TYPE, NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS, OCCUPATION_TYPE, FLAG_OWN_CAR, FLAG_OWN_REALTY |
| Keuangan | CNT_CHILDREN, CNT_FAM_MEMBERS, AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE |
| Waktu & Kerja | DAYS_EMPLOYED, DAYS_LAST_PHONE_CHANGE, AGE_YEARS, DAYS_EMPLOYED_ANOM |
| Skor Eksternal | EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3 |
| Engineering | EXT_SOURCE_MEAN, EXT_SOURCE_MIN, EXT_SOURCE_PROD, DEBT_TO_INCOME, PAYMENT_RATE |

### Model yang Dibandingkan

| Rank | Model | AUC | Gini | F1 | Threshold |
|---|---|---|---|---|---|
| 1 | **LightGBM** ⭐ | 0.7641 | 0.5281 | 0.3159 | 0.680 |
| 2 | XGBoost | 0.7640 | 0.5281 | 0.3169 | 0.687 |
| 3 | CatBoost | 0.7628 | 0.5257 | 0.3148 | 0.691 |
| 4 | Gradient Boosting | 0.7553 | 0.5106 | 0.3099 | 0.650 |
| 5 | Random Forest | 0.7478 | 0.4956 | 0.2989 | 0.620 |
| 6 | Logistic Regression | 0.7453 | 0.4907 | 0.2951 | 0.651 |

### Pipeline Preprocessing

```
ColumnTransformer
├── Numerik  → SimpleImputer(median) → StandardScaler
└── Kategorikal → SimpleImputer(most_frequent) → OneHotEncoder
```

### Penanganan Class Imbalance (rasio 11.4:1)

| Model | Metode |
|---|---|
| Logistic Regression | `class_weight='balanced'` |
| Random Forest | `class_weight='balanced'` |
| Gradient Boosting | `sample_weight` (compute_sample_weight) |
| XGBoost | `scale_pos_weight=ratio` |
| LightGBM | `class_weight='balanced'` |
| CatBoost | `class_weights=[1, ratio]` |

---

## Hasil Prediksi Data Test

Prediksi terhadap **48.744 nasabah** dari `application_test.csv`:

| Status | Jumlah | Persentase |
|---|---|---|
| ✅ DITERIMA (LAYAK) | 42,414 | 87.0% |
| ❌ DITOLAK (TIDAK LAYAK) | 6,330 | 13.0% |

Hasil lengkap tersimpan di `hasil_prediksi_abel.csv`.

---

## Field Input Aplikasi

User mengisi **19 field** di form, sisanya dihitung otomatis oleh sistem.

| # | Field | Tipe | Contoh |
|---|---|---|---|
| 1 | CODE_GENDER | Pilihan | M / F |
| 2 | NAME_INCOME_TYPE | Pilihan | Working / Pensioner / Commercial associate |
| 3 | NAME_EDUCATION_TYPE | Pilihan | Higher education / Secondary education |
| 4 | NAME_FAMILY_STATUS | Pilihan | Married / Single / Civil marriage |
| 5 | OCCUPATION_TYPE | Pilihan | Laborers / Managers / Sales staff |
| 6 | FLAG_OWN_CAR | Pilihan | Y / N |
| 7 | FLAG_OWN_REALTY | Pilihan | Y / N |
| 8 | CNT_CHILDREN | Angka | 0, 1, 2 |
| 9 | CNT_FAM_MEMBERS | Angka | 1, 2, 3 |
| 10 | AMT_INCOME_TOTAL | Angka | 270000 |
| 11 | AMT_CREDIT | Angka | 1000000 |
| 12 | AMT_ANNUITY | Angka | 25000 |
| 13 | AMT_GOODS_PRICE | Angka | 900000 |
| 14 | DAYS_EMPLOYED | Angka | -1000 (negatif = sudah bekerja) |
| 15 | DAYS_LAST_PHONE_CHANGE | Angka | -300 |
| 16 | DAYS_BIRTH | Angka | -12000 (negatif = hari lahir) |
| 17 | EXT_SOURCE_1 | Desimal | 0.50 (range 0–1) |
| 18 | EXT_SOURCE_2 | Desimal | 0.60 (range 0–1) |
| 19 | EXT_SOURCE_3 | Desimal | 0.40 (range 0–1) |

> Field yang dihitung otomatis (tidak perlu diisi user): `AGE_YEARS`, `DAYS_EMPLOYED_ANOM`, `EXT_SOURCE_MEAN`, `EXT_SOURCE_MIN`, `EXT_SOURCE_PROD`, `DEBT_TO_INCOME`, `PAYMENT_RATE`

---

## Cara Pakai Model

```python
import joblib
import pandas as pd
import numpy as np

# Load model
obj       = joblib.load('best_model_abel.pkl')
model     = obj['model']
threshold = obj['threshold']
feat_cols = obj['feature_cols']

# Hitung fitur otomatis dari input user
def feature_engineering(df):
    df = df.copy()
    df['DAYS_EMPLOYED_ANOM'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
    df['DAYS_EMPLOYED']      = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    df['AGE_YEARS']          = abs(df['DAYS_BIRTH']) / 365
    df['DEBT_TO_INCOME']     = df['AMT_CREDIT']  / (df['AMT_INCOME_TOTAL'] + 1)
    df['PAYMENT_RATE']       = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + 1)
    ext = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    df['EXT_SOURCE_MEAN'] = df[ext].mean(axis=1)
    df['EXT_SOURCE_MIN']  = df[ext].min(axis=1)
    df['EXT_SOURCE_PROD'] = df[ext].prod(axis=1)
    return df

# Prediksi dari input user (19 field)
df_input = pd.DataFrame([{
    'CODE_GENDER': 'M', 'NAME_INCOME_TYPE': 'Working', 'AMT_CREDIT': 1000000,
    # ... isi 19 field lainnya
}])

df_fe  = feature_engineering(df_input)
X      = df_fe[feat_cols]
prob   = model.predict_proba(X)[:, 1][0]
pred   = 'TIDAK LAYAK' if prob >= threshold else 'LAYAK'

print(f"Probabilitas gagal bayar : {prob:.2%}")
print(f"Keputusan                : {pred}")
```

---

## Instalasi

```bash
pip install xgboost lightgbm catboost scikit-learn pandas numpy matplotlib seaborn joblib
```

---

## Catatan

- Data memiliki ketidakseimbangan kelas: **91.9% LAYAK vs 8.1% TIDAK LAYAK**
- Threshold optimal dicari via **Precision-Recall curve** (bukan default 0.5)
- Gini Coefficient = `2 × AUC − 1`, standar minimum perbankan adalah **≥ 0.50**
- Detail perbandingan model tersedia di `rangkuman_perbandingan_model.md`
