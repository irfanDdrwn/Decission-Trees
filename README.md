# TA-05: Klasifikasi Churn Pelanggan Telekomunikasi dengan Decision Tree

Proyek akhir Pembelajaran Mesin (Modul 05, Decision Tree). Dua model dibandingkan: **Model A** (tumbuh bebas,
`max_depth=None`, overfit) dan **Model B** (di-*pruning*), lalu Model B dipasang pada aplikasi web Flask.

- **Dataset:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle), 7.043 pelanggan
- **Target:** `Churn` (Yes/No), 26,5% churn
- **Pembuat:** Irfan Dwi Darmawan (301240034), Kelas 5 A

## Struktur

| Berkas | Fungsi |
|---|---|
| `TA05_DecisionTree_Churn.ipynb` | Notebook: EDA, pra-pemrosesan, Model A vs B, evaluasi, visualisasi, penyimpanan model |
| `app.py`, `templates/index.html` | Aplikasi web Flask (formulir 19 fitur, hasil, alur keputusan, perbandingan model) |
| `model/decision_tree.joblib` | Model B (Joblib) |
| `model/decision_tree_overfit.joblib` | Model A (Joblib) |
| `model/tree.json` | Pohon Model B yang sama, dipakai aplikasi (inferensi Python murni, tanpa scikit-learn) |
| `model/metrics.json`, `model/test_set.csv` | Metrik dan data uji |
| `static/` | Grafik hasil analisis |
| `requirements.txt`, `Procfile`, `render.yaml` | Konfigurasi deployment |

## Menjalankan secara lokal

```bash
pip install -r requirements.txt     # hanya flask (+ gunicorn untuk deployment)
python app.py                       # buka http://localhost:5000
```

Aplikasi menelusuri pohon dari `model/tree.json`; hasilnya identik dengan model scikit-learn (diuji pada 7.043 baris).

Melatih ulang: `pip install -r requirements-train.txt`, lalu jalankan seluruh sel `TA05_DecisionTree_Churn.ipynb`.

## Hasil (data uji, 1.409 pelanggan)

| Metrik | Model A (tanpa batas) | Model B (pruned) |
|---|---|---|
| Kedalaman / daun | 22 / 1.109 | 3 / 8 |
| Akurasi latih | 0,998 | 0,791 |
| Akurasi uji | 0,725 | 0,782 |
| AUC | 0,654 | 0,818 |

## Deployment

**Vercel:** unggah ke GitHub (pastikan `model/`, `templates/`, `static/` ikut), lalu *Add New > Project* dan *Deploy*.
**Render:** *New > Web Service* dari repositori; start command ada di `Procfile`. Cek `/health` setelah *deploy*.
