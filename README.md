# Aplikasi Segmentasi Citra Digital (Tugas 3 IF4073)

Aplikasi segmentasi citra terpadu berbasis antarmuka grafis (MATLAB App Designer) yang menggabungkan tiga pendekatan segmentasi:
1. **Deteksi Tepi (Edge-based Segmentation)**: Menggunakan operator Laplace, LoG, Sobel, Prewitt, Roberts (manual konvolusi) dan Canny untuk ekstraksi region objek.
2. **Segmentasi Klasik (Split and Merge)**: Metode berbasis region *divide and conquer* (Quadtree) dengan predikat homogenitas dan penggabungan region tetangga.
3. **Segmentasi Semantik (Deep Learning)**: Segmentasi semantik piksel menggunakan arsitektur CNN terlatih (pretrained) dengan transfer learning / fine-tuning pada dataset **CamVid (11 kelas)**.

Proyek ini disusun untuk memenuhi **Tugas 3 IF4073 Pemrosesan Citra Digital – Semester I Tahun 2026/2027**, Institut Teknologi Bandung.

---

## Anggota Kelompok
| No | Nama Lengkap | NIM | Pembagian Tugas |
|:--:|:---|:---:|:---|
| 1. | Joel Hotlan Haris Siahaan | 13523025 | Modul Deteksi Tepi, Segmentasi Objek, dan Split and Merge |
| 2. | Ivant Samuel Silaban | 13523129 | Modul Deep Learning (CamVid), Arsitektur GUI MATLAB, dan Integrasi Sistem |

---

## Struktur Repositori

```text
IF4073_PCD_Tugas3/
├── README.md                          # Dokumentasi utama dan panduan eksekusi
├── requirements.txt                   # Daftar dependensi Python (Deep Learning)
├── data/
│   ├── bsds500/                       # 8 citra data uji BSDS500
│   └── camvid/
│       └── splits/                    # Daftar split deterministik 60/20/20 & metadata CamVid
├── src/
│   ├── edge_segmentation/             # [Modul 1] Operator deteksi tepi manual & segmentasi objek
│   ├── split_merge/                   # [Modul 2] Algoritma split and merge berbasis quadtree
│   ├── deep_learning/                 # [Modul 3] Pipeline transfer learning, evaluasi, & inferensi CamVid
│   │   └── weights/                   # Bobot terlatih (best_model.pth) & artefak evaluasi
│   └── gui/                           # [Modul 4] GUI MATLAB (SegmentationApp.m & gui_helpers.m)
└── docs/
│   └── 13523025_13523129_Tugas3_IF4073.pdf
```

---

## Dependensi & Prasyarat Lingkungan

### 1. Lingkungan MATLAB
* **MATLAB Versi**: R2021a atau yang lebih baru disarankan.
* **Toolbox yang Diperlukan**:
  * *Image Processing Toolbox* (untuk operator Canny, operasi morfologi, dan penanganan I/O citra).

### 2. Lingkungan Python (Modul Deep Learning / Semantic Segmentation)
* **Python Versi**: 3.9+ (disarankan memakai virtual environment `venv/` pada root repositori agar terdeteksi otomatis oleh GUI MATLAB).
* **Library Utama**:
  * `torch`, `torchvision` (PyTorch)
  * `numpy`, `Pillow`, `matplotlib`, `tqdm`, `scikit-learn`, `optuna`
* Instalasi dependensi Python:
  ```bash
  pip install -r requirements.txt
  ```

---

## Tata Cara Menjalankan Program

### Menjalankan Melalui MATLAB GUI
1. Buka aplikasi **MATLAB**.
2. Arahkan *Current Folder* ke direktori root repositori ini (`IF4073_PCD_Tugas3/`).
3. Tambahkan folder `src/gui` (atau seluruh `src`) ke dalam search path MATLAB:
   ```matlab
   addpath(genpath('src'));
   ```
4. Jalankan aplikasi GUI:
   * Ketik `app = SegmentationApp;` pada Command Window, **ATAU**
   * Buka file `src/gui/SegmentationApp.m` di MATLAB Editor lalu klik **Run**.
5. Gunakan tab navigasi pada GUI untuk memilih mode segmentasi:
   * **Tab 1: Deteksi Tepi**: Pilih operator (Sobel/Prewitt/Roberts/Laplace/LoG/Canny), atur threshold, sigma, radius closing, dan min area, lalu jalankan segmentasi objek.
   * **Tab 2: Split and Merge**: Atur parameter `splitThresh`, `mergeThresh`, dan `minBlock`, lalu amati tahap dekomposisi Quadtree dan hasil penggabungan wilayah.
   * **Tab 3: Semantic Segmentation**: Muat citra uji jalan (CamVid), jalankan inferensi CNN secara luring memakai `weights/best_model.pth`, lalu atur opacity overlay dan amati distribusi 11 kelas.

---

## Dataset

1. **BSDS500 (8 Citra Uji)**:
   * Digunakan untuk pengujian Modul Deteksi Tepi dan Modul Split and Merge.
   * Tersimpan di dalam folder `data/bsds500/`.

2. **CamVid Dataset (Semantic Segmentation)**:
   * **Citra Asli (UCL)**: [701_StillsRaw_full.zip](http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/files/701_StillsRaw_full.zip)
   * **Anotasi Label (UCL)**: [LabeledApproved_full.zip](http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/data/LabeledApproved_full.zip)
   * **Mirror Kaggle**: [carlolepelaars/camvid](https://www.kaggle.com/datasets/carlolepelaars/camvid) (seluruh 701 citra digabung lalu dibagi ulang 60/20/20 pada `data/camvid/splits/`).
   * **11 Kelas Pengelompokan**: Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist.
