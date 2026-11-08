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
| 1. | [Nama Anggota 1] | [NIM 1] | Modul Deteksi Tepi, Segmentasi Objek, dan Split and Merge |
| 2. | [Nama Anggota 2] | [NIM 2] | Modul Deep Learning (CamVid), Arsitektur GUI MATLAB, dan Integrasi Sistem |

---

## Struktur Repositori

```text
IF4073_PCD_Tugas3/
├── README.md                          # Dokumentasi utama dan panduan eksekusi
├── PEMBAGIAN_TUGAS_DAN_WORKFLOW.md    # Dokumen panduan kerja dan workflow kelompok
├── data/
│   ├── bsds500/                       # 8 citra data uji BSDS500
│   └── camvid/                        # Data uji & skrip download dataset CamVid
├── src/
│   ├── edge_segmentation/             # [Modul 1] Operator deteksi tepi manual & segmentasi objek
│   ├── split_merge/                   # [Modul 2] Algoritma split and merge berbasis quadtree
│   ├── deep_learning/                 # [Modul 3] Pipeline transfer learning, evaluasi, & inferensi CamVid
│   │   └── weights/                   # Bobot model pretrained/fine-tuned
│   └── gui/                           # [Modul 4] MATLAB App Designer (.mlapp)
└── docs/
    └── NIM1_NIM2_Tugas3_IF4073.pdf    # Laporan tugas besar format PDF
```

---

## Dependensi & Prasyarat Lingkungan

### 1. Lingkungan MATLAB
* **MATLAB Versi**: R2021a atau yang lebih baru disarankan.
* **Toolbox yang Diperlukan**:
  * *Image Processing Toolbox* (untuk fungsi Canny dan penanganan I/O citra).
  * *Deep Learning Toolbox* (opsional, jika model CNN dijalankan langsung via MATLAB).

### 2. Lingkungan Python (Jika Deep Learning menggunakan Python)
* **Python Versi**: 3.9+
* **Library**:
  * `torch`, `torchvision` (PyTorch)
  * `numpy`, `scipy`
  * `opencv-python` atau `Pillow`
  * `albumentations` (opsional untuk augmentasi)
* Instalasi dependensi Python:
  ```bash
  pip install -r src/deep_learning/requirements.txt
  ```

---

## Tata Cara Menjalankan Program

### Menjalankan Melalui MATLAB GUI
1. Buka aplikasi **MATLAB**.
2. Arahkan *Current Folder* ke direktori root repositori ini (`IF4073_PCD_Tugas3/`).
3. Tambahkan folder `src` dan seluruh subfoldernya ke dalam search path MATLAB:
   ```matlab
   addpath(genpath('src'));
   addpath(genpath('data'));
   ```
4. Jalankan aplikasi GUI:
   * Ketik `app = SegmentationApp;` pada Command Window, **ATAU**
   * Buka file `src/gui/SegmentationApp.mlapp` melalui MATLAB App Designer dan klik tombol **Run**.
5. Gunakan tab navigasi pada GUI untuk memilih mode segmentasi:
   * **Tab 1: Deteksi Tepi**: Pilih operator (Sobel/Prewitt/Roberts/Laplace/LoG/Canny), atur threshold, dan lakukan segmentasi objek.
   * **Tab 2: Split and Merge**: Masukkan parameter threshold homogenitas split dan merge, lalu amati tahapan dekomposisi region.
   * **Tab 3: Semantic Segmentation**: Muat citra uji jalan, jalankan inferensi CNN, dan amati pemetaan 11 kelas warna.

---

## Dataset

1. **BSDS500 (8 Citra Uji)**:
   * Digunakan untuk pengujian Modul Deteksi Tepi dan Modul Split and Merge.
   * Tersimpan di dalam folder `data/bsds500/`.

2. **CamVid Dataset (Semantic Segmentation)**:
   * **Citra Asli**: [701_StillsRaw_full.zip](http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/files/701_StillsRaw_full.zip)
   * **Anotasi Label**: [LabeledApproved_full.zip](http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/data/LabeledApproved_full.zip)
   * **11 Kelas Pengelompokan**: Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist.
