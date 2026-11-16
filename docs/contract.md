# Kontrak Antarmuka Modul (Anggota A ↔ Anggota B)
**Tugas 3 IF4073 Pemrosesan Citra Digital (Semester I 2026/2027)**  
**Tujuan:** Menjadi kesepakatan *function signature*, tipe data, dan struktur file antara **Anggota A** (*Classical CV & Algorithm Specialist*) dan **Anggota B** (*Deep Learning & System Integrator*) agar pengembangan GUI MATLAB (`src/gui/SegmentationApp.mlapp`) dan algoritma backend dapat berjalan paralel tanpa hambatan integrasi.

---

## 1. Aturan Umum Backend (Wajib untuk Semua Fungsi `.m`)

1. **Tidak Boleh Memanggil `figure`, `imshow`, atau `subplot` di Fungsi Backend:**
   - Fungsi di `src/edge_segmentation/` dan `src/split_merge/` **hanya mengembalikan matriks citra/mask** sebagai output.
   - Seluruh proses penampilan citra ke layar (`imshow(..., 'Parent', app.UIAxes)`) ditangani sepenuhnya oleh GUI MATLAB milik Anggota B.
2. **Kompatibilitas Input RGB & Grayscale:**
   - GUI dapat mengirimkan citra RGB (`H x W x 3`, `uint8`) maupun Grayscale (`H x W`, `uint8`).
   - Fungsi backend wajib menangani keduanya dengan aman (misalnya melakukan `if size(img, 3) == 3, gray = rgb2gray(img); else, gray = img; end`).
3. **Konsistensi Tipe Data Output untuk Visualisasi GUI:**
   - Citra berwarna / grayscale hasil segmentasi dikembalikan dalam tipe **`uint8` (rentang 0–255)**.
   - Mask biner atau peta tepi biner dikembalikan dalam tipe **`logical` (0/1)** atau **`uint8` (0/255)**.
4. **Komentar Penjelasan (*Inline Comments*):**
   - Sesuai Spesifikasi A.4, setiap fungsi backend non-GUI wajib memiliki komentar penjelasan pada baris-baris algoritma kunci.

---

## 2. Kontrak Modul 1: Deteksi Tepi & Segmentasi Objek (Tab 1 GUI)
**Penanggung Jawab:** Anggota A  
**Lokasi Direktori:** `src/edge_segmentation/`

### 2.1 Fungsi Entry-Point Utama untuk GUI
Nama file: **`src/edge_segmentation/run_edge_segmentation.m`**

```matlab
function [edgeMap, objMask, segObjImg] = run_edge_segmentation(img, operator, params)
% RUN_EDGE_SEGMENTATION Menjalankan deteksi tepi dan segmentasi objek utama.
%
% Input:
%   img       : (uint8) Citra masukan RGB (H x W x 3) atau Grayscale (H x W).
%   operator  : (char/string) Pilihan operator deteksi tepi (case-insensitive):
%               'laplace' | 'log' | 'sobel' | 'prewitt' | 'roberts' | 'canny'
%   params    : (struct) Parameter konfigurasi dari komponen GUI:
%               - params.thresh     : (double) Nilai ambang deteksi tepi (rentang 0.0 - 1.0
%                                     untuk normalisasi atau 0 - 255, default sesuai operator).
%               - params.sigma      : (double) Standar deviasi Gaussian untuk LoG / Canny
%                                     (opsional, default = 1.0).
%               - params.morphRadius: (integer) Radius structuring element untuk imclose
%                                     (opsional, default = 3).
%               - params.minArea    : (integer) Luas minimum piksel objek untuk bwareafilt/filter
%                                     (opsional, default = 500).
%
% Output:
%   edgeMap   : (logical, H x W) Citra tepi biner hasil operator deteksi tepi.
%   objMask   : (logical, H x W) Mask biner objek utama setelah operasi morfologi
%               (closing -> hole filling -> area filtering).
%   segObjImg : (uint8, H x W x C) Citra objek hasil segmentasi yang telah dipisahkan
%               dari latar belakangnya (piksel background diset 0/hitam atau 255/putih).
```

### 2.2 Ketentuan Tambahan Modul 1
- Operator `'laplace'`, `'log'`, `'sobel'`, `'prewitt'`, dan `'roberts'` **wajib menggunakan implementasi konvolusi manual** (`my_conv2.m`), **dilarang** memakai `edge()` bawaan MATLAB.
- Hanya operator `'canny'` yang diperbolehkan menggunakan `edge(gray, 'canny', ...)` sesuai Spesifikasi B.1.
- Apabila ada field di `params` yang tidak dikirim atau kosong, fungsi harus menyediakan nilai *default* yang aman.

---

## 3. Kontrak Modul 2: Segmentasi Klasik Split and Merge (Tab 2 GUI)
**Penanggung Jawab:** Anggota A  
**Lokasi Direktori:** `src/split_merge/`

### 3.1 Fungsi Entry-Point Utama untuk GUI
Nama file: **`src/split_merge/split_and_merge.m`**

```matlab
function [splitVisImg, mergedImg, stats] = split_and_merge(img, splitThresh, mergeThresh, minBlock)
% SPLIT_AND_MERGE Melakukan segmentasi citra berbasis region dengan Quadtree Split & Merge.
%
% Input:
%   img         : (uint8) Citra masukan RGB (H x W x 3) atau Grayscale (H x W).
%   splitThresh : (double) Threshold predikat homogenitas untuk fase Split
%                 (misal batas max-min atau standar deviasi region, contoh rentang 5 - 100).
%   mergeThresh : (double) Threshold kemiripan untuk fase Merge antar region bertetangga
%                 (misal selisih rata-rata intensitas <= mergeThresh, contoh rentang 5 - 100).
%   minBlock    : (integer) Ukuran blok minimum quadtree dalam piksel (pangkat 2: 2, 4, 8, 16).
%
% Output:
%   splitVisImg : (uint8, H x W x C) Citra visualisasi hasil fase Split (menampilkan
%                 garis batas kuadran quadtree sebelum digabung).
%   mergedImg   : (uint8, H x W x C) Citra hasil akhir setelah fase Merge
%                 (setiap region diwarnai dengan rata-rata intensitas/warnanya atau label).
%   stats       : (struct) Informasi statistik untuk ditampilkan di status bar GUI:
%                 - stats.numSplitRegions  : (integer) Jumlah region setelah fase Split.
%                 - stats.numMergedRegions : (integer) Jumlah region akhir setelah fase Merge.
```

### 3.2 Ketentuan Tambahan Modul 2
- Jika ukuran citra masukan tidak berbentuk persegi pangkat 2 (misal citra BSDS500 berukuran `321 x 481`), fungsi `split_and_merge.m` wajib melakukan *padding* internal terlebih dahulu ke ukuran $2^k \times 2^k$ (atau menangani blok non-persegi) lalu memotong (*crop*) kembali hasilnya ke ukuran asli `H x W` sebelum dikembalikan ke GUI.
- Fungsi wajib berjalan dengan waktu respons yang wajar untuk penggunaan interaktif di GUI.

---

## 4. Kontrak Modul 3: Semantic Segmentation Deep Learning (Tab 3 GUI)
**Penanggung Jawab:** Anggota B  
**Lokasi Direktori:** `src/deep_learning/`

- GUI Tab 3 memanggil `infer_segmentation.py` (atau fungsi MATLAB `infer_segmentation.m`) dengan input citra RGB sembarang ukuran dan menghasilkan:
  1. `colorMask` (`uint8`, `H x W x 3`): Peta warna segmentasi 11 kelas CamVid (`Sky`, `Building`, `Pole`, `Road`, `Pavement`, `Tree`, `SignSymbol`, `Fence`, `Car`, `Pedestrian`, `Bicyclist`).
  2. `overlayImg` (`uint8`, `H x W x 3`): Gabungan citra asli dengan `colorMask` berdasarkan nilai slider *opacity* (`0.0 - 1.0`).

---

## 5. Kontrak Git & Alur Kerja Tim

| Branch | Pemilik | Ruang Lingkup Folder |
| :--- | :--- | :--- |
| `main` | Bersama | Rilis stabil setelah integrasi & review |
| `feat/edge-segmentation` | Anggota A | `src/edge_segmentation/` |
| `feat/split-merge` | Anggota A | `src/split_merge/` |
| `feat/deep-learning` | Anggota B | `src/deep_learning/` |
| `feat/gui` | Anggota B | `src/gui/` |

- **Aturan Merge:** Anggota A tidak mengubah isi `src/deep_learning/` dan `src/gui/`; Anggota B tidak mengubah isi `src/edge_segmentation/` dan `src/split_merge/` (kecuali stub sementara sebelum merge).
- Setiap perubahan pada kontrak di dokumen ini wajib didiskusikan dan disepakati kedua anggota.
