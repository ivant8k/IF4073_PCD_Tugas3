"""
camvid_mapping.py
-----------------
Modul persiapan dan prapemrosesan dataset CamVid untuk Tugas 3 IF4073.

Fungsi utama skrip ini:
1. Mengumpulkan seluruh 701 pasangan citra RGB dan label anotasi 32 kelas dari
   direktori train/val/test bawaan Kaggle.
2. Mengonversi citra label RGB (32 kelas asli) menjadi matriks indeks 2D
   berukuran 1 kanal (11 kelas target: Sky, Building, Pole, Road, Pavement,
   Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist) + nilai 255 untuk Void.
3. Membagi dataset secara deterministik dengan rasio 60% Train, 20% Validation,
   dan 20% Test serta menyimpan daftar file ke `splits/*.txt`.
4. Menghitung bobot kelas menggunakan Median Frequency Balancing untuk
   menangani ketimpangan jumlah piksel antar-kelas (class imbalance).
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

# KONSTANTA PEMETAAN 32 KELAS ASLI CAMVID -> 11 KELAS TARGET

# Urutan nama 11 kelas target sesuai spesifikasi Tugas 3 (indeks 0 s.d. 10)
CLASS_NAMES: List[str] = [
    "Sky",         # 0
    "Building",    # 1
    "Pole",        # 2
    "Road",        # 3
    "Pavement",    # 4
    "Tree",        # 5
    "SignSymbol",  # 6
    "Fence",       # 7
    "Car",         # 8
    "Pedestrian",  # 9
    "Bicyclist",   # 10
]

# Nilai indeks khusus untuk piksel Void / Unlabeled agar diabaikan saat loss & evaluasi
IGNORE_INDEX: int = 255

# Palet warna RGB representasi untuk masing-masing dari 11 kelas (untuk visualisasi)
PALETTE_11: np.ndarray = np.array([
    [128, 128, 128],  # 0: Sky
    [128,   0,   0],  # 1: Building
    [192, 192, 128],  # 2: Pole
    [128,  64, 128],  # 3: Road
    [  0,   0, 192],  # 4: Pavement
    [128, 128,   0],  # 5: Tree
    [192, 128, 128],  # 6: SignSymbol
    [ 64,  64, 128],  # 7: Fence
    [ 64,   0, 128],  # 8: Car
    [ 64,  64,   0],  # 9: Pedestrian
    [  0, 128, 192],  # 10: Bicyclist
], dtype=np.uint8)

# Pemetaan warna RGB asli (32 kelas dari class_dict.csv) -> indeks 11 kelas (0..10)
# Kelas 'Void' (0, 0, 0) sengaja tidak dimasukkan agar tetap bernilai IGNORE_INDEX (255).
RGB_TO_CLASS11: Dict[Tuple[int, int, int], int] = {
    # 0: Sky
    (128, 128, 128): 0,

    # 1: Building (gabungin Building, Bridge, Wall, Tunnel, Archway)
    (128,   0,   0): 1,  # Building
    (  0, 128,  64): 1,  # Bridge
    ( 64, 192,   0): 1,  # Wall
    ( 64,   0,  64): 1,  # Tunnel
    (192,   0, 128): 1,  # Archway

    # 2: Pole (gabungin Column_Pole dan TrafficCone)
    (192, 192, 128): 2,  # Column_Pole
    (  0,   0,  64): 2,  # TrafficCone

    # 3: Road (gabungin Road, LaneMkgsDriv, LaneMkgsNonDriv)
    (128,  64, 128): 3,  # Road
    (128,   0, 192): 3,  # LaneMkgsDriv
    (192,   0,  64): 3,  # LaneMkgsNonDriv

    # 4: Pavement (gabungin Sidewalk, ParkingBlock, RoadShoulder)
    (  0,   0, 192): 4,  # Sidewalk
    ( 64, 192, 128): 4,  # ParkingBlock
    (128, 128, 192): 4,  # RoadShoulder

    # 5: Tree (gabungin Tree dan VegetationMisc)
    (128, 128,   0): 5,  # Tree
    (192, 192,   0): 5,  # VegetationMisc

    # 6: SignSymbol (gabungin SignSymbol, Misc_Text, TrafficLight)
    (192, 128, 128): 6,  # SignSymbol
    (128, 128,  64): 6,  # Misc_Text
    (  0,  64,  64): 6,  # TrafficLight

    # 7: Fence
    ( 64,  64, 128): 7,  # Fence

    # 8: Car (gabungin Car, SUVPickupTruck, Truck_Bus, Train, OtherMoving)
    ( 64,   0, 128): 8,  # Car
    ( 64, 128, 192): 8,  # SUVPickupTruck
    (192, 128, 192): 8,  # Truck_Bus
    (192,  64, 128): 8,  # Train
    (128,  64,  64): 8,  # OtherMoving

    # 9: Pedestrian (gabungin Pedestrian, Child, CartLuggagePram, Animal)
    ( 64,  64,   0): 9,  # Pedestrian
    (192, 128,  64): 9,  # Child
    ( 64,   0, 192): 9,  # CartLuggagePram
    ( 64, 128,  64): 9,  # Animal

    # 10: Bicyclist (gabungin Bicyclist dan MotorcycleScooter)
    (  0, 128, 192): 10, # Bicyclist
    (192,   0, 192): 10, # MotorcycleScooter
}

# Bangun Lookup Table (LUT) 24-bit untuk konversi vektorisasi berkecepatan tinggi.
# Setiap triplet (R, G, B) dikodekan menjadi integer tunggal: (R << 16) | (G << 8) | B.
_LUT = np.full(256 * 256 * 256, IGNORE_INDEX, dtype=np.uint8)
for (r, g, b), cls_id in RGB_TO_CLASS11.items():
    _LUT[(r << 16) | (g << 8) | b] = cls_id


def convert_rgb_to_mask11(label_rgb: np.ndarray) -> np.ndarray:
    """
    Mengonversi citra label RGB (H, W, 3) menjadi matriks indeks kelas 2D (H, W)
    bernilai 0..10 dan 255 untuk piksel Void/Unlabeled menggunakan Lookup Table (LUT).
    """
    # Pastikan tipe data uint32 sebelum operasi bit-shift agar tidak overflow
    r = label_rgb[:, :, 0].astype(np.uint32)
    g = label_rgb[:, :, 1].astype(np.uint32)
    b = label_rgb[:, :, 2].astype(np.uint32)

    # Gabungkan 3 kanal RGB menjadi 1 kode integer unik per piksel, lalu petakan via LUT
    packed_rgb = (r << 16) | (g << 8) | b
    return _LUT[packed_rgb]


def collect_camvid_pairs(camvid_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Mengumpulkan seluruh pasangan (citra_asli, label_rgb) dari sub-folder
    train/val/test bawaan dataset Kaggle CamVid.
    """
    pairs: List[Tuple[Path, Path]] = []

    for subset in ("train", "val", "test"):
        img_dir = camvid_dir / subset
        lbl_dir = camvid_dir / f"{subset}_labels"
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            continue

        for img_path in sorted(img_dir.glob("*.png")):
            # Konvensi penamaan CamVid: <nama>.png <-> <nama>_L.png
            lbl_path = lbl_dir / f"{img_path.stem}_L.png"
            if not lbl_path.is_file():
                raise FileNotFoundError(f"Label tidak ditemukan untuk citra: {img_path}")
            pairs.append((img_path, lbl_path))

    return pairs


def split_dataset(
    items: List[Tuple[str, str]],
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Membagi daftar pasangan data secara acak namun deterministik (reproducible)
    menjadi Train (60%), Validation (20%), dan Test (20%).
    """
    shuffled = list(items)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    total = len(shuffled)
    n_train = round(total * train_ratio)
    n_val = round(total * val_ratio)

    train_items = shuffled[:n_train]
    val_items = shuffled[n_train : n_train + n_val]
    test_items = shuffled[n_train + n_val :]
    return train_items, val_items, test_items


def compute_median_frequency_weights(
    pixel_counts: np.ndarray,
    image_presence_counts: np.ndarray,
) -> List[float]:
    """
    Menghitung bobot kelas dengan metode Median Frequency Balancing:
      freq_c = jumlah_piksel_kelas_c / total_piksel_pada_citra_yang_memuat_kelas_c
      weight_c = median(freq) / freq_c
    Digunakan pada fungsi loss untuk menyeimbangkan kelas kecil (Pole, Bicyclist, dll.).
    """
    # Hindari pembagian nol jika ada kelas yang sangat jarang
    safe_presence = np.maximum(image_presence_counts, 1.0)
    freqs = pixel_counts / safe_presence
    median_freq = float(np.median(freqs[freqs > 0]))
    weights = np.where(freqs > 0, median_freq / freqs, 1.0)
    return [round(float(w), 6) for w in weights]


def run_preprocessing(camvid_dir: Path, seed: int = 42) -> None:
    """Menjalankan seluruh pipeline konversi label, split 60/20/20, dan hitung bobot kelas."""
    pairs = collect_camvid_pairs(camvid_dir)
    if len(pairs) == 0:
        raise RuntimeError(f"Tidak ada pasangan citra/label CamVid ditemukan di {camvid_dir}")

    masks_out_dir = camvid_dir / "masks_11"
    splits_out_dir = camvid_dir / "splits"
    masks_out_dir.mkdir(parents=True, exist_ok=True)
    splits_out_dir.mkdir(parents=True, exist_ok=True)

    rel_pairs: List[Tuple[str, str]] = []
    class_pixel_counts = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    class_image_total_pixels = np.zeros(len(CLASS_NAMES), dtype=np.float64)

    print(f"[INFO] Memproses {len(pairs)} pasangan citra CamVid dari: {camvid_dir}")

    for img_path, lbl_path in pairs:
        # Baca citra label dalam mode RGB (menghindari mode palette/RGBA)
        lbl_rgb = np.array(Image.open(lbl_path).convert("RGB"), dtype=np.uint8)
        mask_11 = convert_rgb_to_mask11(lbl_rgb)

        # Simpan mask 1-kanal sebagai PNG lossless (jangan pernah pakai JPG!)
        mask_filename = f"{img_path.stem}_mask11.png"
        mask_save_path = masks_out_dir / mask_filename
        Image.fromarray(mask_11, mode="L").save(mask_save_path)

        # Simpan path relatif terhadap camvid_dir agar portabel lintas sistem operasi
        rel_img = img_path.relative_to(camvid_dir).as_posix()
        rel_mask = mask_save_path.relative_to(camvid_dir).as_posix()
        rel_pairs.append((rel_img, rel_mask))

        # Akumulasi statistik piksel untuk perhitungan Median Frequency Balancing
        num_img_pixels = float(mask_11.size)
        counts = np.bincount(mask_11[mask_11 != IGNORE_INDEX], minlength=len(CLASS_NAMES))
        class_pixel_counts += counts[: len(CLASS_NAMES)]
        present_classes = counts[: len(CLASS_NAMES)] > 0
        class_image_total_pixels[present_classes] += num_img_pixels

    # Bagi dataset 60% Train / 20% Validation / 20% Test
    train_set, val_set, test_set = split_dataset(rel_pairs, 0.60, 0.20, seed=seed)

    for split_name, subset in (("train", train_set), ("val", val_set), ("test", test_set)):
        split_file = splits_out_dir / f"{split_name}.txt"
        with split_file.open("w", encoding="utf-8") as f:
            for rel_img, rel_mask in subset:
                f.write(f"{rel_img},{rel_mask}\n")

    # Hitung bobot kelas dan simpan metadata kelas ke JSON
    class_weights = compute_median_frequency_weights(
        class_pixel_counts, class_image_total_pixels
    )
    metadata = {
        "total_images": len(rel_pairs),
        "split_counts": {
            "train": len(train_set),
            "val": len(val_set),
            "test": len(test_set),
        },
        "seed": seed,
        "ignore_index": IGNORE_INDEX,
        "class_names": CLASS_NAMES,
        "palette_rgb": PALETTE_11.tolist(),
        "pixel_counts": [int(x) for x in class_pixel_counts],
        "median_freq_weights": class_weights,
    }
    with (splits_out_dir / "camvid_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f">>> Konversi 11 kelas selesai -> {masks_out_dir}")
    print(f">>> Split (seed={seed}): Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
    print(f">>> Metadata & class weights disimpan -> {splits_out_dir / 'camvid_metadata.json'}")


if __name__ == "__main__":
    # Secara default mengarah ke <repo_root>/data/camvid relatif terhadap lokasi script ini
    default_camvid_dir = Path(__file__).resolve().parents[2] / "data" / "camvid"
    parser = argparse.ArgumentParser(description="Preprocessing CamVid 32->11 kelas dan split 60/20/20.")
    parser.add_argument("--camvid-dir", type=Path, default=default_camvid_dir, help="Direktori dataset CamVid.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed untuk split 60/20/20.")
    args = parser.parse_args()

    run_preprocessing(args.camvid_dir, seed=args.seed)
