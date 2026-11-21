"""
infer_segmentation.py
---------------------
Skrip dan modul inferensi Semantic Segmentation (11 kelas CamVid) menggunakan
model DeepLabv3+ (backbone MobileNetV3-Large) yang telah di-fine-tune.

Mendukung dua mode pemakaian:
1. CLI (untuk terminal atau pemanggilan dari GUI MATLAB via `system()`):
   python infer_segmentation.py <input_image> --output-dir <dir> --opacity 0.5
   python infer_segmentation.py <input_image> <overlay_out> --color-out <color> --mask-out <mask>
2. Modul Python (untuk import langsung atau `pyrunfile()` dari MATLAB):
   from infer_segmentation import load_model, predict_segmentation, generate_legend_image
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Backend non-interaktif agar aman dipanggil dari CLI/MATLAB tanpa GUI pop-up
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.segmentation as seg_models
import torchvision.transforms.functional as TF

# Pastikan direktori modul ini ada di sys.path saat dipanggil dari direktori kerja lain (misal MATLAB)
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from camvid_mapping import CLASS_NAMES, IGNORE_INDEX, PALETTE_11  # noqa: E402

# Konstanta prapemrosesan yang wajib identik dengan konfigurasi saat training
INPUT_SIZE: Tuple[int, int] = (352, 480)  # (Height, Width), keduanya kelipatan 32
IMG_MEAN: List[float] = [0.485, 0.456, 0.406]
IMG_STD: List[float] = [0.229, 0.224, 0.225]

DEFAULT_WEIGHTS_PATH: Path = _THIS_DIR / "weights" / "best_model.pth"
DEFAULT_OUTPUT_DIR: Path = _THIS_DIR / "outputs"


def build_deeplabv3_mobilenet(
    num_classes: int = len(CLASS_NAMES),
    pretrained: bool = False,
    aux_loss: Optional[bool] = None,
) -> nn.Module:
    """
    Membangun arsitektur DeepLabv3+ dengan backbone MobileNetV3-Large.

    Parameter `pretrained=False` digunakan saat inferensi/evaluasi agar PyTorch
    tidak mencoba mengunduh bobot dari internet (mendukung eksekusi offline di GUI MATLAB).
    Parameter `pretrained=True` digunakan saat training untuk memuat bobot awal COCO/ImageNet.
    """
    weights = (
        seg_models.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        if pretrained
        else None
    )
    model = seg_models.deeplabv3_mobilenet_v3_large(
        weights=weights,
        weights_backbone=None,
        aux_loss=aux_loss,
    )

    # Ganti classifier head terakhir agar jumlah kanal keluaran sesuai 11 kelas CamVid
    in_ch = model.classifier[0].convs[0][0].in_channels
    model.classifier = seg_models.deeplabv3.DeepLabHead(in_ch, num_classes)

    # Sesuaikan auxiliary classifier apabila tersedia pada arsitektur
    if model.aux_classifier is not None:
        in_ch_aux = model.aux_classifier[0].in_channels
        model.aux_classifier = seg_models.fcn.FCNHead(in_ch_aux, num_classes)

    return model


def resolve_device(device_str: str = "auto") -> torch.device:
    """Menentukan perangkat komputasi (CUDA GPU atau CPU)."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_model(
    weights_path: Union[str, Path] = DEFAULT_WEIGHTS_PATH,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[nn.Module, torch.device]:
    """
    Memuat arsitektur DeepLabv3+ beserta bobot terlatih lokal dari file `.pth`.
    """
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"File bobot model tidak ditemukan: {weights_path}. "
            "Jalankan train_model.py terlebih dahulu untuk menghasilkan best_model.pth."
        )

    target_device = (
        resolve_device("auto")
        if device is None
        else (torch.device(device) if isinstance(device, str) else device)
    )

    # Muat checkpoint lokal ke perangkat target
    checkpoint = torch.load(weights_path, map_location=target_device, weights_only=False)
    num_classes = int(checkpoint.get("num_classes", len(CLASS_NAMES)))
    state_dict = checkpoint.get("model_state", checkpoint)
    has_aux = any(k.startswith("aux_classifier.") for k in state_dict)

    # Inisialisasi arsitektur tanpa mengunduh bobot internet (offline-safe)
    model = build_deeplabv3_mobilenet(
        num_classes=num_classes,
        pretrained=False,
        aux_loss=has_aux,
    )
    model.load_state_dict(state_dict)
    model.to(target_device)
    model.eval()
    return model, target_device


def mask_to_colormap(mask: np.ndarray, palette: np.ndarray = PALETTE_11) -> np.ndarray:
    """
    Memetakan matriks indeks kelas 2D (H, W) bernilai 0..10 menjadi
    citra warna RGB (H, W, 3) bertipe uint8 menggunakan palet 11 kelas CamVid.
    """
    # Pengindeksan vektorisasi langsung: piksel di luar 0..10 (misal Void=255) diberi warna hitam
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(palette))
    color_mask[valid] = palette[mask[valid]]
    return color_mask


def create_overlay(
    image_rgb: np.ndarray,
    color_mask: np.ndarray,
    opacity: float = 0.5,
) -> np.ndarray:
    """
    Menggabungkan citra asli RGB dengan peta warna segmentasi menggunakan alpha-blending:
      overlay = (1 - opacity) * citra_asli + opacity * color_mask
    """
    alpha = float(np.clip(opacity, 0.0, 1.0))
    blended = (
        (1.0 - alpha) * image_rgb.astype(np.float32)
        + alpha * color_mask.astype(np.float32)
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def get_legend_metadata() -> List[Dict[str, object]]:
    """Mengembalikan daftar lengkap 11 kelas beserta indeks dan kode warna RGB."""
    return [
        {"id": idx, "name": name, "rgb": PALETTE_11[idx].tolist()}
        for idx, name in enumerate(CLASS_NAMES)
    ]


def generate_legend_image(output_path: Optional[Union[str, Path]] = None) -> np.ndarray:
    """
    Membuat citra legend 11 kelas CamVid berwarna (uint8 RGB) dan menyimpannya jika
    `output_path` diberikan. Berguna untuk ditampilkan di Tab 3 GUI MATLAB atau laporan.
    """
    fig, ax = plt.subplots(figsize=(10, 1.6))
    for i, (name, rgb) in enumerate(zip(CLASS_NAMES, PALETTE_11)):
        ax.barh(0, 1, left=i, color=rgb / 255.0, edgecolor="white", linewidth=1.2)
        # Pilih warna teks kontras (hitam pada warna terang seperti Pole)
        luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        text_color = "black" if luminance > 155 else "white"
        ax.text(
            i + 0.5, 0, name,
            ha="center", va="center", fontsize=8.5, fontweight="bold", color=text_color,
        )
    ax.set_xlim(0, len(CLASS_NAMES))
    ax.axis("off")
    ax.set_title("Legend 11 Kelas CamVid", fontsize=10, fontweight="bold", pad=6)
    plt.tight_layout()

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    legend_rgb = rgba[:, :, :3].copy()
    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(legend_rgb, mode="RGB").save(out_p)
    plt.close(fig)
    return legend_rgb


def predict_segmentation(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    model: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    opacity: float = 0.5,
    weights_path: Union[str, Path] = DEFAULT_WEIGHTS_PATH,
) -> Dict[str, object]:
    """
    Menjalankan inferensi semantic segmentation pada satu citra RGB berukuran sembarang.

    Langkah inferensi:
    1. Baca citra masukan dalam format RGB dan catat ukuran aslinya (H_asli, W_asli).
    2. Resize ke ukuran input jaringan (352 x 480) dan normalisasi dengan statistik ImageNet.
    3. Forward pass tanpa gradien untuk memperoleh peta logit (1, 11, 352, 480).
    4. Interpolasi bilinear logit kembali ke resolusi asli agar tepi kelas halus,
       lalu ambil argmax per piksel untuk mendapatkan mask label (0..10).
    5. Petakan mask indeks menjadi citra warna RGB dan gabungkan sebagai overlay.
    """
    if model is None or device is None:
        model, device = load_model(weights_path=weights_path, device=device)

    # 1. Baca citra sebagai RGB (aman untuk masukan grayscale, RGB, maupun RGBA)
    if isinstance(image_input, (str, Path)):
        pil_orig = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_orig = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        arr = image_input.astype(np.uint8)
        if arr.ndim == 2:
            pil_orig = Image.fromarray(arr, mode="L").convert("RGB")
        else:
            pil_orig = Image.fromarray(arr[:, :, :3], mode="RGB")
    else:
        raise TypeError(f"Tipe image_input tidak didukung: {type(image_input)}")

    orig_w, orig_h = pil_orig.size
    orig_rgb = np.array(pil_orig, dtype=np.uint8)

    # 2. Prapemrosesan: resize ke resolusi jaringan & normalisasi ImageNet
    resized = TF.resize(pil_orig, INPUT_SIZE, interpolation=TF.InterpolationMode.BILINEAR)
    tensor_in = TF.normalize(TF.to_tensor(resized), IMG_MEAN, IMG_STD).unsqueeze(0).to(device)

    # 3. Forward pass inferensi tanpa pelacakan gradien
    with torch.no_grad():
        logits = model(tensor_in)["out"]  # (1, 11, INPUT_H, INPUT_W)
        # 4. Upsample logit ke ukuran asli sebelum argmax agar batas antar-objek lebih presisi
        logits_orig = F.interpolate(
            logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )
        label_mask = logits_orig.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # 5. Pemetaan warna 11 kelas & pembuatan citra overlay
    color_mask = mask_to_colormap(label_mask, PALETTE_11)
    overlay_img = create_overlay(orig_rgb, color_mask, opacity=opacity)

    # Hitung persentase piksel tiap kelas pada citra hasil prediksi
    total_px = float(label_mask.size)
    counts = np.bincount(label_mask.ravel(), minlength=len(CLASS_NAMES))
    class_percentages = {
        name: round(float(counts[i] / total_px * 100.0), 2)
        for i, name in enumerate(CLASS_NAMES)
    }

    return {
        "label_mask": label_mask,
        "color_mask": color_mask,
        "overlay_img": overlay_img,
        "legend": get_legend_metadata(),
        "class_percentages": class_percentages,
        "input_size": [orig_h, orig_w],
    }


def save_outputs(
    result: Dict[str, object],
    mask_out: Optional[Path],
    color_out: Optional[Path],
    overlay_out: Optional[Path],
    legend_out: Optional[Path],
    json_out: Optional[Path],
) -> None:
    """Menyimpan artefak output inferensi (PNG lossless dan JSON ringkasan)."""
    if mask_out is not None:
        mask_out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result["label_mask"], mode="L").save(mask_out)

    if color_out is not None:
        color_out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result["color_mask"], mode="RGB").save(color_out)

    if overlay_out is not None:
        overlay_out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result["overlay_img"], mode="RGB").save(overlay_out)

    if legend_out is not None:
        generate_legend_image(legend_out)

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_size_hw": result["input_size"],
            "ignore_index": IGNORE_INDEX,
            "legend": result["legend"],
            "class_percentages": result["class_percentages"],
        }
        with json_out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inferensi Semantic Segmentation CamVid (11 Kelas) dengan DeepLabv3+."
    )
    parser.add_argument("input_image", type=Path, help="Path citra masukan (PNG/JPG).")
    parser.add_argument(
        "output_overlay",
        type=Path,
        nargs="?",
        default=None,
        help="Opsional: path langsung output overlay (misal dipanggil via system()).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Direktori penyimpanan mask, color_mask, overlay, legend, dan metadata.json.",
    )
    parser.add_argument("--mask-out", type=Path, default=None, help="Path output mask indeks (0..10).")
    parser.add_argument("--color-out", type=Path, default=None, help="Path output peta warna RGB.")
    parser.add_argument("--overlay-out", type=Path, default=None, help="Path output citra overlay RGB.")
    parser.add_argument("--legend-out", type=Path, default=None, help="Path output gambar legend 11 kelas.")
    parser.add_argument("--json-out", type=Path, default=None, help="Path output metadata JSON.")
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.5,
        help="Tingkat transparansi mask pada overlay (0.0 - 1.0, default=0.5).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS_PATH,
        help="Path file bobot model terlatih (.pth).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Perangkat inferensi (default=auto).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_image.is_file():
        raise FileNotFoundError(f"Citra masukan tidak ditemukan: {args.input_image}")

    # Tentukan path output secara fleksibel agar kompatibel dengan berbagai pola panggilan CLI/MATLAB
    overlay_out = args.overlay_out or args.output_overlay
    mask_out = args.mask_out
    color_out = args.color_out
    legend_out = args.legend_out
    json_out = args.json_out

    if args.output_dir is not None:
        stem = args.input_image.stem
        out_dir = args.output_dir
        mask_out = mask_out or (out_dir / f"{stem}_mask.png")
        color_out = color_out or (out_dir / f"{stem}_color.png")
        overlay_out = overlay_out or (out_dir / f"{stem}_overlay.png")
        legend_out = legend_out or (out_dir / "legend_11_classes.png")
        json_out = json_out or (out_dir / f"{stem}_meta.json")
    elif overlay_out is not None:
        # Jika hanya overlay posisional diberikan, simpan mask & color mask berdampingan bila belum diset
        parent = overlay_out.parent
        stem = args.input_image.stem
        mask_out = mask_out or (parent / f"{stem}_mask.png")
        color_out = color_out or (parent / f"{stem}_color.png")
        legend_out = legend_out or (parent / "legend_11_classes.png")
    elif all(p is None for p in (mask_out, color_out, overlay_out, legend_out, json_out)):
        stem = args.input_image.stem
        mask_out = DEFAULT_OUTPUT_DIR / f"{stem}_mask.png"
        color_out = DEFAULT_OUTPUT_DIR / f"{stem}_color.png"
        overlay_out = DEFAULT_OUTPUT_DIR / f"{stem}_overlay.png"
        legend_out = DEFAULT_OUTPUT_DIR / "legend_11_classes.png"
        json_out = DEFAULT_OUTPUT_DIR / f"{stem}_meta.json"

    model, device = load_model(weights_path=args.weights, device=resolve_device(args.device))
    result = predict_segmentation(
        args.input_image,
        model=model,
        device=device,
        opacity=args.opacity,
    )
    save_outputs(result, mask_out, color_out, overlay_out, legend_out, json_out)

    print("Inferensi Semantic Segmentation Selesai")
    print("-" * 60)
    print(f"Input       : {args.input_image} ({result['input_size'][0]}x{result['input_size'][1]})")
    print(f"Device      : {device}")
    if mask_out:
        print(f"Label Mask  : {mask_out}")
    if color_out:
        print(f"Color Mask  : {color_out}")
    if overlay_out:
        print(f"Overlay     : {overlay_out} (opacity={args.opacity:.2f})")
    if legend_out:
        print(f"Legend      : {legend_out}")
    if json_out:
        print(f"Metadata    : {json_out}")


if __name__ == "__main__":
    main()
