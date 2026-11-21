"""
train_model.py
------------------------------------------
Skrip pelatihan penuh (transfer learning / fine-tuning) model DeepLabv3+
(backbone MobileNetV3-Large) pada dataset CamVid 11 kelas.

Fitur utama:
1. Memuat hyperparameter terbaik hasil Bayesian Optimization dari
   `weights/best_hyperparams.json` secara otomatis (dapat di-override via CLI).
2. Menggunakan data split deterministik 60% Train / 20% Val / 20% Test dari
   `data/camvid/splits/` hasil `camvid_mapping.py`.
3. Menyimpan bobot terbaik berdasarkan Val mIoU ke `weights/best_model.pth`.
4. Menyimpan riwayat loss/metrik (`weights/loss_history.json`), kurva pelatihan
   (`weights/training_curves.png`), dan evaluasi akhir pada Test Set
   (`weights/test_evaluation.json`, `weights/per_class_iou.png`,
   `weights/confusion_matrix.png`) untuk laporan.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
from tqdm import tqdm

# Pastikan modul sibling di src/deep_learning/ dapat diimpor dari working directory mana pun
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from camvid_mapping import CLASS_NAMES, IGNORE_INDEX, PALETTE_11  # noqa: E402
from infer_segmentation import (  # noqa: E402
    IMG_MEAN,
    IMG_STD,
    INPUT_SIZE,
    build_deeplabv3_mobilenet,
)

NUM_CLASSES: int = len(CLASS_NAMES)
REPO_ROOT: Path = _THIS_DIR.parents[1]
DEFAULT_CAMVID_DIR: Path = REPO_ROOT / "data" / "camvid"
DEFAULT_WEIGHTS_DIR: Path = _THIS_DIR / "weights"

# Nilai fallback apabila best_hyperparams.json belum tersedia (hasil Trial #11 BO)
DEFAULT_HYPERPARAMS: Dict[str, object] = {
    "lr": 0.0006102257955906041,
    "weight_decay": 2.1580683806084493e-05,
    "batch_size": 4,
    "scheduler": "plateau",
    "use_class_weights": False,
}


def set_seed(seed: int = 42) -> None:
    """Mengatur seed acak agar eksperimen training bersifat reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_best_hyperparams(json_path: Path) -> Dict[str, object]:
    """Memuat hyperparameter terbaik dari `best_hyperparams.json` hasil notebook."""
    params = dict(DEFAULT_HYPERPARAMS)
    if json_path.is_file():
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        best = payload.get("best_params", {})
        params.update(best)
    return params


def load_split_pairs(camvid_dir: Path, split_name: str) -> List[Tuple[Path, Path]]:
    """Membaca daftar pasangan (citra, mask_11) dari `splits/<split_name>.txt`."""
    split_file = camvid_dir / "splits" / f"{split_name}.txt"
    if not split_file.is_file():
        raise FileNotFoundError(
            f"File split tidak ditemukan: {split_file}. Jalankan camvid_mapping.py dulu."
        )
    pairs: List[Tuple[Path, Path]] = []
    with split_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel_img, rel_mask = line.split(",")
            pairs.append((camvid_dir / rel_img, camvid_dir / rel_mask))
    return pairs


def load_class_weights(camvid_dir: Path) -> torch.Tensor:
    """Memuat bobot kelas Median Frequency Balancing dari `camvid_metadata.json`."""
    meta_path = camvid_dir / "splits" / "camvid_metadata.json"
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return torch.tensor(meta["median_freq_weights"], dtype=torch.float32)


class CamVidDataset(Dataset):
    """PyTorch Dataset untuk pasangan citra RGB dan mask indeks 11 kelas CamVid."""

    def __init__(self, pairs: List[Tuple[Path, Path]], augment: bool = False) -> None:
        self.pairs = pairs
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.pairs[idx]

        # Baca citra sebagai RGB dan mask sebagai citra indeks 1-kanal (mode L)
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Terapkan horizontal flip sinkron menggunakan satu angka acak untuk citra & mask
        if self.augment and random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # Resize: citra dengan BILINEAR, mask WAJIB dengan NEAREST agar indeks kelas tidak tercampur
        img = TF.resize(img, INPUT_SIZE, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, INPUT_SIZE, interpolation=TF.InterpolationMode.NEAREST)

        # Konversi ke tensor & normalisasi dengan statistik ImageNet
        img_t = TF.normalize(TF.to_tensor(img), IMG_MEAN, IMG_STD)
        mask_t = torch.as_tensor(np.array(mask), dtype=torch.long)
        return img_t, mask_t


def evaluate_dataset(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, object]:
    """
    Mengevaluasi model pada satu split (Validation / Test) dengan mengakumulasi
    confusion matrix 11x11 secara global (mengabaikan piksel Void=255).

    Menghasilkan:
    - loss: rata-rata loss
    - pixel_acc: Global Pixel Accuracy (0..1)
    - mean_iou: rata-rata IoU pada kelas yang muncul di target/prediksi
    - per_class_iou: dictionary {nama_kelas: IoU}
    - confusion_matrix: matriks 11x11 (baris=ground truth, kolom=prediksi)
    """
    model.eval()
    loss_sum = 0.0
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    with torch.no_grad():
        for imgs_b, masks_b in loader:
            imgs_b = imgs_b.to(device)
            masks_b = masks_b.to(device)

            logits = model(imgs_b)["out"]
            loss = criterion(logits, masks_b)
            loss_sum += loss.item()

            preds = logits.argmax(dim=1).cpu().numpy()
            targets = masks_b.cpu().numpy()

            # Kecualikan piksel Void (IGNORE_INDEX = 255) dari perhitungan metrik
            valid = targets != IGNORE_INDEX
            t_valid = targets[valid]
            p_valid = preds[valid]

            # Akumulasi confusion matrix via bincount terindeks 2D
            indices = t_valid * NUM_CLASSES + p_valid
            conf_mat += np.bincount(indices, minlength=NUM_CLASSES * NUM_CLASSES).reshape(
                NUM_CLASSES, NUM_CLASSES
            )

    total_valid = conf_mat.sum()
    pixel_acc = float(np.trace(conf_mat) / max(total_valid, 1))

    # Hitung IoU per kelas: TP / (TP + FP + FN)
    tp = np.diag(conf_mat).astype(np.float64)
    fp = conf_mat.sum(axis=0).astype(np.float64) - tp
    fn = conf_mat.sum(axis=1).astype(np.float64) - tp
    union = tp + fp + fn

    per_class_iou: Dict[str, Optional[float]] = {}
    valid_ious: List[float] = []
    for i, cls_name in enumerate(CLASS_NAMES):
        if union[i] > 0:
            iou_val = float(tp[i] / union[i])
            per_class_iou[cls_name] = round(iou_val, 6)
            valid_ious.append(iou_val)
        else:
            per_class_iou[cls_name] = None

    mean_iou = float(np.mean(valid_ious)) if valid_ious else 0.0
    return {
        "loss": float(loss_sum / max(len(loader), 1)),
        "pixel_acc": pixel_acc,
        "mean_iou": mean_iou,
        "per_class_iou": per_class_iou,
        "confusion_matrix": conf_mat,
    }


def save_training_curves(history: Dict[str, List[float]], out_path: Path) -> None:
    """Menyimpan grafik Loss, Pixel Accuracy, dan mIoU selama training untuk laporan."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", linewidth=1.8)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", linewidth=1.8)
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [v * 100 for v in history["val_acc"]], color="green", linewidth=1.8)
    axes[1].set_title("Validation Pixel Accuracy (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, [v * 100 for v in history["val_miou"]], color="darkorange", linewidth=1.8)
    axes[2].set_title("Validation mIoU (%)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("mIoU (%)")
    axes[2].set_ylim(0, 100)
    axes[2].grid(alpha=0.3)

    plt.suptitle("Riwayat Pelatihan DeepLabv3+ MobileNetV3 (CamVid 11 Kelas)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def save_test_plots(
    per_class_iou: Dict[str, Optional[float]],
    conf_mat: np.ndarray,
    weights_dir: Path,
) -> None:
    """Menyimpan grafik bar IoU per kelas dan normalized confusion matrix pada Test Set."""
    # 1. Grafik IoU per kelas
    values = [(per_class_iou[name] or 0.0) * 100.0 for name in CLASS_NAMES]
    colors = [PALETTE_11[i] / 255.0 for i in range(NUM_CLASSES)]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    bars = ax.bar(CLASS_NAMES, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("IoU (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Test Set IoU per Kelas (CamVid 11 Kelas)", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=8.5,
        )
    plt.tight_layout()
    plt.savefig(weights_dir / "per_class_iou.png", dpi=150)
    plt.close(fig)

    # 2. Confusion matrix ternormalisasi per baris (ground truth)
    row_sums = np.maximum(conf_mat.sum(axis=1, keepdims=True), 1)
    cm_norm = conf_mat.astype(np.float64) / row_sums * 100.0

    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=100)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Persentase (%)")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=40, ha="right", fontsize=8.5)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8.5)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Normalized Confusion Matrix - Test Set CamVid", fontsize=12, fontweight="bold")

    for r in range(NUM_CLASSES):
        for c in range(NUM_CLASSES):
            val = cm_norm[r, c]
            txt_color = "white" if val > 50 else "black"
            ax.text(c, r, f"{val:.1f}", ha="center", va="center", color=txt_color, fontsize=7)

    plt.tight_layout()
    plt.savefig(weights_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def parse_args(defaults: Dict[str, object]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tuning DeepLabv3+ MobileNetV3 pada CamVid 11 kelas."
    )
    parser.add_argument("--camvid-dir", type=Path, default=DEFAULT_CAMVID_DIR)
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--epochs", type=int, default=40, help="Jumlah epoch training penuh.")
    parser.add_argument("--lr", type=float, default=float(defaults["lr"]))
    parser.add_argument("--weight-decay", type=float, default=float(defaults["weight_decay"]))
    parser.add_argument("--batch-size", type=int, default=int(defaults["batch_size"]))
    parser.add_argument(
        "--scheduler",
        type=str,
        default=str(defaults["scheduler"]),
        choices=["plateau", "cosine"],
    )
    parser.add_argument(
        "--use-class-weights",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults["use_class_weights"]),
        help="Aktifkan/nonaktifkan Median Frequency class weights pada CrossEntropyLoss.",
    )
    parser.add_argument(
        "--early-stop",
        type=int,
        default=12,
        help="Hentikan training lebih awal jika Val mIoU tidak membaik selama N epoch (0=nonaktif).",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="Jumlah worker DataLoader (0 aman di Windows).")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    best_hp_file = DEFAULT_WEIGHTS_DIR / "best_hyperparams.json"
    hp_defaults = load_best_hyperparams(best_hp_file)
    args = parse_args(hp_defaults)

    set_seed(args.seed)
    args.weights_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_mem = device.type == "cuda"

    # Siapkan dataset split 60% / 20% / 20%
    train_pairs = load_split_pairs(args.camvid_dir, "train")
    val_pairs = load_split_pairs(args.camvid_dir, "val")
    test_pairs = load_split_pairs(args.camvid_dir, "test")

    train_loader = DataLoader(
        CamVidDataset(train_pairs, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,  # Hindari batch ukuran 1 agar BatchNorm2d ASPPPooling tidak error
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        CamVidDataset(val_pairs, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )
    test_loader = DataLoader(
        CamVidDataset(test_pairs, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )

    # Muat arsitektur pretrained ImageNet dan siapkan loss, optimizer, scheduler
    model = build_deeplabv3_mobilenet(num_classes=NUM_CLASSES, pretrained=True).to(device)
    loss_weights = load_class_weights(args.camvid_dir).to(device) if args.use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=loss_weights, ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )

    print("Konfigurasi Training Penuh (DeepLabv3+ MobileNetV3)")
    print("-" * 60)
    print(f"Device            : {device}" + (f" ({torch.cuda.get_device_name(0)})" if pin_mem else ""))
    print(f"Split Dataset     : Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")
    print(f"Epochs            : {args.epochs} (early_stop={args.early_stop})")
    print(f"Learning Rate     : {args.lr:.6g}")
    print(f"Weight Decay      : {args.weight_decay:.6g}")
    print(f"Batch Size        : {args.batch_size}")
    print(f"Scheduler         : {args.scheduler}")
    print(f"Use Class Weights : {args.use_class_weights}")
    print("-" * 60)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_miou": [],
        "lr": [],
    }

    best_val_miou = -1.0
    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    best_model_path = args.weights_dir / "best_model.pth"
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for imgs_b, masks_b in tqdm(
            train_loader, desc=f"Epoch {epoch:2d}/{args.epochs} [Train]", leave=False
        ):
            imgs_b = imgs_b.to(device)
            masks_b = masks_b.to(device)

            optimizer.zero_grad()
            logits = model(imgs_b)["out"]
            loss = criterion(logits, masks_b)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / len(train_loader)
        val_metrics = evaluate_dataset(model, val_loader, criterion, device)
        val_loss = float(val_metrics["loss"])
        val_acc = float(val_metrics["pixel_acc"])
        val_miou = float(val_metrics["mean_iou"])
        current_lr = float(optimizer.param_groups[0]["lr"])

        if args.scheduler == "plateau":
            scheduler.step(val_miou)
        else:
            scheduler.step()

        history["train_loss"].append(round(avg_train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_acc"].append(round(val_acc, 6))
        history["val_miou"].append(round(val_miou, 6))
        history["lr"].append(current_lr)

        improved_tag = ""
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            improved_tag = " [BEST]"

            # Simpan bobot model terbaik beserta metadata yang dibutuhkan inferensi
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "best_val_miou": best_val_miou,
                    "best_val_acc": best_val_acc,
                    "per_class_iou_val": val_metrics["per_class_iou"],
                    "num_classes": NUM_CLASSES,
                    "class_names": CLASS_NAMES,
                    "input_size": INPUT_SIZE,
                    "img_mean": IMG_MEAN,
                    "img_std": IMG_STD,
                    "palette_rgb": PALETTE_11.tolist(),
                    "hyperparams": {
                        "lr": args.lr,
                        "weight_decay": args.weight_decay,
                        "batch_size": args.batch_size,
                        "scheduler": args.scheduler,
                        "use_class_weights": args.use_class_weights,
                        "seed": args.seed,
                    },
                },
                best_model_path,
            )
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch:2d}/{args.epochs} "
            f"| Train Loss: {avg_train_loss:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Acc: {val_acc * 100:.2f}% "
            f"| Val mIoU: {val_miou * 100:.2f}% "
            f"| LR: {current_lr:.2e}{improved_tag}"
        )

        if args.early_stop > 0 and epochs_no_improve >= args.early_stop:
            print(
                f"\n[Early Stopping] Tidak ada peningkatan Val mIoU selama {args.early_stop} epoch berturut-turut."
            )
            break

    elapsed_min = (time.time() - t_start) / 60.0

    # Simpan riwayat loss & kurva pelatihan
    history_payload = {
        "best_epoch": best_epoch,
        "best_val_miou": round(best_val_miou, 6),
        "best_val_acc": round(best_val_acc, 6),
        "training_minutes": round(elapsed_min, 2),
        "hyperparams": {
            "epochs_configured": args.epochs,
            "epochs_run": len(history["train_loss"]),
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "scheduler": args.scheduler,
            "use_class_weights": args.use_class_weights,
            "seed": args.seed,
        },
        "history": history,
    }
    history_path = args.weights_dir / "loss_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2)

    curves_path = args.weights_dir / "training_curves.png"
    save_training_curves(history, curves_path)

    # Evaluasi akhir model terbaik pada Test Set (20% data uji CamVid yang belum pernah dilihat)
    best_ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    test_metrics = evaluate_dataset(model, test_loader, criterion, device)

    test_payload = {
        "best_epoch": best_epoch,
        "test_count": len(test_pairs),
        "test_loss": round(float(test_metrics["loss"]), 6),
        "global_pixel_accuracy": round(float(test_metrics["pixel_acc"]), 6),
        "mean_iou": round(float(test_metrics["mean_iou"]), 6),
        "per_class_iou": test_metrics["per_class_iou"],
        "confusion_matrix": test_metrics["confusion_matrix"].tolist(),
        "class_names": CLASS_NAMES,
    }
    test_eval_path = args.weights_dir / "test_evaluation.json"
    with test_eval_path.open("w", encoding="utf-8") as f:
        json.dump(test_payload, f, indent=2)

    save_test_plots(
        test_metrics["per_class_iou"],
        test_metrics["confusion_matrix"],
        args.weights_dir,
    )

    print("\nRingkasan Hasil Training & Evaluasi Test Set")
    print("-" * 60)
    print(f"Waktu training       : {elapsed_min:.2f} menit")
    print(f"Best Epoch (Val)     : {best_epoch}")
    print(f"Best Val mIoU        : {best_val_miou * 100:.2f}%")
    print(f"Best Val Pixel Acc   : {best_val_acc * 100:.2f}%")
    print(f"Test Global Accuracy : {test_metrics['pixel_acc'] * 100:.2f}%")
    print(f"Test Mean IoU (mIoU) : {test_metrics['mean_iou'] * 100:.2f}%")
    print("\nIoU per Kelas (Test Set):")
    for cls_name in CLASS_NAMES:
        iou_val = test_metrics["per_class_iou"][cls_name]
        iou_str = f"{iou_val * 100:6.2f}%" if iou_val is not None else "   N/A "
        print(f"  - {cls_name:<12}: {iou_str}")
    print("\nArtefak Tersimpan:")
    print(f"  - Bobot terbaik  : {best_model_path}")
    print(f"  - Riwayat loss   : {history_path}")
    print(f"  - Kurva training : {curves_path}")
    print(f"  - Evaluasi test  : {test_eval_path}")
    print(f"  - Plot IoU/CM    : {args.weights_dir / 'per_class_iou.png'}, {args.weights_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
