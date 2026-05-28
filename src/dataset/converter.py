import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# VisDrone class mapping
VALID_CLASSES = {k: k - 1 for k in range(1, 11)}

CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]


def convert_annotation(ann_path: Path, img_w: int, img_h: int):
    """Convert one VisDrone annotation file to YOLO format."""
    yolo_lines = []

    with open(ann_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(",")
            x      = int(parts[0])
            y      = int(parts[1])
            w      = int(parts[2])
            h      = int(parts[3])
            score  = int(parts[4])
            category = int(parts[5])

            # Skip ignored regions and zero size boxes
            if score == 0 or category not in VALID_CLASSES:
                continue
            if w == 0 or h == 0:
                continue

            # Convert to YOLO normalized format
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h

            # Clamp to [0, 1]
            cx = np.clip(cx, 0, 1)
            cy = np.clip(cy, 0, 1)
            nw = np.clip(nw, 0, 1)
            nh = np.clip(nh, 0, 1)

            cls_id = VALID_CLASSES[category]
            yolo_lines.append(
                f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
            )

    return yolo_lines


def convert_split(raw_split_dir: Path, out_dir: Path, split_name: str):
    """Convert a full split (train/val/test)."""
    img_src = raw_split_dir / "images"
    ann_src = raw_split_dir / "annotations"

    img_dst = out_dir / split_name / "images"
    lbl_dst = out_dir / split_name / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    images   = sorted(img_src.glob("*.jpg"))
    skipped  = 0
    converted = 0

    for img_path in tqdm(images, desc=f"Converting {split_name}"):
        ann_path = ann_src / (img_path.stem + ".txt")

        if not ann_path.exists():
            skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        h, w = img.shape[:2]
        yolo_lines = convert_annotation(ann_path, w, h)

        # Copy image
        shutil.copy2(img_path, img_dst / img_path.name)

        # Write label file
        lbl_file = lbl_dst / (img_path.stem + ".txt")
        with open(lbl_file, "w") as f:
            f.write("\n".join(yolo_lines))

        converted += 1

    print(f"[{split_name}] converted={converted} skipped={skipped}")
    return converted


def convert_all(raw_root: str = "data/raw", out_root: str = "data/processed"):
    """Convert all splits."""
    raw_root = Path(raw_root)
    out_root = Path(out_root)

    splits = {
        "train": raw_root / "VisDrone2019-DET-train",
        "val":   raw_root / "VisDrone2019-DET-val",
        "test":  raw_root / "VisDrone2019-DET-test-dev",
    }

    for split_name, split_dir in splits.items():
        if not split_dir.exists():
            print(f"[warn] {split_dir} not found, skipping")
            continue
        convert_split(split_dir, out_root, split_name)

    print("\nConversion complete.")


if __name__ == "__main__":
    convert_all()