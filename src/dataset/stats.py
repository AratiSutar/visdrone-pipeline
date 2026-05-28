import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from tqdm import tqdm

CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]


def compute_split_stats(label_dir: Path):
    """Count classes and images for a split."""
    class_counts  = Counter()
    empty_images  = 0
    total_images  = 0

    for lbl_file in tqdm(sorted(label_dir.glob("*.txt")), desc=str(label_dir)):
        total_images += 1
        lines = lbl_file.read_text().strip().split("\n")
        lines = [line for line in lines if line.strip()]

        if not lines:
            empty_images += 1
            continue

        for line in lines:
            parts = line.split()
            cls   = int(parts[0])
            class_counts[cls] += 1

    return {
        "total_images": total_images,
        "empty_images": empty_images,
        "class_counts": class_counts,
    }


def plot_class_distribution(stats: dict, save_path: str = "logs/class_distribution.png"):
    """Bar chart of class counts per split."""
    Path(save_path).parent.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, len(stats), figsize=(14, 5))
    if len(stats) == 1:
        axes = [axes]

    for ax, (split, s) in zip(axes, stats.items()):
        counts = [s["class_counts"].get(i, 0) for i in range(len(CLASS_NAMES))]
        ax.barh(CLASS_NAMES, counts, color="steelblue")
        ax.set_title(f"{split} ({s['total_images']} images)")
        ax.set_xlabel("Instance count")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[saved] {save_path}")


def run_stats(processed_root: str = "data/processed"):
    """Run stats on all splits."""
    root      = Path(processed_root)
    all_stats = {}

    for split in ["train", "val", "test"]:
        lbl_dir = root / split / "labels"
        if lbl_dir.exists():
            all_stats[split] = compute_split_stats(lbl_dir)

    # Print summary
    rows = []
    for split, s in all_stats.items():
        row = {
            "split":  split,
            "images": s["total_images"],
            "empty":  s["empty_images"]
        }
        row.update({CLASS_NAMES[i]: s["class_counts"].get(i, 0) for i in range(10)})
        rows.append(row)

    df = pd.DataFrame(rows).set_index("split")
    print("\n=== Dataset statistics ===")
    print(df.to_string())

    plot_class_distribution(all_stats)
    return all_stats


if __name__ == "__main__":
    run_stats()