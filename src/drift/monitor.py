import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt


def get_confidences(
    model: YOLO,
    img_dir: Path,
    max_images: int = 100,
) -> list:
    """Run model on images and collect confidence scores."""
    confidences = []
    images = sorted(img_dir.glob("*.jpg"))[:max_images]

    for img_path in tqdm(images, desc=f"Scoring {img_dir.name}"):
        results = model(str(img_path), verbose=False)
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                confs = r.boxes.conf.cpu().numpy().tolist()
                confidences.extend(confs)

    return confidences


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL divergence between two distributions."""
    p = p + 1e-10
    q = q + 1e-10
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def compute_drift_score(
    baseline_confs: list,
    drifted_confs: list,
    n_bins: int = 20,
) -> float:
    """Compute drift score using KL divergence."""
    bins = np.linspace(0, 1, n_bins + 1)
    p, _ = np.histogram(baseline_confs, bins=bins)
    q, _ = np.histogram(drifted_confs, bins=bins)
    return kl_divergence(p.astype(float), q.astype(float))


def plot_confidence_distributions(
    baseline_confs: list,
    drifted_confs: list,
    drift_type: str,
    save_path: str,
):
    """Plot confidence histograms for baseline vs drifted."""
    plt.figure(figsize=(10, 5))
    plt.hist(baseline_confs, bins=20, alpha=0.6, label="Baseline", color="steelblue")
    plt.hist(drifted_confs, bins=20, alpha=0.6, label=f"Drifted ({drift_type})", color="coral")
    plt.xlabel("Confidence score")
    plt.ylabel("Count")
    plt.title(f"Confidence distribution — {drift_type}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[saved] {save_path}")


def monitor_drift(
    model_path: str = "models/baseline/run/weights/best.pt",
    baseline_dir: str = "data/processed/val/images",
    drifted_root: str = "data/drifted",
    output_dir: str = "logs",
    drift_threshold: float = 0.5,
    max_images: int = 100,
):
    """Monitor drift across all drift types."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    print("[monitor] Loading model...")
    model = YOLO(model_path)

    # Get baseline confidences
    print("[monitor] Scoring baseline images...")
    baseline_confs = get_confidences(model, Path(baseline_dir), max_images)
    print(f"[monitor] Baseline mean confidence: {np.mean(baseline_confs):.4f}")

    # Check each drift type
    results = []
    drift_types = ["fog", "night", "blur", "rain"]

    for drift_type in drift_types:
        drifted_dir = Path(drifted_root) / drift_type / "val" / "images"
        if not drifted_dir.exists():
            print(f"[warn] {drifted_dir} not found, skipping")
            continue

        print(f"\n[monitor] Scoring {drift_type} images...")
        drifted_confs = get_confidences(model, drifted_dir, max_images)

        drift_score = compute_drift_score(baseline_confs, drifted_confs)
        mean_conf = np.mean(drifted_confs) if drifted_confs else 0
        detected = drift_score > drift_threshold

        print(f"[{drift_type}] drift_score={drift_score:.4f} mean_conf={mean_conf:.4f} detected={detected}")

        # Plot
        plot_path = str(Path(output_dir) / f"drift_{drift_type}.png")
        plot_confidence_distributions(baseline_confs, drifted_confs, drift_type, plot_path)

        results.append({
            "drift_type":  drift_type,
            "drift_score": round(drift_score, 4),
            "mean_conf":   round(mean_conf, 4),
            "detected":    detected,
        })

    # Save results
    df = pd.DataFrame(results)
    print("\n=== Drift Monitor Results ===")
    print(df.to_string(index=False))

    csv_path = Path(output_dir) / "drift_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[saved] {csv_path}")

    return df


if __name__ == "__main__":
    monitor_drift()