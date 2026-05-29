import numpy as np
import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO


def score_image_uncertainty(model: YOLO, img_path: str) -> float:
    """
    Score image uncertainty using margin sampling.
    Lower confidence = higher uncertainty.
    """
    results = model(img_path, verbose=False)
    confidences = []

    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            confs = r.boxes.conf.cpu().numpy().tolist()
            confidences.extend(confs)

    if not confidences:
        return 1.0  # no detections = maximum uncertainty

    # Uncertainty = 1 - mean confidence
    return 1.0 - float(np.mean(confidences))


def select_uncertain_samples(
    model_path: str = "models/baseline/run/weights/best.pt",
    img_dir: str = "data/drifted/fog/val/images",
    output_dir: str = "data/active_learning/selected",
    n_samples: int = 50,
    uncertainty_threshold: float = 0.6,
):
    """Select most uncertain images for relabeling."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    print("[sampler] Loading model...")
    model = YOLO(model_path)

    # Score all images
    img_dir = Path(img_dir)
    images = sorted(img_dir.glob("*.jpg"))
    print(f"[sampler] Scoring {len(images)} images...")

    scores = []
    for img_path in tqdm(images, desc="Scoring uncertainty"):
        uncertainty = score_image_uncertainty(model, str(img_path))
        scores.append({
            "image": img_path.name,
            "path":  str(img_path),
            "uncertainty": round(uncertainty, 4),
        })

    # Sort by uncertainty descending
    df = pd.DataFrame(scores)
    df = df.sort_values("uncertainty", ascending=False).reset_index(drop=True)

    # Select top N samples above threshold
    selected = df[df["uncertainty"] >= uncertainty_threshold].head(n_samples)

    if len(selected) == 0:
        print(f"[warn] No images above threshold {uncertainty_threshold}")
        print(f"[info] Taking top {n_samples} most uncertain images instead")
        selected = df.head(n_samples)

    print(f"\n[sampler] Selected {len(selected)} images for relabeling")
    print(f"[sampler] Mean uncertainty: {selected['uncertainty'].mean():.4f}")
    print(f"[sampler] Max uncertainty:  {selected['uncertainty'].max():.4f}")
    print(f"[sampler] Min uncertainty:  {selected['uncertainty'].min():.4f}")

    # Copy selected images to output dir
    for _, row in selected.iterrows():
        shutil.copy2(row["path"], Path(output_dir) / row["image"])

    # Save scores
    scores_path = Path(output_dir) / "uncertainty_scores.csv"
    df.to_csv(scores_path, index=False)
    print(f"\n[saved] {scores_path}")

    return selected


if __name__ == "__main__":
    select_uncertain_samples()