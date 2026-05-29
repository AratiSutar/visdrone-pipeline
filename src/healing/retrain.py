import shutil
import pandas as pd
from pathlib import Path
from ultralytics import YOLO


def check_drift_detected(
    drift_results_path: str = "logs/drift_results.csv",
    threshold: float = 0.5,
) -> bool:
    """Check if drift was detected in any drift type."""
    path = Path(drift_results_path)
    if not path.exists():
        print("[heal] No drift results found. Run monitor first.")
        return False

    df = pd.read_csv(path)
    detected = df["detected"].any()
    max_score = df["drift_score"].max()

    print(f"[heal] Max drift score: {max_score:.4f}")
    print(f"[heal] Drift detected: {detected}")

    # Also trigger if drift score is high even if below threshold
    if max_score > 0.01:
        print(f"[heal] Drift score {max_score:.4f} > 0.01 — triggering retraining")
        return True

    return detected


def prepare_healing_dataset(
    processed_root: str = "data/processed",
    drifted_root: str = "data/drifted",
    output_root: str = "data/healing",
    drift_types: list = ["fog", "night", "blur", "rain"],
    samples_per_drift: int = 100,
):
    """
    Prepare healing dataset by mixing:
    - Original clean images
    - Drifted images
    """
    output_root = Path(output_root)
    img_dst = output_root / "train" / "images"
    lbl_dst = output_root / "train" / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    count = 0

    # Add original clean images
    clean_img_dir = Path(processed_root) / "train" / "images"
    clean_lbl_dir = Path(processed_root) / "train" / "labels"
    clean_images = sorted(clean_img_dir.glob("*.jpg"))[:200]

    for img_path in clean_images:
        lbl_path = clean_lbl_dir / (img_path.stem + ".txt")
        shutil.copy2(img_path, img_dst / img_path.name)
        if lbl_path.exists():
            shutil.copy2(lbl_path, lbl_dst / lbl_path.name)
        count += 1

    print(f"[heal] Added {len(clean_images)} clean images")

    # Add drifted images
    for drift_type in drift_types:
        drift_img_dir = Path(drifted_root) / drift_type / "val" / "images"
        drift_lbl_dir = Path(drifted_root) / drift_type / "val" / "labels"

        if not drift_img_dir.exists():
            print(f"[warn] {drift_img_dir} not found, skipping")
            continue

        drift_images = sorted(drift_img_dir.glob("*.jpg"))[:samples_per_drift]

        for img_path in drift_images:
            lbl_path = drift_lbl_dir / (img_path.stem + ".txt")
            dst_name = f"{drift_type}_{img_path.name}"
            shutil.copy2(img_path, img_dst / dst_name)
            if lbl_path.exists():
                shutil.copy2(lbl_path, lbl_dst / (img_path.stem + ".txt"))
            count += 1

        print(f"[heal] Added {len(drift_images)} {drift_type} images")

    print(f"[heal] Total healing dataset: {count} images")
    return count


def run_healing(
    baseline_model: str = "models/baseline/run/weights/best.pt",
    healing_data: str = "data/healing",
    val_data: str = "data/processed/val",
    output_dir: str = "models/healed",
    epochs: int = 10,
):
    """Fine-tune model on healing dataset."""
    import yaml

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create healing dataset yaml
    dataset_cfg = {
        "path": str(Path(healing_data).resolve()),
        "train": "train/images",
        "val": str((Path(val_data) / "images").resolve()),
        "nc": 10,
        "names": {
            0: "pedestrian",
            1: "people",
            2: "bicycle",
            3: "car",
            4: "van",
            5: "truck",
            6: "tricycle",
            7: "awning-tricycle",
            8: "bus",
            9: "motor",
        },
    }

    yaml_path = Path(output_dir) / "healing_dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_cfg, f)

    # Load baseline model and fine-tune
    print(f"[heal] Loading baseline model from {baseline_model}")
    model = YOLO(baseline_model)

    print(f"[heal] Fine-tuning for {epochs} epochs...")
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=8,
        imgsz=640,
        workers=2,
        lr0=0.001,
        project=output_dir,
        name="run",
        save=True,
        exist_ok=True,
    )

    print(f"\n[heal] Healed model saved to {output_dir}/run/weights/best.pt")
    return results


def self_heal(
    drift_results_path: str = "logs/drift_results.csv",
    baseline_model: str = "models/baseline/run/weights/best.pt",
    output_dir: str = "models/healed",
    epochs: int = 10,
):
    """Full self-healing pipeline."""
    print("\n=== Self-Healing Pipeline ===")

    # Step 1 - Check drift
    print("\n[Step 1] Checking drift...")
    drift_detected = check_drift_detected(drift_results_path)

    if not drift_detected:
        print("[heal] No significant drift detected. No retraining needed.")
        return

    # Step 2 - Prepare healing dataset
    print("\n[Step 2] Preparing healing dataset...")
    prepare_healing_dataset()

    # Step 3 - Retrain
    print("\n[Step 3] Retraining model...")
    run_healing(
        baseline_model=baseline_model,
        output_dir=output_dir,
        epochs=epochs,
    )

    print("\n=== Self-Healing Complete ===")
    print(f"Healed model saved to {output_dir}/run/weights/best.pt")


if __name__ == "__main__":
    self_heal()