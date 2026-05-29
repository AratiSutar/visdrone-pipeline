from pathlib import Path
from ultralytics import YOLO
import pandas as pd


CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]


def evaluate(
    model_path: str = "models/baseline/run/weights/best.pt",
    dataset_path: str = "configs/dataset.yaml",
    output_dir: str = "logs",
):
    """Evaluate YOLOv8 model on validation set."""

    # Check model exists
    if not Path(model_path).exists():
        print(f"[error] Model not found at {model_path}")
        return

    # Create output dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    model = YOLO(model_path)

    # Run validation
    metrics = model.val(
        data=dataset_path,
        split="val",
        project=output_dir,
        name="eval",
        exist_ok=True,
    )

    # Print results
    print("\n=== Evaluation Results ===")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")

    # Per class metrics
    rows = []
    for i, name in enumerate(CLASS_NAMES):
        rows.append({
            "class": name,
            "ap50":  round(metrics.box.ap50[i], 4) if i < len(metrics.box.ap50) else 0,
        })

    df = pd.DataFrame(rows)
    print("\n=== Per Class AP50 ===")
    print(df.to_string(index=False))

    # Save to csv
    csv_path = Path(output_dir) / "eval_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[saved] {csv_path}")

    return metrics


if __name__ == "__main__":
    evaluate()