from pathlib import Path
from ultralytics import YOLO
import pandas as pd
import yaml


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