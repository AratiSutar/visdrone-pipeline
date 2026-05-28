from pathlib import Path
from ultralytics import YOLO
import yaml


def train(
    config_path: str = "configs/train.yaml",
    dataset_path: str = "configs/dataset.yaml",
    output_dir: str = "models/baseline",
):
    """Train YOLOv8 on VisDrone dataset."""

    # Load train config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Create output dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    model = YOLO(cfg.get("model", "yolov8n.pt"))

    # Train
    results = model.train(
        data=dataset_path,
        epochs=cfg.get("epochs", 50),
        batch=cfg.get("batch", 8),
        imgsz=cfg.get("imgsz", 640),
        workers=cfg.get("workers", 2),
        optimizer=cfg.get("optimizer", "SGD"),
        lr0=cfg.get("lr0", 0.01),
        momentum=cfg.get("momentum", 0.937),
        weight_decay=cfg.get("weight_decay", 0.0005),
        fraction=cfg.get("fraction", 0.1),
        project=output_dir,
        name="run",
        save=True,
        save_period=10,
        exist_ok=True,
    )

    print(f"\nTraining complete. Model saved to {output_dir}/run/weights/best.pt")
    return results


if __name__ == "__main__":
    train()