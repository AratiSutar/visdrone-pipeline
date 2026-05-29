import cv2
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm


def add_fog(img: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """Add fog effect to image."""
    fog = np.ones_like(img, dtype=np.uint8) * 255
    return cv2.addWeighted(img, 1 - intensity, fog, intensity, 0)


def add_night(img: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """Darken image to simulate night."""
    return cv2.convertScaleAbs(img, alpha=1 - intensity, beta=0)


def add_blur(img: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Add motion blur to image."""
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    return cv2.filter2D(img, -1, kernel)


def add_rain(img: np.ndarray, intensity: int = 500) -> np.ndarray:
    """Add rain streaks to image."""
    rain = img.copy()
    h, w = rain.shape[:2]
    for _ in range(intensity):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        length = np.random.randint(10, 30)
        angle = np.random.randint(-20, 20)
        x2 = int(x + length * np.sin(np.radians(angle)))
        y2 = int(y + length * np.cos(np.radians(angle)))
        cv2.line(rain, (x, y), (x2, y2), (200, 200, 200), 1)
    return rain


DRIFT_TYPES = {
    "fog":   lambda img: add_fog(img, intensity=0.5),
    "night": lambda img: add_night(img, intensity=0.6),
    "blur":  lambda img: add_blur(img, kernel_size=15),
    "rain":  lambda img: add_rain(img, intensity=500),
}


def simulate_drift(
    processed_root: str = "data/processed",
    output_root: str = "data/drifted",
    split: str = "val",
    drift_type: str = "fog",
):
    """Apply drift augmentation to a split."""

    if drift_type not in DRIFT_TYPES:
        print(f"[error] Unknown drift type: {drift_type}")
        print(f"Available: {list(DRIFT_TYPES.keys())}")
        return

    img_src = Path(processed_root) / split / "images"
    lbl_src = Path(processed_root) / split / "labels"

    img_dst = Path(output_root) / drift_type / split / "images"
    lbl_dst = Path(output_root) / drift_type / split / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    images = sorted(img_src.glob("*.jpg"))
    print(f"[drift] Applying {drift_type} to {len(images)} images...")

    for img_path in tqdm(images, desc=drift_type):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Apply drift
        drifted = DRIFT_TYPES[drift_type](img)

        # Save drifted image
        cv2.imwrite(str(img_dst / img_path.name), drifted)

        # Copy labels unchanged
        lbl_path = lbl_src / (img_path.stem + ".txt")
        if lbl_path.exists():
            shutil.copy2(lbl_path, lbl_dst / lbl_path.name)

    print(f"[done] {drift_type} drift saved to {img_dst}")


def simulate_all_drifts(
    processed_root: str = "data/processed",
    output_root: str = "data/drifted",
    split: str = "val",
):
    """Apply all drift types."""
    for drift_type in DRIFT_TYPES:
        simulate_drift(processed_root, output_root, split, drift_type)

    print("\nAll drifts complete.")


if __name__ == "__main__":
    simulate_all_drifts()