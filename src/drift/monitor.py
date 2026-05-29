import cv2
import numpy as np
import json
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

    for img_path