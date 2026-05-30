import cv2
import numpy as np
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO


CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]

app = FastAPI(
    title="VisDrone Detection API",
    description="Self-healing aerial object detection pipeline",
    version="1.0.0",
)

# Load model once at startup
MODEL_PATH = "models/baseline/run/weights/best.pt"
model = None


@app.on_event("startup")
async def load_model():
    global model
    if Path(MODEL_PATH).exists():
        model = YOLO(MODEL_PATH)
        print(f"[api] Model loaded from {MODEL_PATH}")
    else:
        print(f"[api] Warning: model not found at {MODEL_PATH}")


@app.get("/")
async def root():
    return {
        "message": "VisDrone Detection API",
        "status": "running",
        "model": MODEL_PATH,
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Upload an image and get object detections.
    Returns list of detections with class, confidence, and bounding box.
    """
    global model

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Run detection
    results = model(img, verbose=False)

    # Parse detections
    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls.item())
            conf   = round(float(box.conf.item()), 4)
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class_id":   cls_id,
                "class_name": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown",
                "confidence": conf,
                "bbox": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                }
            })

    return JSONResponse({
        "filename":   file.filename,
        "image_size": {"width": img.shape[1], "height": img.shape[0]},
        "total_detections": len(detections),
        "detections": detections,
    })


@app.get("/classes")
async def get_classes():
    """Get list of all detectable classes."""
    return {
        "total": len(CLASS_NAMES),
        "classes": {i: name for i, name in enumerate(CLASS_NAMES)}
    }