# Self-Healing Aerial Object Detection Pipeline

A production-style computer vision pipeline built on VisDrone DET dataset and YOLOv8,
featuring automatic drift detection, active learning, and self-healing retraining.

## Overview

Most object detection projects only build a baseline model.
This pipeline goes further with 6 complete systems:

| System | Description |
|--------|-------------|
| 1. Dataset Processing | VisDrone annotations → YOLO format |
| 2. Baseline Detection | YOLOv8 training and evaluation |
| 3. Drift Simulation | Fog, night, blur, rain augmentations |
| 4. Drift Monitoring | KL divergence confidence scoring |
| 5. Active Learning | Uncertainty sampling for relabeling |
| 6. Self-Healing | Automatic retraining on drifted data |

## Results

| Metric | Baseline |
|--------|----------|
| mAP50 | 0.307 |
| mAP50-95 | 0.172 |
| Best class | car (0.735) |
| Worst class | bicycle (0.062) |

## Project Structure
visdrone-pipeline/
├── src/
│   ├── dataset/        # System 1 - data processing
│   ├── detection/      # System 2 - YOLOv8 training
│   ├── drift/          # Systems 3 & 4 - drift simulation and monitoring
│   ├── active_learning/# System 5 - uncertainty sampling
│   ├── healing/        # System 6 - self-healing retraining
│   └── api/            # FastAPI inference server
├── configs/            # Dataset and training configs
├── tests/              # Unit tests
├── Dockerfile
├── docker-compose.yml
└── requirements.txt

## Setup

### Local

```bash
git clone https://github.com/AratiSutar/visdrone-pipeline.git
cd visdrone-pipeline
python -m venv venv
source Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Docker

```bash
docker build -t visdrone-pipeline:latest .
docker-compose run --rm -p 8000:8000 pipeline uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

## Usage

### Run dataset conversion

```bash
python src/dataset/converter.py
```

### Train baseline model

```bash
python src/detection/train.py
```

### Evaluate model

```bash
python src/detection/evaluate.py
```

### Simulate drift

```bash
python src/drift/simulator.py
```

### Monitor drift

```bash
python src/drift/monitor.py
```

### Run active learning

```bash
python src/active_learning/sampler.py
```

### Run self-healing

```bash
python src/healing/retrain.py
```

### Start API server

```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

### API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | API status |
| GET | /health | Health check |
| POST | /detect | Upload image and get detections |
| GET | /classes | List all classes |

### Test API

```bash
curl -X POST "http://localhost:8000/detect" -F "file=@your_image.jpg"
```

## Dataset

VisDrone DET 2019 — 10 classes:
pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor

## Tech Stack

- YOLOv8 (Ultralytics)
- FastAPI
- Docker
- GitHub Actions
- OpenCV
- PyTorch