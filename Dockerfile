FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/       ./src/
COPY configs/   ./configs/

# Mount points for data and models
VOLUME ["/app/data", "/app/models", "/app/logs"]

ENV PYTHONUNBUFFERED=1