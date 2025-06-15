# syntax=docker/dockerfile:1

# Stage 1: Build dependencies
FROM python:3.11-slim as builder
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    build-essential \
    cmake \
    pkg-config \
    git \
    # For audio processing libraries
    libsndfile1-dev \
    libffi-dev \
    # For some Python packages that need system libraries
    libblas-dev \
    liblapack-dev \
    # For librosa
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    libavfilter-dev \
    # For numba
    llvm-dev \
    # For opencv-python
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --user -r requirements.txt && \
    pip install --user gunicorn uvloop httptools

# Stage 2: Runtime image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PYTHONPATH=/app \
    PORT=8080 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Audio/video processing
    ffmpeg \
    libsndfile1 \
    libavcodec60 \
    libavformat60 \
    libavutil58 \
    libswresample4 \
    libavfilter8 \
    # For librosa
    libblas3 \
    liblapack3 \
    # For opencv-python
    libgl1-mesa-glx \
    # For webrtcvad
    libasound2 \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Ensure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/uploads /app/data/responses /app/data/sessions /app/logs

# Set proper permissions
RUN chmod -R a+rwx /app/data

# Expose the port the app runs on
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8080/health || exit 1

# Command to run the application with Gunicorn
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "120"]