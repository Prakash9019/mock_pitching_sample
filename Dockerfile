# syntax=docker/dockerfile:1

# -------------------------------
# Stage 1: Build dependencies
# -------------------------------
    FROM python:3.11-slim as builder
    WORKDIR /app
    
    # Install system build tools and audio/video libs
    RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ python3-dev build-essential cmake pkg-config git \
        libsndfile1-dev libffi-dev \
        libblas-dev liblapack-dev \
        libavcodec-dev libavformat-dev libavutil-dev libswresample-dev libavfilter-dev \
        llvm-dev libgl1-mesa-glx \
        && rm -rf /var/lib/apt/lists/*
    
    # Copy and install Python requirements
    COPY requirements.txt .
    RUN pip install --upgrade pip && \
        pip install --user -r requirements.txt && \
        pip install --user uvicorn
    
    # -------------------------------
    # Stage 2: Runtime environment
    # -------------------------------
    FROM python:3.11-slim
    
    # Set environment vars
    ENV PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1 \
        PYTHONPATH=/app \
        PORT=8080 \
        PATH=/root/.local/bin:$PATH \
        DEBIAN_FRONTEND=noninteractive
    
    # Install only runtime audio dependencies
    RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 libasound2 libgl1-mesa-glx \
        && apt-get clean && rm -rf /var/lib/apt/lists/*
    
    # Working directory
    WORKDIR /app
    
    # Copy dependencies from builder
    COPY --from=builder /root/.local /root/.local
    
    # Copy app code
    COPY . .
    
    # Create expected directories for uploads/sessions
    RUN mkdir -p /app/data/uploads /app/data/responses /app/data/sessions && \
        chmod -R a+rwx /app/data
    
    # Healthcheck endpoint
    HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
        CMD curl --fail http://localhost:8080/health || exit 1
    
    # Expose port for Cloud Run
    EXPOSE 8080
    
    # Start the app using Uvicorn (ASGI for FastAPI + Socket.IO)
    CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
    