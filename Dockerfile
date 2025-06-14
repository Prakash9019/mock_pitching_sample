# Build stage
FROM python:3.10-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir=/app/wheels -r requirements.txt

# Runtime stage
FROM python:3.10-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /app/wheels /wheels

# Install Python dependencies
RUN pip install --no-cache-dir /wheels/* \
    && rm -rf /wheels

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/uploads /app/data/responses /app/data/sessions

# Set environment variables
ENV PYTHONPATH=/app \
    PORT=8080 \
    HOST=0.0.0.0 \
    RELOAD=false \
    LOG_LEVEL=info 

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD exec uvicorn app.main:fastapi_app \
    --host $HOST \
    --port $PORT \
    --reload $RELOAD \
    --log-level $LOG_LEVEL \
    --workers 4
