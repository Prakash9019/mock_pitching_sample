# syntax=docker/dockerfile:1

# Build arguments for optimization levels
ARG BUILD_MODE=full
ARG PYTHON_VERSION=3.11

# -------------------------------
# Stage 1: Base system setup
# -------------------------------
FROM python:${PYTHON_VERSION}-slim as base

# Build and runtime arguments
ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_MODE=full

# Environment variables for optimization
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies based on build mode
RUN apt-get update && \
    if [ "$BUILD_MODE" = "lite" ]; then \
        apt-get install -y --no-install-recommends \
            gcc python3-dev build-essential \
            libffi-dev curl; \
    else \
        apt-get install -y --no-install-recommends \
            gcc g++ python3-dev build-essential cmake pkg-config git \
            libsndfile1-dev libffi-dev \
            libblas-dev liblapack-dev \
            libavcodec-dev libavformat-dev libavutil-dev libswresample-dev \
            curl; \
    fi && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# -------------------------------
# Stage 2: Python dependencies
# -------------------------------
FROM base as builder

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python packages based on build mode
RUN pip install --upgrade pip setuptools wheel && \
    if [ "$BUILD_MODE" = "lite" ]; then \
        # Lite mode: Core functionality without heavy ML models
        pip install --user --no-cache-dir --compile \
            # Web framework
            fastapi==0.115.12 \
            uvicorn==0.34.3 \
            python-socketio==5.13.0 \
            python-engineio==4.12.2 \
            python-multipart==0.0.20 \
            starlette==0.46.2 \
            # Database
            pymongo==4.13.2 \
            motor==3.7.1 \
            dnspython==2.7.0 \
            SQLAlchemy==2.0.41 \
            # HTTP clients
            aiohttp==3.12.12 \
            httpx==0.28.1 \
            requests==2.32.4 \
            # Core utilities
            pydantic==2.11.6 \
            click==8.2.1 \
            PyYAML==6.0.2 \
            orjson==3.10.18 \
            # Google services (essential for your app)
            google-cloud-texttospeech==2.27.0 \
            google-generativeai==0.8.5 \
            google-api-core==2.25.1 \
            google-auth==2.40.3 \
            # LangChain (core functionality)
            langchain==0.3.15 \
            langchain-core==0.3.65 \
            langchain-google-genai==2.0.10 \
            langchain-text-splitters==0.3.8 \
            langsmith==0.3.45 \
            langgraph==0.2.65 \
            # Basic audio processing (lightweight)
            pydub==0.25.1 \
            soundfile==0.12.1; \
    else \
        # Full mode: Install all packages including heavy ML models
        pip install --user --no-cache-dir --compile -r requirements.txt; \
    fi && \
    # Clean up
    find /root/.local -name "*.pyc" -delete && \
    find /root/.local -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# -------------------------------
# Stage 3: Final runtime image
# -------------------------------
FROM python:${PYTHON_VERSION}-slim as runtime

# Build argument
ARG BUILD_MODE=full

# Runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8080 \
    PATH=/root/.local/bin:$PATH \
    DEBIAN_FRONTEND=noninteractive

# Install only essential runtime dependencies
RUN apt-get update && \
    if [ "$BUILD_MODE" = "lite" ]; then \
        apt-get install -y --no-install-recommends curl; \
    else \
        apt-get install -y --no-install-recommends \
            ffmpeg libsndfile1 libasound2 curl; \
    fi && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /root/.local /root/.local

# Copy application code (use .dockerignore to exclude unnecessary files)
COPY --chown=appuser:appuser . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/data/uploads /app/data/responses /app/data/sessions && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app/data

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8080/health || exit 1

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
    