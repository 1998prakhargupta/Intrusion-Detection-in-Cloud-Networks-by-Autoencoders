# Production-Ready NIDS Autoencoder System
# Use Python 3.9 slim image for smaller footprint
FROM python:3.9-slim

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r nids && useradd -r -g nids nids

# Copy requirements first for better caching
COPY pyproject.toml ./
COPY requirements.txt* ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e .

# Copy source code
COPY src/ src/
COPY config/ config/
COPY data/ data/
COPY models/ models/

# Create necessary directories and set permissions
RUN mkdir -p logs data/processed models/artifacts && \
    chown -R nids:nids /app

# Switch to non-root user
USER nids

# Expose port for FastAPI
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden)
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
