# =================================================================
# Dockerfile for MLOps Pipeline Execution
# Purpose: Run the complete training pipeline in containerized environment
# Author: Carlos Daniel Jiménez
# =================================================================

FROM python:3.12-slim

LABEL maintainer="danieljimenez88m@gmail.com"
LABEL description="Housing Price Prediction - MLOps Pipeline"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml ./
COPY requirements.txt* ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir uv && \
    uv pip install --system -e .

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create necessary directories
RUN mkdir -p mlruns outputs models

# Default command runs the pipeline
CMD ["python", "main.py"]
