# AIQuant v5.0 - Production Docker Image
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libta-lib0-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    fastapi uvicorn \
    dash dash-bootstrap-components dash-ag-grid

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/
COPY docs/ ./docs/
COPY scripts/ ./scripts/
COPY app.py start_all.sh stop_all.sh ./

# Create logs directory
RUN mkdir -p logs

# Environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8000 8050

# Default command: start both API and Dashboard
CMD ["bash", "start_all.sh"]
