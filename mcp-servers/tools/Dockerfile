FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    tar \
    gzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY server.py .

# Create workspace and data directories
RUN mkdir -p /app/workspace /app/data && \
    chmod 755 /app/workspace /app/data

# Expose port
EXPOSE 4006

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "from pathlib import Path; Path('/app/workspace').exists() and print('Tools MCP healthy')" || exit 1

# Run the MCP server
CMD ["python", "server.py"]