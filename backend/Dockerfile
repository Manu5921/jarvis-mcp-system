# Dockerfile pour Jarvis MCP Backend
FROM python:3.11-slim

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Installer dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Créer utilisateur non-root
RUN useradd --create-home --shell /bin/bash jarvis

# Définir répertoire de travail
WORKDIR /app

# Copier requirements et installer dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier code source
COPY . .

# Créer dossiers nécessaires
RUN mkdir -p logs alembic/versions && \
    chown -R jarvis:jarvis /app

# Passer à l'utilisateur non-root
USER jarvis

# Exposer port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Script de démarrage
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]