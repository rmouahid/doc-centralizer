# Utilise une image Python officielle comme base
FROM python:3.12-slim

# Définit le répertoire de travail dans le conteneur
WORKDIR /app

# Installe tesseract-ocr (requis par pytesseract pour l'extraction de texte depuis les images)
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copie les fichiers de dépendances Poetry
COPY pyproject.toml poetry.lock ./

# Installe Poetry
RUN pip install --no-cache-dir poetry

# Installe les dépendances du projet
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi

# Copie le reste du code source dans le conteneur
COPY . .

EXPOSE 8501

# Définit la commande de démarrage
CMD ["poetry", "run", "streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.headless=true"]

