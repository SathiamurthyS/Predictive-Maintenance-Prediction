
FROM python:3.10-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# HF Spaces requires port 7860
ENV PORT=7860

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Production WSGI server
CMD ["gunicorn", "-w", "2", "-k", "gthread", "-b", "0.0.0.0:7860", "app:app"]
