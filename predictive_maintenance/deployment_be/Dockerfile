
FROM python:3.10-slim

# ----------------------------------------------------
# Python & HF runtime settings
# ----------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

WORKDIR /app

# ----------------------------------------------------
# System dependencies (minimal)
# ----------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------
# Install Python dependencies
# ----------------------------------------------------
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ----------------------------------------------------
# Copy application code
# ----------------------------------------------------
COPY . .

# ----------------------------------------------------
# Production WSGI server (HF SAFE)
# ----------------------------------------------------
CMD ["gunicorn", \
     "-w", "2", \
     "--threads", "4", \
     "--timeout", "60", \
     "--graceful-timeout", "30", \
     "--keep-alive", "5", \
     "-b", "0.0.0.0:7860", \
     "app:app"]
