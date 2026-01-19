#Use a minimal base image with Python 3.9 installed
FROM python:3.9-slim

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

#Copy project files
COPY . .

#Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

#Streamlit runs on 7860 in Hugging Face Spaces
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableXsrfProtection=false"]
