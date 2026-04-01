FROM python:3.10-slim-bookworm

# 1. Instalar dependencias del sistema (Actualizar lista + Instalar wget + librerías de imagen)
RUN apt-get update && apt-get install -y \
    wget \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Instalar librerías de Python primero (mejor para la caché de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Ahora wget sí funcionará (Exit code 127 solucionado)
RUN wget -q https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26x-seg.pt -O yoloe-26x-seg.pt

COPY handler.py .

EXPOSE 8000

CMD ["uvicorn", "handler:app", "--host", "0.0.0.0", "--port", "8000"]
