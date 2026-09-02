FROM python:3.11-slim

# Configurações do Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Diretório da aplicação dentro do container
WORKDIR /app

# Dependências do sistema
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-por \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copia as dependências Python
COPY requirements.txt .

# Atualiza pip e instala as dependências.
#
# torch/torchvision vêm ANTES e do índice CPU-only: a produção roda em
# notebook sem GPU, e as rodas padrão do PyPI trazem junto as bibliotecas
# CUDA (mais de 800 MB de imagem que nunca seriam usados).
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision && \
    pip install --no-cache-dir -r requirements.txt

# Copia o código da aplicação (app/training fica de fora — ver .dockerignore)
COPY app ./app

# Copia os dados
COPY data ./data

# Copia os testes
COPY tests ./tests

EXPOSE 8000

# Comando padrão: sobe a API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
