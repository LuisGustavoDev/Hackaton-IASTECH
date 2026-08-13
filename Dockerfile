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

# Atualiza pip e instala as dependências
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia o código da aplicação
COPY app ./app

# Copia os dados
COPY data ./data

# Copia os testes
COPY tests ./tests

# Comando padrão
CMD ["python", "-m", "app.main"]