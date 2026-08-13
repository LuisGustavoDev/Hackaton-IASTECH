# IASTECH - Análise de Fluxogramas

Sistema desenvolvido para o Hackathon IASTECH, com o objetivo de realizar a extração e estruturação de informações presentes em fluxogramas industriais.

---

## 🚀 Tecnologias

- Python 3.11
- Docker
- Docker Compose
- OpenCV
- NumPy
- Pillow
- Tesseract OCR
- Pytesseract
- PaddleOCR
- Pandas
- OpenPyXL

---

## 📋 Pré-requisitos

Para executar o projeto utilizando Docker, é necessário ter instalado:

- Docker
- Docker Compose
- Git

O projeto foi configurado para funcionar em ambientes **Linux e Windows**.

### Verificar instalação

```bash
docker --version
docker compose version
git --version
```

---

## 📦 Instalação

### 1. Clonar o repositório

```bash
git clone https://github.com/LuisGustavoDev/Hackaton-IASTECH.git
```

Entre na pasta do projeto:

```bash
cd "Hackton-IASTECH"
```

---

## 🐳 Executando com Docker

O Docker é utilizado para padronizar o ambiente de desenvolvimento entre os integrantes da equipe.

Dessa forma, todos utilizam as mesmas versões do Python e das bibliotecas necessárias, independentemente de estarem utilizando Linux ou Windows.

### Primeira execução

Na primeira execução, construa a imagem:

```bash
docker compose build
```

Depois, inicie o projeto:

```bash
docker compose up
```

Ou execute as duas etapas de uma vez:

```bash
docker compose up --build
```

### Executando em segundo plano

Para executar o container em segundo plano:

```bash
docker compose up -d
```

Para visualizar os logs:

```bash
docker compose logs -f
```

### Parando o projeto

Para parar os containers:

```bash
docker compose down
```

---

## 🛠️ Desenvolvimento

O projeto utiliza volumes do Docker para compartilhar os arquivos locais com o container.

As principais pastas compartilhadas são:

```text
app/    → código da aplicação
data/   → arquivos e dados utilizados pelo sistema
tests/  → testes
```

Dessa forma, alterações realizadas no código local podem ser utilizadas pelo container sem a necessidade de reconstruir a imagem a cada alteração.

### Quando reconstruir a imagem

Não é necessário reconstruir a imagem quando houver apenas alterações no código Python:

```bash
docker compose up
```

Porém, caso sejam alterados arquivos relacionados ao ambiente, como:

- `Dockerfile`
- `requirements.txt`
- dependências do projeto

é necessário reconstruir a imagem:

```bash
docker compose up --build
```

---

## ▶️ Executando a aplicação

A aplicação pode ser executada utilizando:

```bash
docker compose up
```

O comando principal utilizado pelo container atualmente é:

```bash
python -m app.main
```

Também é possível executar diretamente:

```bash
docker compose run --rm app python -m app.main
```

---

## ✅ Testando o ambiente

Para verificar se as principais dependências estão funcionando dentro do Docker:

```bash
docker compose run --rm app python -c "import cv2, numpy, PIL, pytesseract, pandas, openpyxl, paddleocr; print('==================================='); print(' AMBIENTE IASTECH'); print('==================================='); print('Python: OK'); print('OpenCV:', cv2.__version__); print('NumPy:', numpy.__version__); print('Pillow: OK'); print('Pytesseract: OK'); print('Tesseract:', pytesseract.get_tesseract_version()); print('Pandas:', pandas.__version__); print('OpenPyXL:', openpyxl.__version__); print('PaddleOCR: OK'); print('==================================='); print(' AMBIENTE OK'); print('===================================')"
```

O ambiente foi validado com:

| Dependência | Versão |
|---|---|
| Python | 3.11 |
| OpenCV | 4.10.0 |
| NumPy | 2.3.5 |
| Tesseract | 5.5.0 |
| Pandas | 3.0.5 |
| OpenPyXL | 3.1.5 |
| Pillow | OK |
| Pytesseract | OK |
| PaddleOCR | OK |

Se todas as dependências forem carregadas corretamente, será exibido:

```text
===================================
 AMBIENTE IASTECH
===================================
Python: OK
OpenCV: ...
NumPy: ...
Pillow: OK
Pytesseract: OK
Tesseract: ...
Pandas: ...
OpenPyXL: ...
PaddleOCR: OK
===================================
 AMBIENTE OK
===================================
```

---

## 📁 Estrutura do projeto

```text
Hackton IASTECH/
│
├── app/
│   ├── core/
│   ├── export/
│   ├── models/
│   ├── ui/
│   └── main.py
│
├── data/
│   └── docs/
│
├── tests/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
├── .gitignore
├── README.md
└── LICENSE
```

### Organização das pastas

#### `app/`

Contém o código principal da aplicação.

- **`app/core/`** — Componentes relacionados ao processamento principal do sistema.
- **`app/export/`** — Responsável pela exportação dos dados processados.
- **`app/models/`** — Modelos e estruturas de dados utilizados pela aplicação.
- **`app/ui/`** — Componentes relacionados à interface da aplicação.
- **`app/main.py`** — Ponto de entrada da aplicação.

#### `data/`

Armazena arquivos utilizados durante o processamento.

```text
data/
└── docs/
```

> Arquivos de entrada e resultados gerados localmente não devem ser enviados ao GitHub quando estiverem cobertos pelo `.gitignore`.

#### `tests/`

Contém os testes automatizados e testes auxiliares do projeto.

---

### Branches

A branch principal do projeto é:

```text
main
```

Para novas funcionalidades, recomenda-se trabalhar em uma branch própria:

```bash
git checkout -b feature/nome-da-funcionalidade
```

Exemplo:

```bash
git checkout -b feature/processamento-imagem
```

Após concluir a funcionalidade, as alterações devem ser revisadas antes de serem integradas à `main`.

---

## 🧰 Comandos úteis do Docker

| Ação | Comando |
|---|---|
| Ver containers | `docker compose ps` |
| Ver logs | `docker compose logs -f` |
| Construir a imagem | `docker compose build` |
| Reconstruir sem cache | `docker compose build --no-cache` |
| Iniciar | `docker compose up` |
| Iniciar em segundo plano | `docker compose up -d` |
| Parar | `docker compose down` |
| Executar comando no ambiente | `docker compose run --rm app COMANDO` |

Exemplo:

```bash
docker compose run --rm app python --version
```

---

## 🩹 Solução de problemas

**Container encerra com `exit code 0`**

Caso apareça `exited with code 0`, isso significa que o processo principal terminou normalmente. Para executar novamente a aplicação:

```bash
docker compose run --rm app python -m app.main
```

**Alterei o `requirements.txt`**

Reconstrua a imagem:

```bash
docker compose up --build
```

**Alterei apenas arquivos Python**

Normalmente não é necessário reconstruir a imagem:

```bash
docker compose up
```

**Verificar o estado dos containers**

```bash
docker compose ps
```

---

## 📄 Licença

Este projeto está distribuído sob a licença MIT. Consulte o arquivo `LICENSE` para obter os termos completos da licença.

---

## 🏆 Hackathon IASTECH

Projeto desenvolvido como parte do Hackathon IASTECH.

O objetivo do projeto é desenvolver uma solução para processamento e extração de informações presentes em fluxogramas industriais, utilizando técnicas de processamento de imagens, OCR e estruturação de dados.
