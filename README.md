# IASTECH - Análise de Fluxogramas

Sistema desenvolvido para o Hackathon IASTECH, com o objetivo de realizar a extração e estruturação de informações presentes em fluxogramas industriais.

---

## 🚀 Tecnologias

- Python 3.11
- Docker
- Docker Compose
- FastAPI
- Uvicorn
- PyTorch
- Torchvision (Faster R-CNN)
- OpenCV
- NumPy
- Pillow
- Tesseract OCR
- Pytesseract
- PaddleOCR
- Pandas
- OpenPyXL
- pytest

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

A aplicação é uma API HTTP (FastAPI). Para subir:

```bash
docker compose up
```

A API fica disponível em `http://localhost:8000`, com a documentação
interativa em `http://localhost:8000/docs`.

### Rotas

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/api/health` | Verificação de disponibilidade |
| `POST` | `/api/process` | Recebe uma imagem de P&ID e devolve a planilha `.xlsx` |

Enviando uma imagem para processamento:

```bash
curl -X POST http://localhost:8000/api/process -F "file=@diagrama.jpg" -o resultado.xlsx
```

A planilha devolvida tem exatamente estas colunas, nesta ordem:

| TAG | Tipo | Descrição | Coordenada X | Coordenada Y | Grupo |
|---|---|---|---|---|---|

| Coluna | De onde vem |
|---|---|
| **TAG** | Texto lido pelo OCR dentro (ou ao lado) do equipamento detectado |
| **Tipo** | Classe atribuída pelo Faster R-CNN (ex.: `Válvula`, `Bomba`, `Tanque`) |
| **Descrição** | Decomposição das letras da TAG pela norma ISA-5.1 (`FT-210` → "Transmissor de Vazão"). Sem TAG legível, cai para o nome da classe detectada |
| **Coordenada X / Y** | Centro da *bounding box* do equipamento, em pixels da imagem original |
| **Grupo** | Primeiro dígito do número da TAG (`FT-210` → `2`), que indica o conjunto/equipamento a que o item pertence |

### Erros

| Situação | Status | Resposta |
|---|---|---|
| Arquivo não é uma imagem, está corrompido, vazio ou fora dos limites de tamanho | `400` | Mensagem explicando o problema |
| Checkpoint do detector ausente ou inválido | `503` | Mensagem indicando onde o modelo é esperado |
| Falha durante o processamento | `500` | Mensagem com a origem da falha |

A validação olha o **conteúdo** do arquivo, não o `content-type` nem a
extensão: um PDF renomeado para `.png` é recusado com `400`.

---

## 🧠 Detecção de equipamentos (Faster R-CNN)

O reconhecimento dos símbolos usa **Faster R-CNN** (`fasterrcnn_resnet50_fpn_v2`),
escolhido em benchmark contra RT-DETR e RetinaNet.

O treino e a inferência rodam em **máquinas diferentes**, e é por isso que
são dois módulos independentes:

| | Treino | Inferência |
|---|---|---|
| Onde roda | Desktop com GPU NVIDIA (CUDA) | Notebook, **CPU apenas** |
| Módulo | `app/training/` | `app/detection/` |
| Dependências | `requirements-treino.txt` | `requirements.txt` |
| Precisa do dataset | Sim | **Não** |
| Vai na imagem Docker | Não (`.dockerignore`) | Sim |

A única coisa que atravessa de uma máquina para a outra é **um arquivo**:
o checkpoint portátil.

```text
   MÁQUINA FORTE (GPU)                      MÁQUINA FRACA (CPU)

   dataset + anotações
          │
          ▼
   app/training/treinar.py
          │
          ▼
   faster_rcnn.pt  ──────── copiar ────────►  data/models/faster_rcnn.pt
   (~170 MB)                                          │
                                                      ▼
                                              app/detection/detector.py
                                                      │
                                                      ▼
                                                  planilha
```

### O que vai dentro do checkpoint

O arquivo `.pt` é autossuficiente. Além dos pesos, ele carrega tudo que é
necessário para reconstruir o modelo e traduzir as predições:

```python
{
    "formato_versao": 1,
    "arquitetura":    "fasterrcnn_resnet50_fpn_v2",
    "classes":        ["Acumulador", ..., "Válvula"],  # ordem canônica
    "num_classes":    24,        # sem contar o background
    "state_dict":     {...},     # pesos, já em CPU
    "metadados":      {...},     # data, épocas, mAP, versão do torch...
}
```

> **Por que a lista de classes vai junto:** o label devolvido pelo modelo
> é um número (`1`, `2`, `3`...), e o nome do equipamento é
> `classes[label - 1]` — o índice `0` é reservado para *background*. Sem a
> lista dentro do arquivo, a máquina de inferência precisaria dos JSONs de
> anotação para saber que `7` significa "Filtro". Pior: os exportadores
> COCO numeram a mesma classe de formas diferentes em cada split (no nosso
> dataset, "Bomba" é id 3 no `train.json` e id 1 no `val.json`), então a
> lista é construída casando as categorias pelo **nome**, nunca pelo id.

### 1. Preparar o dataset (máquina de treino)

As anotações vêm do Make Sense AI em Pascal VOC (`.xml`). Converta para COCO:

```bash
python -m app.training.converter_anotacoes
```

Isso lê `dataset/original/treinamento` e `dataset/original/testes` e escreve
`dataset/coco/` com `images/{train,val}` e `annotations/{train,val}.json`.

### 2. Treinar

```bash
pip install -r requirements-treino.txt
python -m app.training.treinar --data-dir dataset/coco --epochs 15 --saida data/models/faster_rcnn.pt
```

| Argumento | Padrão | Descrição |
|---|---|---|
| `--data-dir` | `dataset/coco` | Pasta com `images/` e `annotations/` em COCO |
| `--saida` | `data/models/faster_rcnn.pt` | Onde gravar o checkpoint portátil |
| `--epochs` | `15` | Épocas de fine-tuning |
| `--batch-size` | `4` | Tamanho do batch |
| `--lr` | `0.005` | Learning rate do SGD |
| `--sem-avaliacao` | desligado | Pula o cálculo de mAP no fim |

O treino usa CUDA automaticamente quando disponível e avisa se cair em CPU.

### 3. Copiar o checkpoint para a máquina de inferência

```bash
# só este arquivo — nem dataset, nem anotações, nem nada de app/training
cp faster_rcnn.pt <projeto>/data/models/faster_rcnn.pt
```

O caminho pode ser mudado pela variável de ambiente
`DETECTOR_CHECKPOINT_PATH`.

### 4. Rodar a inferência

Nada além de subir a API — o detector é carregado uma única vez, na
primeira requisição, e reaproveitado nas seguintes.

```bash
docker compose up
```

### Configuração

| Variável | Padrão | Descrição |
|---|---|---|
| `DETECTOR_CHECKPOINT_PATH` | `data/models/faster_rcnn.pt` | Caminho do checkpoint |
| `DETECTOR_SCORE_THRESHOLD` | `0.5` | Confiança mínima para uma detecção entrar na planilha |

### Aproveitar pesos já treinados pelo benchmark

O `benchmark_pid_models.py` (repositório `modelos_base_claude`) salva apenas
o `state_dict`, sem a lista de classes — carregá-lo direto em produção
daria classes trocadas. Para convertê-lo no formato portátil sem treinar
de novo:

```bash
python -m app.training.empacotar_checkpoint --pesos ../modelos_base_claude/resultados_benchmark/faster_rcnn.pt --train-json ../modelos_base_claude/dataset/annotations/train.json --val-json ../modelos_base_claude/dataset/annotations/val.json --saida data/models/faster_rcnn.pt
```

> Use os **mesmos** arquivos de anotação usados no treino daqueles pesos.
> Com JSONs diferentes o checkpoint carrega sem erro e devolve as classes
> trocadas.

---

## 🧪 Testes

```bash
docker compose run --rm app pytest
```

Localmente (fora do Docker), os testes que dependem do binário do Tesseract
são pulados automaticamente:

```bash
pytest
```

Para pular também os testes que carregam o modelo (dezenas de segundos):

```bash
pytest -m "not lento"
```

| Arquivo | O que cobre |
|---|---|
| `tests/test_validacao_imagem.py` | Arquivo vazio, PDF renomeado, imagem truncada, limites de tamanho |
| `tests/test_checkpoint.py` | Formato portátil, carga em CPU, recusa de `state_dict` puro |
| `tests/test_modelo.py` | Cabeça com `num_classes + 1` saídas e a convenção `classes[label - 1]` |
| `tests/test_detector.py` | Carga do checkpoint, formato das detecções, limiar, singleton |
| `tests/test_dataset_coco.py` | Canonicalização por nome e encoding UTF-8 das categorias |
| `tests/test_tag_service.py` | Descrição pela ISA-5.1 e Grupo pelo primeiro dígito |
| `tests/test_associacao.py` | Casamento entre texto do OCR e equipamento detectado |
| `tests/test_planilha_service.py` | Colunas exatas da planilha, em `.xlsx` e `.csv` |
| `tests/test_api.py` | Tradução dos erros de domínio em `400` / `503` / `500` |
| `tests/test_integracao_pipeline.py` | Pipeline completo com imagens reais do dataset |

---

## ✅ Testando o ambiente

Para verificar se as principais dependências estão funcionando dentro do Docker:

```bash
docker compose run --rm app python -c "import cv2, numpy, PIL, pytesseract, pandas, openpyxl, paddleocr, torch, torchvision; print('==================================='); print(' AMBIENTE IASTECH'); print('==================================='); print('Python: OK'); print('OpenCV:', cv2.__version__); print('NumPy:', numpy.__version__); print('Pillow: OK'); print('Pytesseract: OK'); print('Tesseract:', pytesseract.get_tesseract_version()); print('Pandas:', pandas.__version__); print('OpenPyXL:', openpyxl.__version__); print('PaddleOCR: OK'); print('PyTorch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('CUDA disponível:', torch.cuda.is_available(), '(produção roda em CPU)'); print('==================================='); print(' AMBIENTE OK'); print('===================================')"
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
| PyTorch | 2.x (CPU) |
| Torchvision | 0.x (CPU) |

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
PyTorch: ...
Torchvision: ...
CUDA disponível: False (produção roda em CPU)
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
│   ├── api/            → rotas HTTP
│   ├── core/           → exceções de domínio
│   ├── detection/      → inferência: Faster R-CNN em CPU
│   ├── export/
│   ├── models/         → banco SQLite e referência ISA-5.1
│   ├── pipeline/       → tratamento de imagem, MSER e OCR
│   ├── services/       → orquestração, validação, planilha
│   ├── training/       → treino do detector (NÃO vai para produção)
│   ├── ui/
│   └── main.py
│
├── data/
│   ├── docs/
│   ├── models/         → checkpoints .pt (fora do Git)
│   └── output/         → planilhas geradas (fora do Git)
│
├── dataset/
│   └── original/       → imagens e anotações Pascal VOC
│
├── tests/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-treino.txt
├── pytest.ini
├── .dockerignore
├── .gitignore
├── README.md
└── LICENSE
```

### Organização das pastas

#### `app/`

Contém o código principal da aplicação.

- **`app/api/`** — Rotas HTTP da API (FastAPI).
- **`app/core/`** — Exceções de domínio, traduzidas em status HTTP pela camada de rotas.
- **`app/detection/`** — Inferência do Faster R-CNN: leitura do checkpoint portátil e detecção dos equipamentos. Roda em CPU e não depende de nada de treino.
- **`app/export/`** — Responsável pela exportação dos dados processados.
- **`app/models/`** — Modelos e estruturas de dados utilizados pela aplicação, incluindo a tabela de referência ISA-5.1.
- **`app/pipeline/`** — Tratamento de imagem, detecção de regiões de texto (MSER), agrupamento e OCR.
- **`app/services/`** — Orquestração do processamento, validação de imagem, associação texto↔equipamento e geração da planilha.
- **`app/training/`** — Treino do detector. Roda só na máquina com GPU e é excluído da imagem Docker pelo `.dockerignore`.
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

**`503` com "Checkpoint do detector não encontrado"**

O modelo não está instalado. Treine na máquina com GPU e copie o `.pt`
para `data/models/faster_rcnn.pt`, ou aponte `DETECTOR_CHECKPOINT_PATH`
para onde ele está. Veja
[Detecção de equipamentos](#-detecção-de-equipamentos-faster-r-cnn).

**`400` em uma imagem que abre normalmente no visualizador**

A validação recusa formatos fora da lista aceita (PNG, JPEG, BMP, TIFF,
WEBP) e arquivos truncados. Converta para PNG ou JPEG e tente de novo.

**A primeira requisição demora muito mais que as seguintes**

Esperado: o checkpoint (~170 MB) é carregado na primeira chamada e fica em
memória para as próximas.

**A imagem Docker ficou gigante depois de adicionar o torch**

O `Dockerfile` instala `torch`/`torchvision` do índice CPU-only
(`download.pytorch.org/whl/cpu`). Instalar pelo PyPI padrão traz junto as
bibliotecas CUDA, que a produção não usa. Reconstrua com
`docker compose build --no-cache`.

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
