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
| `POST` | `/api/process` | Processa uma imagem. `?formato=xlsx` (padrão), `csv` ou `json` |
| `GET` | `/api/resultado/{id}` | Baixa a planilha de uma execução já processada. `?formato=xlsx` (padrão) ou `csv` |
| `GET` | `/api/execucoes` | Histórico de processamentos, do mais recente para o mais antigo |

Enviando uma imagem para processamento:

```bash
curl -X POST http://localhost:8000/api/process -F "file=@diagrama.jpg" -o resultado.xlsx
```

### Consumindo pelo frontend

O `curl` acima já recebe o arquivo. Um **navegador**, porém, precisa de
duas coisas a mais, e ambas estão configuradas:

- **CORS.** Sem os cabeçalhos, o navegador bloqueia a chamada antes de ela
  sair — e como o `curl` não passa por CORS, o problema só aparece quando
  a interface entra em cena. Origens permitidas via `CORS_ORIGINS`
  (padrão `*`, adequado para uma ferramenta local sem autenticação).
- **`Content-Disposition` exposto.** Por padrão o JavaScript só enxerga
  alguns cabeçalhos da resposta, e esse não está entre eles: sem
  `expose_headers`, o front recebe o arquivo mas não consegue ler o nome
  sugerido para o download.

O fluxo que a Etapa 04 do Plano de Desenvolvimento pede — mostrar a
quantidade encontrada e uma pré-visualização da lista, e só então exportar
o Excel — se faz em duas chamadas, sem reprocessar a imagem:

```js
// 1. processa e recebe as linhas para pré-visualizar
const dados = new FormData();
dados.append("file", arquivo);

const resposta = await fetch("http://localhost:8000/api/process?formato=json", {
  method: "POST",
  body: dados,
});

if (!resposta.ok) {
  const { detail } = await resposta.json();
  throw new Error(detail);          // 400 arquivo inválido, 503 sem modelo
}

const { execucao_id, quantidade, linhas, download } = await resposta.json();
// -> renderiza `quantidade` e a tabela `linhas`

// 2. só quando o usuário clicar em "Exportar Excel"
const arquivoXlsx = await fetch(`http://localhost:8000${download.xlsx}`);
const blob = await arquivoXlsx.blob();
```

Resposta de `formato=json`:

```json
{
  "execucao_id": 23,
  "arquivo": "101.jpg",
  "quantidade": 20,
  "linhas": [
    {"TAG": "PI-0013", "Tipo": "Instrumento", "Descrição": "Indicador de Pressão",
     "Coordenada X": 159, "Coordenada Y": 142, "Grupo": "1"}
  ],
  "download": {
    "xlsx": "/api/resultado/23?formato=xlsx",
    "csv": "/api/resultado/23?formato=csv"
  }
}
```

`download` vem `null` quando a execução não pôde ser gravada no banco (a
persistência é best-effort): não há de onde baixar depois, e um link que
daria 404 seria pior que a ausência dele.

A planilha devolvida tem exatamente estas colunas, nesta ordem:

| TAG | Tipo | Descrição | Coordenada X | Coordenada Y | Grupo |
|---|---|---|---|---|---|

| Coluna | De onde vem |
|---|---|
| **TAG** | Texto lido pelo OCR dentro (ou ao lado) do equipamento detectado, normalizado para `LETRAS-NÚMERO`. Um balão ISA traz as letras numa linha e o número na outra (`PI` / `0013`): todos os textos internos são juntados na ordem de leitura antes de interpretar. Texto que não casa com o padrão ISA fica de fora — o Tesseract lê o traço dos próprios símbolos como letras (`(X)`, `Oo`, `SH`) e isso não pode virar identificador |
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

### Primeira configuração da máquina de treino

Feito uma vez só, na máquina com GPU. **O treino roda direto no sistema,
fora do Docker** — o `docker-compose.yml` não expõe GPU ao container.

**1. Pré-requisitos**

| Item | Verificação |
|---|---|
| Driver NVIDIA recente | `nvidia-smi` (precisa mostrar a GPU e CUDA 12.x) |
| Git | `git --version` |
| Python 3.11–3.13 | `py --version` (Windows) / `python3 --version` (Linux) |

Não é preciso instalar o CUDA Toolkit separadamente: as rodas do PyTorch
já trazem as bibliotecas CUDA. Só o **driver** precisa estar atualizado.

**2. Clonar o repositório**

```bash
git clone https://github.com/LuisGustavoDev/Hackaton-IASTECH.git
```

O dataset anotado (`dataset/original/`, 92 imagens com os `.xml`) está
versionado — vem junto no clone, não precisa ser copiado à parte.

**3. Criar o ambiente virtual**

```bash
py -3.13 -m venv .venv
```

Ative-o: `.venv\Scripts\activate` (Windows) ou `source .venv/bin/activate` (Linux).

**4. Instalar o PyTorch com CUDA — antes de tudo**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Esta é a etapa que mais dá errado. Instalar `torch` pelo PyPI padrão traz
a build de **CPU**, e o treino roda dezenas de vezes mais devagar sem
avisar. A GTX 1660 (Turing, SM 7.5) é atendida pela build `cu126`.

**5. Instalar o resto**

```bash
pip install -r requirements-treino.txt
```

**6. Confirmar que a GPU foi reconhecida**

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'sem GPU')"
```

Tem de imprimir uma versão terminada em `+cu126` e `CUDA: True`. Se
aparecer `+cpu` ou `CUDA: False`, refaça o passo 4 com
`pip uninstall torch torchvision` antes.

### 1. Preparar o dataset (máquina de treino)

As anotações vêm do Make Sense AI em Pascal VOC (`.xml`). Converta para COCO:

```bash
python -m app.training.converter_anotacoes
```

Isso lê `dataset/original/treinamento` e `dataset/original/testes` e escreve
`dataset/coco/` com `images/{train,val}` e `annotations/{train,val}.json`.

### 2. Treinar

```bash
python -m app.training.treinar --data-dir dataset/coco --epochs 15 --batch-size 2 --saida data/models/faster_rcnn.pt
```

| Argumento | Padrão | Descrição |
|---|---|---|
| `--data-dir` | `dataset/coco` | Pasta com `images/` e `annotations/` em COCO |
| `--saida` | `data/models/faster_rcnn.pt` | Onde gravar o checkpoint portátil |
| `--epochs` | `15` | Épocas de fine-tuning |
| `--batch-size` | `4` | Tamanho do batch |
| `--lr` | `0.005` | Learning rate do SGD |
| `--sem-avaliacao` | desligado | Pula o cálculo de mAP no fim |

O treino usa CUDA automaticamente quando disponível e avisa se cair em
CPU.

> **VRAM:** a GTX 1660 tem 6 GB. Comece com `--batch-size 2`. Se
> aparecer `torch.OutOfMemoryError: CUDA out of memory`, baixe para
> `--batch-size 1`; se sobrar memória (acompanhe com `nvidia-smi`),
> suba para 4. No Windows, `--num-workers 0` evita problemas de
> multiprocessing se o DataLoader travar.

A primeira execução baixa os pesos pré-treinados em COCO de
`download.pytorch.org` (~170 MB) e precisa de internet. As execuções
seguintes usam o cache local.

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

### 5. Testar na máquina de inferência

Depois de copiar o `.pt`, três checagens em ordem crescente de custo.

**a) O checkpoint é válido?** Não constrói o modelo nem precisa de imagem
— é instantâneo e mostra o que veio de dentro do arquivo, incluindo o mAP
registrado no treino:

```bash
python -m app.detection.verificar
```

```text
========================================================================
 CHECKPOINT
========================================================================
arquivo       : .../data/models/faster_rcnn.pt
tamanho       : 165.8 MB
formato       : versão 1
arquitetura   : fasterrcnn_resnet50_fpn_v2
classes       : 24
    label  1 -> Acumulador
    ...
    label 24 -> Válvula
metadados     :
    epocas             = 50
    mAP@[.5:.95]       = 0.31
    ...
```

**b) O modelo roda em CPU e detecta alguma coisa?** Carrega o detector de
verdade e roda numa imagem, sem precisar do Tesseract nem da API:

```bash
python -m app.detection.verificar --imagem dataset/original/testes/diagramas/qtd_baixa/qld_alta/101.jpg
```

Mostra tempo de carga, tempo de inferência, contagem por classe e as
detecções mais confiantes. Para inspecionar as caixas visualmente:

```bash
python -m app.detection.verificar --imagem dataset/original/testes/diagramas/qtd_baixa/qld_alta/101.jpg --limiar 0.3 --salvar data/output/anotada.png
```

Se não sair nenhuma detecção, baixe o limiar (`--limiar 0.1`) antes de
concluir que há algo errado: modelo pouco treinado gera confiança baixa.

**c) O pipeline completo, com OCR e planilha.** O Tesseract só está
instalado dentro do container, então este teste roda no Docker:

```bash
docker compose up --build
```

Em outro terminal:

```bash
curl -X POST http://localhost:8000/api/process -F "file=@dataset/original/testes/diagramas/qtd_baixa/qld_alta/101.jpg" -o resultado.xlsx
```

Abra o `resultado.xlsx` e confira as seis colunas. Para checar a rejeição
de arquivo inválido:

```bash
curl -i -X POST http://localhost:8000/api/process -F "file=@README.md;filename=falso.png;type=image/png"
```

Tem de responder `400` com "O arquivo enviado não é uma imagem
reconhecível".

---

### Configuração

| Variável | Padrão | Descrição |
|---|---|---|
| `DETECTOR_CHECKPOINT_PATH` | `data/models/faster_rcnn.pt` | Caminho do checkpoint |
| `DETECTOR_SCORE_THRESHOLD` | `0.05` | Confiança mínima para uma detecção entrar na planilha. Baixo de propósito — ver [Política de detecção](#política-de-detecção-cobertura-acima-de-precisão) |
| `CORS_ORIGINS` | `*` | Origens permitidas para o frontend, separadas por vírgula |
| `DB_PATH` | `:memory:` | Banco de execuções. O `docker-compose.yml` aponta para `data/iastech.db`; com o padrão `:memory:` o histórico some quando o processo cai |

Ajustáveis em `app/config.py` (não exigem retreinar):

| Constante | Padrão | Descrição |
|---|---|---|
| `DETECTOR_MAX_DETECCOES` | `300` | Teto de detecções por imagem. O torchvision usa 100; diagramas densos do dataset chegam a 175 símbolos e perderiam o excedente em silêncio |
| `DETECTOR_NMS_ENTRE_CLASSES` | `0.5` | IoU para descartar caixas duplicadas classificadas em classes diferentes. `0` desliga |
| `OCR_CONFIANCA_MINIMA` | `40.0` | Confiança mínima do Tesseract para um texto virar candidato a TAG |
| `ASSOCIACAO_RAIO_RELATIVO` | `0.75` | Raio de busca por uma TAG fora da caixa, como fração da diagonal dela |

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

## 🗄️ Histórico de execuções

Cada imagem processada é registrada no SQLite (`app/models/database.py`),
em duas tabelas:

| Tabela | Uma linha por | Guarda |
|---|---|---|
| `execucoes` | imagem processada | arquivo, dimensões, **checkpoint**, limiar, nº de detecções, nº de TAGs lidas, tempos de detecção/OCR/total |
| `deteccoes` | equipamento detectado | classe, score, bounding box, TAG normalizada, texto bruto do OCR, Descrição, Grupo |

Existem porque a planilha em `data/output` responde *"o que tem neste
diagrama?"*, e há uma segunda pergunta — *"o modelo está melhorando?"* —
que exige comparar rodadas e por isso não sobrevive num arquivo
regravado a cada execução.

O `checkpoint` fica gravado em cada linha justamente para isso: sem saber
qual modelo gerou cada resultado, os números de duas rodadas não são
comparáveis.

```python
from app.models import execucoes

execucoes.listar(limite=10)        # rodadas mais recentes
execucoes.obter(3)                 # uma rodada
execucoes.deteccoes_de(3)          # o que o modelo previu nela
```

Os indicadores da Etapa 05 do Plano de Desenvolvimento saem direto de SQL:

```sql
SELECT COUNT(*)                              AS execucoes,
       AVG(tempo_total_ms)                   AS tempo_medio_ms,
       SUM(qtd_tags_lidas) * 1.0
           / NULLIF(SUM(qtd_deteccoes), 0)   AS taxa_acerto_ocr
  FROM execucoes;
```

### Matriz de confusão

O gabarito **não** é duplicado no banco: os `.xml` em `dataset/original`
já estão versionados. A matriz sai do cruzamento entre eles e a tabela
`deteccoes`, casando por IoU e classe:

- **VP** — predição que casou com uma anotação da mesma classe;
- **FP** — predição sem anotação correspondente, ou com a classe errada;
- **FN** — anotação que nenhuma predição cobriu.

> **VN não existe naturalmente em detecção de objetos.** Não há um
> conjunto finito de "caixas que poderiam ter sido detectadas e
> corretamente não foram", então não há o que contar. VP, FP e FN saem do
> cruzamento; o VN pedido na reunião de 14/08 precisa antes de uma
> definição de negócio (por exemplo: classes do catálogo ausentes no
> diagrama e corretamente não previstas). Registrado aqui para ser
> combinado com o cliente, e não inventado na hora de calcular.

### Configuração

O padrão é `:memory:`, e nesse modo o histórico morre junto com o
processo. O `docker-compose.yml` aponta `DB_PATH` para
`data/iastech.db`, que o volume `./data` expõe no host. O arquivo está
no `.gitignore`: é resultado de rodada, não código.

O registro é **best-effort**. Se o banco falhar, o processamento entrega
a planilha assim mesmo e devolve `execucao_id = None` — telemetria não
pode derrubar o produto. Só `sqlite3.Error` é engolido; qualquer outra
exceção continua subindo, porque seria bug de verdade.

---

## 📐 Medição do modelo (Etapa 05)

Três CLIs em `app/diagnostico/`, para responder *"o modelo está
melhorando?"* com número em vez de impressão. Precisam do dataset e, no
caso do lote, do Tesseract — ambos existem dentro do container via
volume, então o caminho normal é `docker compose run --rm app ...`.

### 1. Rodar o conjunto de teste inteiro

```bash
docker compose run --rm app python -m app.diagnostico.lote --imagens dataset/original/testes
```

Processa cada imagem pelo **mesmo** `processar_imagem` que a API usa — uma
medição que passasse por um caminho paralelo mediria esse caminho, não a
produção — e grava tudo no banco de execuções.

### 2. Matriz de confusão

```bash
docker compose run --rm app python -m app.diagnostico.matriz_confusao --ultimas 21
```

Cruza as detecções gravadas com os `.xml` do gabarito, casando as caixas
por IoU. Também aceita `--execucoes 3,4,5`, `--checkpoint <caminho>` e
`--iou 0.5`.

O casamento acontece **antes** de comparar as classes. Uma válvula
rotulada como "Outro" aparece como classe trocada, e não como um FP
somado a um FN — o modelo achou o equipamento e errou só o rótulo, que é
um problema diferente, com solução diferente.

> **VN não é reportado.** Detecção de objetos não tem verdadeiro negativo
> natural: não existe um conjunto finito de "caixas que poderiam ter sido
> previstas e corretamente não foram". VP, FP e FN saem do cruzamento; o
> VN pedido na reunião de 14/08 precisa antes de uma definição de negócio
> a combinar com o cliente. O relatório diz "não aplicável" em vez de
> imprimir um zero que pareceria medição.

### 3. Calibrar o limiar do OCR

```bash
docker compose run --rm app python -m app.diagnostico.calibrar_ocr --imagens dataset/original/testes
```

Roda o OCR sem filtro nenhum, separa o que casa com o padrão ISA do que é
ruído e varre os limiares, mostrando quanta TAG legítima cada corte
custaria. Use para escolher `OCR_CONFIANCA_MINIMA` com dados.

### 4. Relatório comparativo (Excel)

```bash
docker compose run --rm app python -m app.diagnostico.relatorio --saida data/output/relatorio.xlsx
```

Agrupa as execuções gravadas **por checkpoint**: rodar o lote com um
modelo, depois com outro, e gerar o relatório uma vez põe as duas rodadas
lado a lado em todas as abas.

| Aba | Conteúdo |
|---|---|
| `Leia-me` | Como cada número é apurado e o que mais épocas podem ou não resolver |
| `Resumo` | Uma linha por checkpoint: VP/FP/FN, precisão, revocação, F1, tempos, e os metadados do treino (épocas, mAP, loss) lidos do próprio `.pt` |
| `Curva` | Precisão × revocação em cada limiar de score |
| `Por classe` | Desempenho por classe **cruzado com quantos exemplos de treino a classe tem** |
| `Por imagem` | Ordenada pela pior revocação — por onde começar a olhar |
| `Confusões` | Pares de classe trocados |
| `Detecções` | Uma linha por predição e por anotação perdida, com IoU, coordenadas e TAG |

Todas as abas têm filtro do Excel ligado e cabeçalho congelado.

**A coluna que faz o diagnóstico** é `Exemplos no treino`, na aba
`Por classe`, lida ao lado da revocação:

- muitos exemplos + revocação baixa → **subtreino**, mais épocas tendem a ajudar;
- poucos exemplos + revocação zero → **falta de dado**, épocas não criam exemplo;
- muitos exemplos + revocação zero → **anotação suspeita**, vale reanotar antes de treinar de novo.

---

### Política de detecção: cobertura acima de precisão

O sistema é calibrado para **detectar o máximo possível**, incluindo
símbolos mal desenhados ou ambíguos. A razão é assimétrica: um
equipamento que não aparece na planilha é invisível para quem confere o
diagrama, enquanto um a mais é uma linha que se apaga.

Na prática isso significa `DETECTOR_SCORE_THRESHOLD = 0.05`, e não o 0.5
que seria o padrão conservador. O valor foi **medido**, não escolhido —
sobre as 21 imagens anotadas de teste, IoU 0.5:

| limiar | previstas | VP | FP | FN | classe trocada | precisão | revocação | F1 |
|---|---|---|---|---|---|---|---|---|
| 0.01 | 946 | 317 | 408 | 289 | 221 | 0.335 | 0.383 | — |
| **0.05** | **776** | **303** | **258** | **309** | **215** | **0.390** | **0.366** | **0.378** |
| 0.10 | 702 | 292 | 204 | 329 | 206 | 0.416 | 0.353 | 0.382 |
| 0.30 | 564 | 269 | 133 | 396 | 162 | 0.477 | 0.325 | 0.387 |
| 0.50 | 455 | 237 | 92 | 464 | 126 | 0.521 | 0.287 | 0.370 |
| 0.80 | 296 | 158 | 46 | 577 | 92 | 0.534 | 0.191 | 0.281 |

Reproduza com:

```bash
docker compose run --rm app python -m app.diagnostico.matriz_confusao --ultimas 21 --curva
```

Três coisas que essa tabela mostra:

**Não se está trocando acurácia por cobertura.** O F1 é praticamente
plano entre 0.05 e 0.5 (0.378 contra 0.370) — baixar o limiar só anda na
curva, não piora o modelo.

**O limiar não é o principal limitador da cobertura.** Mesmo em 0.01,
289 equipamentos anotados não recebem caixa nenhuma. O modelo
simplesmente não propõe região ali. Descer de 0.05 para 0.01 custa +150
falsos positivos para ganhar +14 acertos — disponível via variável de
ambiente, mas o retorno é ruim.

**O gargalo maior é classificação, não detecção.** Em 0.05 o modelo põe
caixa no lugar certo em **62,6%** dos equipamentos anotados, mas acerta o
rótulo em apenas 36,6%. Os 215 restantes estão localizados com o `Tipo`
errado — já foram "detectados" no sentido do requisito, e a correção
deles passa por treino e qualidade de anotação, não por limiar.

#### O que NÃO foi afrouxado, e por quê

**NMS entre classes continua ligado.** Desligá-lo acrescenta 654
predições e ganha 2 acertos; a precisão desaba de 0.390 para 0.213. O que
ele remove é a mesma caixa saindo duas vezes com rótulos diferentes —
duplicata, não cobertura.

**O teto de 300 detecções por imagem não está limitando.** A imagem mais
densa do conjunto produziu 124 detecções em 0.05 e 168 em 0.01.

**Equipamento sem TAG legível continua na planilha**, com o `Tipo` e as
coordenadas preenchidos e a coluna TAG vazia. O filtro do padrão ISA
limpa o campo TAG, nunca descarta a linha.

---

### Baseline — Faster R-CNN, 15 épocas

Medido sobre as 21 imagens anotadas de `dataset/original/testes`, IoU 0.5.
Com o limiar **0.05** que passou a ser o padrão:

| Métrica | Valor |
|---|---|
| VP | 303 |
| FP | 258 |
| FN | **309** |
| Classe trocada | 215 |
| Precisão | 0.390 |
| Revocação | 0.366 |
| F1 | 0.378 |
| Equipamentos localizados | 62,6% |

O erro dominante é o falso negativo: o modelo **deixa de ver** mais do que
erra. E 10 das 24 classes ficaram em zero — as mesmas que têm ≤8 exemplos
no treino.

Para referência, o mesmo checkpoint no limiar conservador de 0.5 dava
VP 237 / FP 92 / FN 464, revocação 0.287.

Maior confusão isolada: **83× `Outro` previsto como `Instrumento`**.

> `174.jpg` está em `dataset/original/testes` sem o `.xml` correspondente
> e por isso fica fora da medição. O relatório avisa na linha
> "sem gabarito".

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
| `tests/test_verificar.py` | CLI de diagnóstico: inspeção do checkpoint e imagem anotada |
| `tests/test_execucoes.py` | Persistência das rodadas: contexto do checkpoint, contagens e durabilidade em arquivo |
| `tests/test_matriz_confusao.py` | Casamento por IoU e contagem de VP/FP/FN — se ele errar, todo o relatório mente junto |
| `tests/test_relatorio.py` | Casamento detalhado, totais e o cruzamento com os exemplos de treino |
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
│   ├── diagnostico/    → medição: lote, matriz, relatório, OCR
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
- **`app/diagnostico/`** — CLIs de medição (Etapa 05): rodar o conjunto de teste em lote, matriz de confusão contra o gabarito e calibração do limiar do OCR. Não fazem parte do fluxo que atende o usuário.
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
