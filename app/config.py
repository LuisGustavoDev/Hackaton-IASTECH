import os


# Configurações MSER

MSER_MIN_WIDTH = 2
MSER_MIN_HEIGHT = 5
MSER_MAX_WIDTH = 60
MSER_MAX_HEIGHT = 60
MSER_MIN_AREA = 20

DARK_PIXEL_THRESHOLD = 100
MIN_DARK_RATIO = 0.08


# Configurações de agrupamento

GROUP_VERTICAL_TOLERANCE = 0.6
GROUP_HORIZONTAL_TOLERANCE = 2.0
MERGE_VERTICAL_TOLERANCE = 0.7
MERGE_HORIZONTAL_TOLERANCE = 1.5


# Configurações OCR

OCR_SCALE = 3
OCR_MARGIN = 10
OCR_LANGUAGE = "eng"
OCR_PSM = 7

# Configurações do detector de equipamentos (Faster R-CNN)
#
# Lidas de variável de ambiente seguindo o mesmo padrão de DB_PATH em
# app/models/database.py: a máquina de inferência pode apontar para o
# checkpoint onde ele estiver, sem editar código.

DETECTOR_CHECKPOINT_ENV = "DETECTOR_CHECKPOINT_PATH"
DEFAULT_DETECTOR_CHECKPOINT = "data/models/faster_rcnn.pt"

DETECTOR_SCORE_THRESHOLD_ENV = "DETECTOR_SCORE_THRESHOLD"

# Política do projeto: detectar o MÁXIMO possível, mesmo símbolo mal
# desenhado ou ambíguo. Um equipamento que não aparece na planilha é
# invisível para quem confere; um a mais é uma linha que se apaga.
#
# 0.05 foi medido, não escolhido: sobre as 21 imagens anotadas de
# teste (IoU 0.5), baixar de 0.5 para 0.05 leva a revocação de 0.287
# para 0.366 e os equipamentos localizados de 43% para 63%, com o F1
# ligeiramente MELHOR (0.378 contra 0.370) — não se está trocando
# acurácia por cobertura, só andando na curva.
#
# Descer abaixo disso rende pouco: a 0.01 são +14 acertos ao custo de
# +150 falsos positivos. Reproduza com:
#     python -m app.diagnostico.matriz_confusao --ultimas 21 --curva
DEFAULT_DETECTOR_SCORE_THRESHOLD = 0.05


def caminho_checkpoint() -> str:
    return os.environ.get(
        DETECTOR_CHECKPOINT_ENV,
        DEFAULT_DETECTOR_CHECKPOINT,
    )


def limiar_deteccao() -> float:
    bruto = os.environ.get(DETECTOR_SCORE_THRESHOLD_ENV)

    if not bruto:
        return DEFAULT_DETECTOR_SCORE_THRESHOLD

    try:
        return float(bruto)
    except ValueError:
        return DEFAULT_DETECTOR_SCORE_THRESHOLD


# Associação entre texto (OCR) e equipamento (detector)
#
# Quando nenhum texto cai dentro da bounding box do equipamento, procura
# num raio proporcional ao tamanho da própria caixa — assim uma tag
# escrita ao lado de um símbolo pequeno continua sendo encontrada, sem
# que um símbolo grande "puxe" tags do outro lado do diagrama.

ASSOCIACAO_RAIO_RELATIVO = 0.75


# Validação do arquivo de entrada

IMAGEM_FORMATOS_ACEITOS = ("PNG", "JPEG", "BMP", "TIFF", "WEBP")
IMAGEM_TAMANHO_MAXIMO_BYTES = 50 * 1024 * 1024
IMAGEM_DIMENSAO_MINIMA = 16
IMAGEM_DIMENSAO_MAXIMA = 20000


# Detecção: limites de saída
#
# O torchvision usa detections_per_img=100 por padrão. Diagramas densos do
# nosso dataset chegam a 175 símbolos anotados — com o padrão, tudo que
# passa do 100º é descartado em silêncio.

DETECTOR_MAX_DETECCOES = 300

# NMS entre classes diferentes.
#
# O Faster R-CNN já aplica NMS por classe, mas nada impede que o MESMO
# símbolo saia duas vezes com rótulos diferentes (ex.: "Válvula" e
# "Outro" na mesma caixa). Símbolos de P&ID legitimamente vizinhos
# raramente passam de 0.5 de IoU, então esse limiar remove duplicata sem
# comer símbolo empilhado. Use 0 para desligar.

DETECTOR_NMS_ENTRE_CLASSES = 0.5


# Confiança mínima do OCR para um texto virar candidato a TAG.
#
# O MSER encontra traços do próprio desenho da válvula e o Tesseract
# devolve "letras" a partir deles ("(X)", "Oo", "SH"). Esses fragmentos
# vêm com confiança baixa; descartá-los antes da associação evita que
# poluam a TAG montada.

OCR_CONFIANCA_MINIMA = 40.0


# Piso interno do modelo.
#
# O torchvision descarta, dentro do próprio Faster R-CNN, tudo abaixo de
# box_score_thresh=0.05 — antes de a predição chegar ao nosso código.
# Baixar isto é a única forma de sequer AVALIAR detecções mais fracas;
# DETECTOR_SCORE_THRESHOLD filtra depois, e não consegue recuperar o que
# o modelo já jogou fora.

DETECTOR_SCORE_MINIMO_MODELO = 0.05


# CORS
#
# O curl não precisa disto, mas um frontend no navegador sim: sem os
# cabeçalhos de CORS, uma página servida em outra origem (localhost:3000,
# por exemplo) tem a requisição bloqueada pelo próprio navegador antes de
# chegar à API.
#
# O padrão "*" libera qualquer origem, o que é adequado para uma
# ferramenta local sem autenticação. Em rede aberta, liste as origens:
#     CORS_ORIGINS="http://localhost:3000,https://app.exemplo.com"

CORS_ORIGINS_ENV = "CORS_ORIGINS"
DEFAULT_CORS_ORIGINS = "*"


def origens_cors() -> list[str]:
    bruto = os.environ.get(CORS_ORIGINS_ENV, DEFAULT_CORS_ORIGINS)

    return [origem.strip() for origem in bruto.split(",") if origem.strip()]
