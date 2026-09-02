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
DEFAULT_DETECTOR_SCORE_THRESHOLD = 0.5


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
