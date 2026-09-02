"""
Construção da arquitetura Faster R-CNN usada pelo projeto.

Este módulo é compartilhado por treino e inferência e é a ÚNICA fonte da
verdade da convenção de rótulos:

    label devolvido pelo modelo = índice da classe + 1
    nome da classe              = classes[label - 1]

O índice 0 é reservado pelo torchvision para "background". Se treino e
inferência discordarem dessa convenção, nada quebra visivelmente: as
predições continuam saindo, só que apontando para o equipamento errado.
Por isso a conversão label -> nome mora aqui e não é reescrita em cada
lugar que precisa dela.
"""

from __future__ import annotations

from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FastRCNNPredictor,
)

from app import config

ARQUITETURA = "fasterrcnn_resnet50_fpn_v2"

# Índice reservado pelo torchvision para "background".
LABEL_BACKGROUND = 0


def construir_faster_rcnn(
    num_classes: int,
    pesos_pretreinados: bool = False,
    max_deteccoes: int | None = None,
    score_minimo: float | None = None,
):
    """
    Monta o Faster R-CNN com a cabeça de classificação dimensionada para
    `num_classes` classes de equipamento + 1 (background).

    `num_classes` NÃO inclui o background — quem soma o +1 é esta função,
    exatamente como no benchmark.

    pesos_pretreinados=True baixa os pesos COCO do PyTorch Hub e só faz
    sentido no treino (exige internet na primeira execução). Na inferência
    use False: os pesos vêm inteiros do checkpoint e a máquina de produção
    não precisa de rede nenhuma.

    `max_deteccoes` substitui o `box_detections_per_img` do torchvision,
    que vale 100 por padrão. Diagramas densos do nosso dataset chegam a
    175 símbolos anotados numa imagem só; com o padrão, tudo que passa do
    100º é descartado sem aviso. Não afeta os pesos, então pode ser
    mudado sem retreinar.

    `score_minimo` é o `box_score_thresh`: o corte que o modelo aplica
    INTERNAMENTE, antes de a predição chegar ao nosso código. É
    diferente do limiar do detector, que filtra o que já saiu — o que
    for descartado aqui não tem como ser recuperado depois.
    """
    if num_classes < 1:
        raise ValueError(
            f"num_classes precisa ser >= 1 (sem contar o background), "
            f"recebido: {num_classes}"
        )

    if max_deteccoes is None:
        max_deteccoes = config.DETECTOR_MAX_DETECCOES

    if score_minimo is None:
        score_minimo = config.DETECTOR_SCORE_MINIMO_MODELO

    weights = (
        FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        if pesos_pretreinados
        else None
    )

    modelo = fasterrcnn_resnet50_fpn_v2(
        weights=weights,
        weights_backbone=None,
        box_detections_per_img=max_deteccoes,
        box_score_thresh=score_minimo,
    )

    in_features = modelo.roi_heads.box_predictor.cls_score.in_features

    # +1 pela classe "background", exigida pelo Faster R-CNN
    modelo.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes + 1,
    )

    return modelo


def nome_da_classe(label: int, classes: list[str]) -> str:
    """
    Traduz o label bruto devolvido pelo modelo (1..C) para o nome da
    classe, usando a lista canônica guardada no checkpoint.
    """
    indice = int(label) - 1

    if indice < 0 or indice >= len(classes):
        raise ValueError(
            f"Label {label} fora do intervalo do checkpoint "
            f"(1..{len(classes)}). O checkpoint provavelmente não "
            f"corresponde ao modelo carregado."
        )

    return classes[indice]
