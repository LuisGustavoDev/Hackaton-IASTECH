"""
Construção do Faster R-CNN e a convenção de rótulos.

A convenção "label = índice + 1" é o tipo de coisa que, se quebrar, não
gera erro nenhum: o sistema continua devolvendo detecções, só que com o
nome do equipamento errado. Daí os testes.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from app.detection.modelo import (  # noqa: E402
    ARQUITETURA,
    construir_faster_rcnn,
    nome_da_classe,
)

CLASSES = ["Bomba", "Conexão", "Válvula"]


@pytest.mark.lento
def test_cabeca_tem_uma_saida_a_mais_que_as_classes():
    """24 classes de equipamento -> 25 saídas (a extra é o background)."""
    modelo = construir_faster_rcnn(24)

    assert modelo.roi_heads.box_predictor.cls_score.out_features == 25
    assert modelo.roi_heads.box_predictor.bbox_pred.out_features == 25 * 4


@pytest.mark.lento
def test_construcao_nao_baixa_pesos_por_padrao():
    """
    Sem pesos pré-treinados a máquina de inferência não precisa de rede.
    Se este teste passar offline, o requisito está atendido.
    """
    modelo = construir_faster_rcnn(3)

    assert modelo.roi_heads.box_predictor.cls_score.out_features == 4


def test_num_classes_invalido_e_recusado():
    with pytest.raises(ValueError, match="num_classes"):
        construir_faster_rcnn(0)


def test_label_1_e_a_primeira_classe():
    assert nome_da_classe(1, CLASSES) == "Bomba"


def test_ultimo_label_e_a_ultima_classe():
    assert nome_da_classe(len(CLASSES), CLASSES) == "Válvula"


@pytest.mark.parametrize("label", [0, -1, 4, 99])
def test_label_fora_do_intervalo_e_erro(label):
    with pytest.raises(ValueError, match="fora do intervalo"):
        nome_da_classe(label, CLASSES)


def test_arquitetura_e_a_do_benchmark():
    assert ARQUITETURA == "fasterrcnn_resnet50_fpn_v2"
