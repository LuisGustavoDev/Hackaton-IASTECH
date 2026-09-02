"""
Detector em CPU, carregado a partir de um checkpoint portátil.

O checkpoint destes testes é construído na hora com pesos aleatórios: o
que está sendo verificado é a mecânica de carregar e traduzir rótulos, não
a qualidade da detecção.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from app.core.erros import CheckpointInvalidoError  # noqa: E402
from app.detection.checkpoint import salvar_checkpoint  # noqa: E402
from app.detection.detector import (  # noqa: E402
    DetectorEquipamentos,
    obter_detector,
    redefinir_detector,
)
from app.detection.modelo import construir_faster_rcnn  # noqa: E402

CLASSES = ["Bomba", "Conexão", "Válvula"]

pytestmark = pytest.mark.lento


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """Um Faster R-CNN de verdade, com pesos aleatórios."""
    destino = tmp_path_factory.mktemp("modelos") / "faster_rcnn.pt"

    modelo = construir_faster_rcnn(len(CLASSES), pesos_pretreinados=False)

    return salvar_checkpoint(
        destino, modelo, CLASSES, metadados={"origem": "teste"}
    )


@pytest.fixture(scope="module")
def detector(checkpoint):
    return DetectorEquipamentos(checkpoint, limiar=0.0)


def test_classes_vem_do_checkpoint(detector):
    assert detector.classes == CLASSES


def test_metadados_vem_do_checkpoint(detector):
    assert detector.metadados["origem"] == "teste"


def test_roda_em_cpu(detector):
    assert detector.device.type == "cpu"
    assert all(p.device.type == "cpu" for p in detector.modelo.parameters())


def test_modelo_fica_em_modo_avaliacao(detector):
    assert not detector.modelo.training


def test_deteccoes_tem_o_formato_esperado(detector):
    imagem = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

    deteccoes = detector.detectar(imagem)

    assert isinstance(deteccoes, list)

    for deteccao in deteccoes:
        assert deteccao["classe"] in CLASSES
        assert 0.0 <= deteccao["score"] <= 1.0
        assert deteccao["x1"] <= deteccao["x2"]
        assert deteccao["y1"] <= deteccao["y2"]
        assert deteccao["centro_x"] == (deteccao["x1"] + deteccao["x2"]) // 2
        assert deteccao["centro_y"] == (deteccao["y1"] + deteccao["y2"]) // 2


def test_deteccoes_saem_ordenadas_por_confianca(detector):
    imagem = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

    scores = [d["score"] for d in detector.detectar(imagem)]

    assert scores == sorted(scores, reverse=True)


def test_limiar_filtra_deteccoes(checkpoint):
    imagem = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

    permissivo = DetectorEquipamentos(checkpoint, limiar=0.0).detectar(imagem)
    restritivo = DetectorEquipamentos(checkpoint, limiar=0.99).detectar(imagem)

    assert len(restritivo) <= len(permissivo)


def test_entrada_que_nao_e_imagem_e_recusada(detector):
    with pytest.raises(ValueError, match="3 dimensões"):
        detector.detectar(np.zeros((10, 10), dtype=np.uint8))


def test_checkpoint_inexistente_da_erro_claro(tmp_path):
    with pytest.raises(CheckpointInvalidoError, match="não encontrado"):
        DetectorEquipamentos(tmp_path / "nao_existe.pt")


def test_obter_detector_reaproveita_a_instancia(checkpoint, monkeypatch):
    """Carregar ~170 MB de pesos a cada requisição inviabilizaria a API."""
    monkeypatch.setenv("DETECTOR_CHECKPOINT_PATH", str(checkpoint))
    redefinir_detector()

    assert obter_detector() is obter_detector()


def test_nms_entre_classes_remove_caixa_duplicada(detector):
    """
    O Faster R-CNN aplica NMS por classe, então o mesmo símbolo pode sair
    duas vezes com rótulos diferentes. Aqui vence a de maior confiança.
    """
    boxes = torch.tensor(
        [
            [10.0, 10.0, 100.0, 100.0],
            [12.0, 12.0, 102.0, 102.0],  # mesma caixa, outra classe
        ]
    )
    scores = torch.tensor([0.9, 0.7])
    labels = torch.tensor([1, 2])

    boxes, scores, labels = detector._nms_entre_classes(boxes, scores, labels)

    assert len(boxes) == 1
    assert float(scores[0]) == pytest.approx(0.9)
    assert int(labels[0]) == 1


def test_nms_entre_classes_preserva_simbolos_vizinhos(detector):
    """
    Válvulas empilhadas numa mesma linha de tubulação são caixas
    distintas e próximas — não podem ser fundidas.
    """
    boxes = torch.tensor(
        [
            [10.0, 10.0, 100.0, 60.0],
            [10.0, 70.0, 100.0, 120.0],
        ]
    )
    scores = torch.tensor([0.9, 0.8])
    labels = torch.tensor([1, 1])

    boxes, _, _ = detector._nms_entre_classes(boxes, scores, labels)

    assert len(boxes) == 2


def test_nms_entre_classes_pode_ser_desligado(checkpoint):
    sem_nms = DetectorEquipamentos(
        checkpoint, limiar=0.0, nms_entre_classes=0
    )

    boxes = torch.tensor(
        [[10.0, 10.0, 100.0, 100.0], [12.0, 12.0, 102.0, 102.0]]
    )
    scores = torch.tensor([0.9, 0.7])
    labels = torch.tensor([1, 2])

    boxes, _, _ = sem_nms._nms_entre_classes(boxes, scores, labels)

    assert len(boxes) == 2


def test_nms_entre_classes_com_zero_deteccoes(detector):
    vazio_boxes = torch.zeros((0, 4))
    vazio = torch.zeros((0,))

    boxes, _, _ = detector._nms_entre_classes(vazio_boxes, vazio, vazio)

    assert len(boxes) == 0


def test_detector_usa_o_teto_de_deteccoes_configurado(checkpoint):
    from app import config

    detector = DetectorEquipamentos(checkpoint, limiar=0.0)

    assert detector.modelo.roi_heads.detections_per_img == (
        config.DETECTOR_MAX_DETECCOES
    )


def test_limiar_padrao_privilegia_cobertura():
    """
    Política do projeto: detectar o máximo possível, mesmo símbolo mal
    desenhado. O padrão foi medido sobre as 21 imagens anotadas — a 0.05
    a revocação é 0.366 contra 0.287 a 0.5, com F1 ligeiramente melhor.
    """
    from app import config

    assert config.DEFAULT_DETECTOR_SCORE_THRESHOLD <= 0.05


def test_piso_do_modelo_acompanha_um_limiar_mais_baixo(checkpoint):
    """
    O torchvision descarta internamente abaixo de box_score_thresh, antes
    de a predição chegar ao nosso código. Pedir um limiar de 0.01 sem
    baixar o piso junto não devolveria detecção nenhuma abaixo de 0.05 —
    o filtro externo não recupera o que o modelo já jogou fora.
    """
    detector = DetectorEquipamentos(checkpoint, limiar=0.01)

    assert detector.modelo.roi_heads.score_thresh <= 0.01


def test_piso_do_modelo_nao_sobe_com_limiar_alto(checkpoint):
    """Limiar alto filtra depois; o piso não precisa subir junto."""
    from app import config

    detector = DetectorEquipamentos(checkpoint, limiar=0.9)

    assert detector.modelo.roi_heads.score_thresh == (
        config.DETECTOR_SCORE_MINIMO_MODELO
    )


def test_nms_entre_classes_continua_ligado_por_padrao():
    """
    Medido: desligá-lo acrescenta 654 predições sobre as 21 imagens de
    teste e ganha 2 acertos — a precisão cai de 0.390 para 0.213. O que
    ele remove é duplicata quase pura, não cobertura.
    """
    from app import config

    assert config.DETECTOR_NMS_ENTRE_CLASSES > 0
