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
