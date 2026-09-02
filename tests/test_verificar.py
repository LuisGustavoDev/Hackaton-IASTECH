"""
CLI de verificação do detector na máquina de inferência.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app.core.erros import CheckpointInvalidoError  # noqa: E402
from app.detection.checkpoint import salvar_checkpoint  # noqa: E402
from app.detection.verificar import (  # noqa: E402
    _sem_acento,
    inspecionar,
    salvar_anotada,
)

CLASSES = ["Bomba", "Conexão", "Válvula"]


@pytest.fixture
def checkpoint(tmp_path):
    return salvar_checkpoint(
        tmp_path / "modelo.pt",
        torch.nn.Linear(4, 2),
        CLASSES,
        metadados={"epocas": 50, "mAP@.5": 0.42},
    )


def test_inspecionar_mostra_classes_com_o_label_de_cada_uma(
    checkpoint, capsys
):
    inspecionar(checkpoint)

    saida = capsys.readouterr().out

    assert "label  1 -> Bomba" in saida
    assert "label  3 -> Válvula" in saida


def test_inspecionar_mostra_os_metadados_do_treino(checkpoint, capsys):
    inspecionar(checkpoint)

    saida = capsys.readouterr().out

    assert "epocas" in saida
    assert "50" in saida
    assert "0.42" in saida


def test_inspecionar_devolve_o_checkpoint(checkpoint):
    assert inspecionar(checkpoint)["classes"] == CLASSES


def test_inspecionar_recusa_checkpoint_invalido(tmp_path):
    caminho = tmp_path / "lixo.pt"
    caminho.write_bytes(b"nao e um checkpoint")

    with pytest.raises(CheckpointInvalidoError):
        inspecionar(caminho)


@pytest.mark.parametrize(
    "entrada, esperado",
    [
        ("Válvula", "Valvula"),
        ("Conexão", "Conexao"),
        ("SistemadePotência", "SistemadePotencia"),
        ("Bomba", "Bomba"),
    ],
)
def test_sem_acento_prepara_o_texto_para_o_opencv(entrada, esperado):
    """As fontes do OpenCV não desenham acentuação."""
    assert _sem_acento(entrada) == esperado


def test_salvar_anotada_grava_a_imagem(tmp_path):
    pytest.importorskip("cv2")

    array = np.zeros((100, 120, 3), dtype=np.uint8)

    deteccoes = [
        {
            "classe": "Válvula",
            "score": 0.87,
            "x1": 10,
            "y1": 10,
            "x2": 50,
            "y2": 40,
            "centro_x": 30,
            "centro_y": 25,
        }
    ]

    destino = salvar_anotada(array, deteccoes, tmp_path / "sub" / "saida.png")

    assert destino.is_file()
    # A original não é alterada.
    assert array.sum() == 0


def test_salvar_anotada_sem_deteccoes_ainda_grava(tmp_path):
    pytest.importorskip("cv2")

    array = np.zeros((50, 50, 3), dtype=np.uint8)

    assert salvar_anotada(array, [], tmp_path / "vazia.png").is_file()
