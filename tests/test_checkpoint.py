"""
Checkpoint portátil.

O requisito é que o arquivo salvo no desktop com GPU baste, sozinho, para
rodar a inferência no notebook: nada de dataset, nada de json de anotação,
nada de CUDA.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from app.core.erros import CheckpointInvalidoError  # noqa: E402
from app.detection.checkpoint import (  # noqa: E402
    FORMATO_VERSAO,
    carregar_checkpoint,
    salvar_checkpoint,
)

CLASSES = ["Bomba", "Conexão", "Válvula"]


@pytest.fixture
def modelo_falso():
    """Um módulo qualquer: o formato do checkpoint não depende da arquitetura."""
    return torch.nn.Linear(4, 2)


@pytest.fixture
def checkpoint(tmp_path, modelo_falso):
    return salvar_checkpoint(
        tmp_path / "modelo.pt",
        modelo_falso,
        CLASSES,
        metadados={"epocas": 15},
    )


def test_checkpoint_carrega_o_que_foi_salvo(checkpoint):
    dados = carregar_checkpoint(checkpoint)

    assert dados["formato_versao"] == FORMATO_VERSAO
    assert dados["arquitetura"] == "fasterrcnn_resnet50_fpn_v2"
    assert dados["classes"] == CLASSES
    assert dados["num_classes"] == 3
    assert dados["metadados"]["epocas"] == 15
    assert "weight" in dados["state_dict"]


def test_classes_com_acento_sobrevivem(checkpoint):
    assert "Conexão" in carregar_checkpoint(checkpoint)["classes"]


def test_ordem_das_classes_e_preservada(checkpoint):
    """A ordem É o rótulo: trocá-la troca as classes das predições."""
    assert carregar_checkpoint(checkpoint)["classes"] == CLASSES


def test_pesos_ficam_em_cpu(checkpoint):
    for tensor in carregar_checkpoint(checkpoint)["state_dict"].values():
        assert tensor.device.type == "cpu"


def test_lista_de_classes_vazia_e_recusada(tmp_path, modelo_falso):
    with pytest.raises(ValueError, match="classes"):
        salvar_checkpoint(tmp_path / "x.pt", modelo_falso, [])


def test_arquivo_inexistente_da_erro_claro(tmp_path):
    with pytest.raises(CheckpointInvalidoError, match="não encontrado"):
        carregar_checkpoint(tmp_path / "nao_existe.pt")


def test_state_dict_puro_do_benchmark_e_recusado(tmp_path, modelo_falso):
    """
    O benchmark salva torch.save(model.state_dict(), path) — sem as
    classes. Carregar isso em produção daria classe trocada; o erro precisa
    apontar o caminho da conversão.
    """
    caminho = tmp_path / "benchmark.pt"
    torch.save(modelo_falso.state_dict(), caminho)

    with pytest.raises(CheckpointInvalidoError, match="empacotar_checkpoint"):
        carregar_checkpoint(caminho)


def test_formato_de_versao_desconhecida_e_recusado(tmp_path, checkpoint):
    dados = torch.load(checkpoint, map_location="cpu", weights_only=True)
    dados["formato_versao"] = 99

    caminho = tmp_path / "futuro.pt"
    torch.save(dados, caminho)

    with pytest.raises(CheckpointInvalidoError, match="formato versão"):
        carregar_checkpoint(caminho)


def test_arquitetura_diferente_e_recusada(tmp_path, checkpoint):
    dados = torch.load(checkpoint, map_location="cpu", weights_only=True)
    dados["arquitetura"] = "retinanet_resnet50_fpn_v2"

    caminho = tmp_path / "retinanet.pt"
    torch.save(dados, caminho)

    with pytest.raises(CheckpointInvalidoError, match="retinanet"):
        carregar_checkpoint(caminho)


def test_num_classes_inconsistente_e_recusado(tmp_path, checkpoint):
    dados = torch.load(checkpoint, map_location="cpu", weights_only=True)
    dados["num_classes"] = 99

    caminho = tmp_path / "inconsistente.pt"
    torch.save(dados, caminho)

    with pytest.raises(CheckpointInvalidoError, match="inconsistente"):
        carregar_checkpoint(caminho)


def test_arquivo_que_nao_e_checkpoint_e_recusado(tmp_path):
    caminho = tmp_path / "lixo.pt"
    caminho.write_bytes(b"isto nao e um checkpoint")

    with pytest.raises(CheckpointInvalidoError):
        carregar_checkpoint(caminho)


def test_metadados_nao_primitivos_nao_quebram_a_leitura(tmp_path, modelo_falso):
    """
    Regressão: o treino grava torch.__version__ nos metadados, que é um
    TorchVersion e não uma str. Com weights_only=True o torch.load recusa
    o arquivo INTEIRO por causa desse único valor — o checkpoint era salvo
    com sucesso e só falhava na máquina de inferência.
    """
    caminho = salvar_checkpoint(
        tmp_path / "modelo.pt",
        modelo_falso,
        CLASSES,
        metadados={
            "torch_version": torch.__version__,
            "device_treino": torch.device("cpu"),
            "aninhado": {"caminho": tmp_path},
            "lista": [torch.__version__],
        },
    )

    metadados = carregar_checkpoint(caminho)["metadados"]

    assert metadados["torch_version"] == str(torch.__version__)
    assert metadados["device_treino"] == "cpu"
    assert isinstance(metadados["aninhado"]["caminho"], str)
    assert isinstance(metadados["lista"][0], str)


def test_metadados_primitivos_mantem_o_tipo(tmp_path, modelo_falso):
    caminho = salvar_checkpoint(
        tmp_path / "modelo.pt",
        modelo_falso,
        CLASSES,
        metadados={"epocas": 15, "lr": 0.005, "avaliado": True, "obs": None},
    )

    metadados = carregar_checkpoint(caminho)["metadados"]

    assert metadados == {
        "epocas": 15,
        "lr": 0.005,
        "avaliado": True,
        "obs": None,
    }
