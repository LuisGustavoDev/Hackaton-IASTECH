"""
Contrato HTTP da rota de processamento.

O foco aqui é a tradução de cada erro de domínio no status code certo — a
detecção propriamente dita é testada em test_detector.py e
test_integracao_pipeline.py.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.erros import (
    CheckpointInvalidoError,
    ImagemInvalidaError,
    ProcessamentoError,
)
from app.main import app
from tests.conftest import bytes_de_imagem


@pytest.fixture
def cliente():
    return TestClient(app)


def test_health_responde_ok(cliente):
    resposta = cliente.get("/api/health")

    assert resposta.status_code == 200
    assert resposta.json()["status"] == "ok"


def test_arquivo_que_nao_e_imagem_responde_400(cliente):
    resposta = cliente.post(
        "/api/process",
        files={"file": ("diagrama.png", b"%PDF-1.7\n%fake", "image/png")},
    )

    assert resposta.status_code == 400
    assert "não é uma imagem" in resposta.json()["detail"]


def test_arquivo_vazio_responde_400(cliente):
    resposta = cliente.post(
        "/api/process",
        files={"file": ("vazio.png", b"", "image/png")},
    )

    assert resposta.status_code == 400
    assert "vazio" in resposta.json()["detail"]


def test_imagem_valida_sem_checkpoint_responde_503(cliente, monkeypatch):
    """
    Sem modelo instalado, o sistema não pode trabalhar — mas o problema é
    de instalação, não do arquivo enviado. 503, não 400 nem 500.
    """
    monkeypatch.setenv("DETECTOR_CHECKPOINT_PATH", "nao/existe/modelo.pt")

    resposta = cliente.post(
        "/api/process",
        files={"file": ("d.png", bytes_de_imagem(), "image/png")},
    )

    assert resposta.status_code == 503
    assert "Checkpoint" in resposta.json()["detail"]


def test_falha_no_processamento_responde_500(cliente, monkeypatch):
    def explodir(_):
        raise ProcessamentoError("falha simulada no OCR")

    monkeypatch.setattr(
        "app.api.routes.processar_imagem", explodir
    )

    resposta = cliente.post(
        "/api/process",
        files={"file": ("d.png", bytes_de_imagem(), "image/png")},
    )

    assert resposta.status_code == 500
    assert "falha simulada" in resposta.json()["detail"]


def test_planilha_e_devolvida_como_xlsx(cliente, monkeypatch, tmp_path):
    from app.services.planilha_service import gerar_planilha
    from app.services.processamento import ResultadoProcessamento

    xlsx, arquivo_csv = gerar_planilha(
        [
            {
                "TAG": "FT-210",
                "Tipo": "Instrumento",
                "Descrição": "Transmissor de Vazão",
                "Coordenada X": 10,
                "Coordenada Y": 20,
                "Grupo": "2",
            }
        ],
        tmp_path / "r.xlsx",
        tmp_path / "r.csv",
    )

    monkeypatch.setattr(
        "app.api.routes.processar_imagem",
        lambda _: ResultadoProcessamento(
            linhas=[],
            planilha_xlsx=xlsx,
            planilha_csv=arquivo_csv,
            imagem=tmp_path / "entrada.png",
        ),
    )

    resposta = cliente.post(
        "/api/process",
        files={"file": ("d.png", bytes_de_imagem(), "image/png")},
    )

    assert resposta.status_code == 200
    assert resposta.headers["content-type"] == (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    # Assinatura de arquivo .xlsx (zip).
    assert resposta.content[:2] == b"PK"


def test_upload_sem_arquivo_responde_422(cliente):
    assert cliente.post("/api/process").status_code == 422


def test_erro_de_dominio_nao_vaza_como_500(cliente, monkeypatch):
    def explodir(_):
        raise ImagemInvalidaError("imagem estranha")

    monkeypatch.setattr("app.api.routes.processar_imagem", explodir)

    resposta = cliente.post(
        "/api/process",
        files={"file": ("d.png", bytes_de_imagem(), "image/png")},
    )

    assert resposta.status_code == 400


def test_checkpoint_invalido_no_meio_do_processo_responde_503(
    cliente, monkeypatch
):
    def explodir(_):
        raise CheckpointInvalidoError("checkpoint corrompido")

    monkeypatch.setattr("app.api.routes.processar_imagem", explodir)

    resposta = cliente.post(
        "/api/process",
        files={"file": ("d.png", bytes_de_imagem(), "image/png")},
    )

    assert resposta.status_code == 503
