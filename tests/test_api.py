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
    def explodir(_conteudo, **_kwargs):
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
        lambda _conteudo, **_kwargs: ResultadoProcessamento(
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
    def explodir(_conteudo, **_kwargs):
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
    def explodir(_conteudo, **_kwargs):
        raise CheckpointInvalidoError("checkpoint corrompido")

    monkeypatch.setattr("app.api.routes.processar_imagem", explodir)

    resposta = cliente.post(
        "/api/process",
        files={"file": ("d.png", bytes_de_imagem(), "image/png")},
    )

    assert resposta.status_code == 503


def test_rota_de_processamento_nao_e_async():
    """
    A inferência do Faster R-CNN gasta segundos de CPU. Num `async def`
    ela rodaria dentro do event loop e travaria o servidor inteiro:
    requisições simultâneas serializariam e até o /api/health ficaria
    pendurado. Como `def`, o FastAPI executa no threadpool.

    Verificado por inspeção da assinatura, e não por um teste de
    concorrência: o TestClient isola cada requisição no seu próprio
    portal, então um teste de "o /health responde durante o
    processamento?" passa igual com a rota async e daria falsa
    segurança.
    """
    import inspect

    from app.api.routes import process

    assert not inspect.iscoroutinefunction(process)


def test_nome_do_arquivo_enviado_chega_ao_processamento(cliente, monkeypatch):
    """O nome vai para a tabela `execucoes`, para rastrear qual diagrama
    gerou cada rodada."""
    recebido = {}

    def capturar(_conteudo, **kwargs):
        recebido.update(kwargs)
        raise ProcessamentoError("parou aqui")

    monkeypatch.setattr("app.api.routes.processar_imagem", capturar)

    cliente.post(
        "/api/process",
        files={"file": ("diagrama-101.jpg", bytes_de_imagem(), "image/jpeg")},
    )

    assert recebido["arquivo_nome"] == "diagrama-101.jpg"


# ---------------------------------------------------------------------
# Consumo pelo frontend
# ---------------------------------------------------------------------


def test_cors_liberado_para_o_navegador(cliente):
    """
    Sem CORS o navegador bloqueia a chamada antes de ela sair — e o curl
    não passa por isso, então o problema só aparece quando a interface
    entra em cena.
    """
    resposta = cliente.options(
        "/api/process",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert resposta.status_code in (200, 204)
    assert resposta.headers["access-control-allow-origin"] in (
        "*",
        "http://localhost:3000",
    )


def test_content_disposition_e_exposto_ao_javascript(cliente, monkeypatch, tmp_path):
    """
    Por padrão o JS só enxerga alguns cabeçalhos da resposta, e
    Content-Disposition não está entre eles: sem expô-lo, o frontend
    recebe o arquivo mas não consegue ler o nome sugerido.
    """
    _stub_resultado(monkeypatch, tmp_path)

    resposta = cliente.post(
        "/api/process",
        files={"file": ("d.png", bytes_de_imagem(), "image/png")},
        headers={"Origin": "http://localhost:3000"},
    )

    expostos = resposta.headers.get("access-control-expose-headers", "")

    assert "Content-Disposition" in expostos
    assert "resultado.xlsx" in resposta.headers["content-disposition"]


def test_formato_json_devolve_as_linhas_para_previsualizacao(
    cliente, monkeypatch, tmp_path
):
    """
    A interface precisa mostrar a quantidade e uma prévia da lista antes
    de o usuário baixar o Excel; um corpo binário não serve para isso.
    """
    _stub_resultado(monkeypatch, tmp_path, execucao_id=7)

    resposta = cliente.post(
        "/api/process?formato=json",
        files={"file": ("d.png", bytes_de_imagem(), "image/png")},
    )

    corpo = resposta.json()

    assert resposta.status_code == 200
    assert corpo["execucao_id"] == 7
    assert corpo["quantidade"] == 1
    assert corpo["linhas"][0]["TAG"] == "FT-210"
    assert corpo["download"]["xlsx"] == "/api/resultado/7?formato=xlsx"


def test_formato_csv_devolve_o_csv(cliente, monkeypatch, tmp_path):
    _stub_resultado(monkeypatch, tmp_path)

    resposta = cliente.post(
        "/api/process?formato=csv",
        files={"file": ("d.png", bytes_de_imagem(), "image/png")},
    )

    assert resposta.status_code == 200
    assert resposta.headers["content-type"].startswith("text/csv")


def test_formato_invalido_e_recusado(cliente):
    resposta = cliente.post(
        "/api/process?formato=pdf",
        files={"file": ("d.png", bytes_de_imagem(), "image/png")},
    )

    assert resposta.status_code == 422


def test_json_sem_execucao_gravada_nao_inventa_link(
    cliente, monkeypatch, tmp_path
):
    """
    A persistência é best-effort. Se ela falhou, não há de onde baixar
    depois — o campo vem nulo em vez de um link que daria 404.
    """
    _stub_resultado(monkeypatch, tmp_path, execucao_id=None)

    resposta = cliente.post(
        "/api/process?formato=json",
        files={"file": ("d.png", bytes_de_imagem(), "image/png")},
    )

    assert resposta.json()["download"] is None


def test_download_de_execucao_inexistente_responde_404(cliente):
    assert cliente.get("/api/resultado/999999").status_code == 404


def _stub_resultado(monkeypatch, tmp_path, execucao_id=1):
    from app.services.planilha_service import gerar_planilha
    from app.services.processamento import ResultadoProcessamento

    linhas = [
        {
            "TAG": "FT-210",
            "Tipo": "Instrumento",
            "Descrição": "Transmissor de Vazão",
            "Coordenada X": 10,
            "Coordenada Y": 20,
            "Grupo": "2",
        }
    ]

    xlsx, arquivo_csv = gerar_planilha(
        linhas, tmp_path / "resultado.xlsx", tmp_path / "resultado.csv"
    )

    monkeypatch.setattr(
        "app.api.routes.processar_imagem",
        lambda _conteudo, **_kwargs: ResultadoProcessamento(
            linhas=linhas,
            planilha_xlsx=xlsx,
            planilha_csv=arquivo_csv,
            imagem=tmp_path / "entrada.png",
            execucao_id=execucao_id,
        ),
    )
