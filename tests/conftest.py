"""
Fixtures compartilhadas pelos testes.

Rodar:  pytest
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent

# Duas imagens reais do dataset, usadas nos testes de integração.
IMAGENS_REAIS = [
    RAIZ / "dataset/original/testes/diagramas/qtd_baixa/qld_baixa/113.jpg",
    RAIZ / "dataset/original/testes/diagramas/qtd_baixa/qld_alta/101.jpg",
]


def bytes_de_imagem(
    largura: int = 64,
    altura: int = 64,
    formato: str = "PNG",
    cor: str = "white",
) -> bytes:
    """Gera uma imagem válida em memória."""
    buffer = io.BytesIO()
    Image.new("RGB", (largura, altura), cor).save(buffer, format=formato)
    return buffer.getvalue()


@pytest.fixture
def png_valido() -> bytes:
    return bytes_de_imagem()


@pytest.fixture
def imagem_real() -> Path:
    for caminho in IMAGENS_REAIS:
        if caminho.is_file():
            return caminho

    pytest.skip(
        "Nenhuma imagem do dataset encontrada — o dataset não está "
        "presente nesta cópia do repositório."
    )


@pytest.fixture
def tesseract_disponivel() -> None:
    """Pula o teste quando o binário do Tesseract não está instalado."""
    pytesseract = pytest.importorskip("pytesseract")

    try:
        pytesseract.get_tesseract_version()
    except Exception as erro:
        pytest.skip(f"Tesseract não disponível nesta máquina: {erro}")


@pytest.fixture(autouse=True)
def _limpar_singletons():
    """
    Zera os caches de processo entre testes.

    O detector e o dicionário ISA são singletons (mesmo padrão do get_db);
    sem isso, um teste que carrega um checkpoint temporário contaminaria o
    seguinte.
    """
    yield

    try:
        from app.detection.detector import redefinir_detector

        redefinir_detector()
    except ImportError:
        pass

    from app.services.tag_service import redefinir_dicionario_isa

    redefinir_dicionario_isa()
