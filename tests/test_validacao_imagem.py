"""
Validação do arquivo enviado.

O critério é o CONTEÚDO do arquivo, não o content-type nem a extensão.
"""

from __future__ import annotations

import pytest

from app.core.erros import ImagemInvalidaError
from app.services.validacao_imagem import decodificar_imagem, validar_imagem
from tests.conftest import bytes_de_imagem


def test_png_valido_e_aceito(png_valido):
    assert validar_imagem(png_valido) == "PNG"


def test_jpeg_valido_e_aceito():
    assert validar_imagem(bytes_de_imagem(formato="JPEG")) == "JPEG"


def test_arquivo_vazio_e_rejeitado():
    with pytest.raises(ImagemInvalidaError, match="vazio"):
        validar_imagem(b"")


def test_arquivo_que_nao_e_imagem_e_rejeitado():
    # Cabeçalho de PDF: é exatamente o caso de "renomeei para .png".
    conteudo = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<< >>\nendobj\n"

    with pytest.raises(ImagemInvalidaError, match="não é uma imagem"):
        validar_imagem(conteudo)


def test_texto_puro_e_rejeitado():
    with pytest.raises(ImagemInvalidaError):
        validar_imagem(b"isto nao e uma imagem, e um texto qualquer")


def test_png_truncado_e_rejeitado():
    conteudo = bytes_de_imagem(largura=256, altura=256)

    # Mantém o cabeçalho e corta o resto: o arquivo "parece" um PNG mas
    # não tem os dados de pixel completos.
    with pytest.raises(ImagemInvalidaError, match="corrompida|incompleta"):
        validar_imagem(conteudo[: len(conteudo) // 2])


def test_imagem_pequena_demais_e_rejeitada():
    with pytest.raises(ImagemInvalidaError, match="pequena"):
        validar_imagem(bytes_de_imagem(largura=4, altura=4))


def test_arquivo_grande_demais_e_rejeitado(monkeypatch):
    from app import config

    monkeypatch.setattr(config, "IMAGEM_TAMANHO_MAXIMO_BYTES", 100)

    with pytest.raises(ImagemInvalidaError, match="limite"):
        validar_imagem(bytes_de_imagem(largura=256, altura=256))


def test_decodificar_devolve_array_bgr(png_valido):
    array = decodificar_imagem(png_valido)

    assert array.shape == (64, 64, 3)
    assert array.dtype.name == "uint8"


def test_decodificar_rejeita_arquivo_invalido():
    with pytest.raises(ImagemInvalidaError):
        decodificar_imagem(b"nada disso")
