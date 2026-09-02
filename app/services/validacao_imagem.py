"""
Validação do arquivo enviado pelo usuário.

O content-type do upload é informado pelo cliente e não prova nada: um PDF
renomeado para .png chega com "image/png". Aqui a decisão é tomada pelo
CONTEÚDO do arquivo — se o Pillow não conseguir decodificar, não é imagem.
"""

from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from app import config
from app.core.erros import ImagemInvalidaError


def validar_imagem(conteudo: bytes) -> str:
    """
    Confere que os bytes recebidos são uma imagem íntegra e suportada.

    Retorna o formato detectado (ex.: "PNG"). Levanta ImagemInvalidaError
    com uma mensagem legível em qualquer outro caso.
    """
    if not conteudo:
        raise ImagemInvalidaError("O arquivo enviado está vazio.")

    if len(conteudo) > config.IMAGEM_TAMANHO_MAXIMO_BYTES:
        limite_mb = config.IMAGEM_TAMANHO_MAXIMO_BYTES / (1024 * 1024)
        tamanho_mb = len(conteudo) / (1024 * 1024)
        raise ImagemInvalidaError(
            f"O arquivo tem {tamanho_mb:.1f} MB e o limite é "
            f"{limite_mb:.0f} MB."
        )

    # verify() detecta arquivo truncado/corrompido, mas invalida o objeto:
    # depois dele é obrigatório reabrir para conseguir ler os pixels.
    try:
        with Image.open(BytesIO(conteudo)) as imagem:
            imagem.verify()
    except UnidentifiedImageError as erro:
        raise ImagemInvalidaError(
            "O arquivo enviado não é uma imagem reconhecível. "
            "Formatos aceitos: "
            f"{', '.join(config.IMAGEM_FORMATOS_ACEITOS)}."
        ) from erro
    except Exception as erro:
        raise ImagemInvalidaError(
            f"A imagem enviada está corrompida ou incompleta: {erro}"
        ) from erro

    try:
        with Image.open(BytesIO(conteudo)) as imagem:
            formato = imagem.format
            largura, altura = imagem.size
            # Força a decodificação real dos pixels: há arquivos que passam
            # pelo verify() (cabeçalho íntegro) e só quebram aqui.
            imagem.load()
    except Exception as erro:
        raise ImagemInvalidaError(
            f"A imagem enviada está corrompida ou incompleta: {erro}"
        ) from erro

    if formato not in config.IMAGEM_FORMATOS_ACEITOS:
        raise ImagemInvalidaError(
            f"Formato de imagem não suportado: {formato}. "
            f"Formatos aceitos: "
            f"{', '.join(config.IMAGEM_FORMATOS_ACEITOS)}."
        )

    menor = min(largura, altura)
    maior = max(largura, altura)

    if menor < config.IMAGEM_DIMENSAO_MINIMA:
        raise ImagemInvalidaError(
            f"A imagem é pequena demais ({largura}x{altura} px). "
            f"O mínimo é {config.IMAGEM_DIMENSAO_MINIMA} px por lado."
        )

    if maior > config.IMAGEM_DIMENSAO_MAXIMA:
        raise ImagemInvalidaError(
            f"A imagem é grande demais ({largura}x{altura} px). "
            f"O máximo é {config.IMAGEM_DIMENSAO_MAXIMA} px por lado."
        )

    return formato


def decodificar_imagem(conteudo: bytes) -> np.ndarray:
    """
    Valida e devolve a imagem como array BGR — o formato que o restante do
    pipeline (cv2) já usa.
    """
    validar_imagem(conteudo)

    imagem = cv2.imdecode(
        np.frombuffer(conteudo, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    if imagem is None:
        # O Pillow aceitou mas o OpenCV não conseguiu decodificar. Acontece
        # com formatos que o cv2 não compila por padrão (certos TIFF/WEBP).
        raise ImagemInvalidaError(
            "A imagem foi reconhecida mas não pôde ser decodificada para "
            "processamento. Converta para PNG ou JPEG e tente novamente."
        )

    return imagem
