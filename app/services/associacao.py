"""
Associação entre os textos lidos pelo OCR e os equipamentos detectados.

O Faster R-CNN diz ONDE está cada equipamento e QUE tipo ele é; o OCR diz
que textos existem e onde. A TAG de um equipamento é o texto que "pertence"
à sua bounding box.

Estratégia, em duas passadas:

1. Textos cujo centro cai DENTRO da bounding box — é o caso da esmagadora
   maioria dos instrumentos, que trazem a tag escrita dentro do círculo.

2. Para os equipamentos que sobraram, o texto mais próximo dentro de um
   raio proporcional ao tamanho da própria caixa. O raio é relativo, e não
   fixo em pixels, para que um símbolo pequeno não "puxe" a tag de um
   vizinho e um símbolo grande não fique sem tag em diagramas de alta
   resolução.

Cada texto é usado por no máximo um equipamento: as atribuições são
avaliadas da mais próxima para a mais distante e a primeira que aparecer
para um par (equipamento, texto) fica valendo.
"""

from __future__ import annotations

import math

from app import config


def associar(
    deteccoes: list[dict],
    textos: list[dict],
    raio_relativo: float | None = None,
) -> list[dict]:
    """
    Devolve uma cópia de `deteccoes`, cada uma acrescida de:

        "tag":           texto associado ("" se nenhum)
        "confianca_ocr": confiança do OCR desse texto (0.0 se nenhum)

    `textos` são os dicionários produzidos por
    app/services/image_service.py: {"texto", "confianca", "x", "y", "w", "h"}.
    """
    if raio_relativo is None:
        raio_relativo = config.ASSOCIACAO_RAIO_RELATIVO

    resultado = [
        {**deteccao, "tag": "", "confianca_ocr": 0.0}
        for deteccao in deteccoes
    ]

    candidatos = [
        texto
        for texto in textos
        if (texto.get("texto") or "").strip()
    ]

    usados: set[int] = set()

    _atribuir(
        resultado,
        candidatos,
        usados,
        somente_internos=True,
        raio_relativo=raio_relativo,
    )

    _atribuir(
        resultado,
        candidatos,
        usados,
        somente_internos=False,
        raio_relativo=raio_relativo,
    )

    return resultado


def _atribuir(
    deteccoes: list[dict],
    textos: list[dict],
    usados: set[int],
    somente_internos: bool,
    raio_relativo: float,
) -> None:
    pares = []

    for indice_deteccao, deteccao in enumerate(deteccoes):
        if deteccao["tag"]:
            continue

        raio = _raio(deteccao, raio_relativo)

        for indice_texto, texto in enumerate(textos):
            if indice_texto in usados:
                continue

            centro_x, centro_y = _centro(texto)

            dentro = _dentro_da_caixa(deteccao, centro_x, centro_y)

            if somente_internos and not dentro:
                continue

            distancia = math.dist(
                (centro_x, centro_y),
                (deteccao["centro_x"], deteccao["centro_y"]),
            )

            if not somente_internos and distancia > raio:
                continue

            pares.append((distancia, indice_deteccao, indice_texto))

    pares.sort()

    atribuidos: set[int] = set()

    for _, indice_deteccao, indice_texto in pares:
        if indice_deteccao in atribuidos or indice_texto in usados:
            continue

        texto = textos[indice_texto]

        deteccoes[indice_deteccao]["tag"] = texto["texto"].strip()
        deteccoes[indice_deteccao]["confianca_ocr"] = float(
            texto.get("confianca", 0.0)
        )

        atribuidos.add(indice_deteccao)
        usados.add(indice_texto)


def _centro(texto: dict) -> tuple[float, float]:
    return (
        texto["x"] + texto["w"] / 2,
        texto["y"] + texto["h"] / 2,
    )


def _dentro_da_caixa(deteccao: dict, x: float, y: float) -> bool:
    return (
        deteccao["x1"] <= x <= deteccao["x2"]
        and deteccao["y1"] <= y <= deteccao["y2"]
    )


def _raio(deteccao: dict, raio_relativo: float) -> float:
    largura = deteccao["x2"] - deteccao["x1"]
    altura = deteccao["y2"] - deteccao["y1"]

    return raio_relativo * math.hypot(largura, altura)
