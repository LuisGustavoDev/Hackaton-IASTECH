"""
Associação entre os textos lidos pelo OCR e os equipamentos detectados.

O Faster R-CNN diz ONDE está cada equipamento e QUE tipo ele é; o OCR diz
que textos existem e onde. A TAG de um equipamento é o texto que
"pertence" à sua bounding box.

Estratégia, em duas passadas:

1. **Todos** os textos cujo centro cai DENTRO da bounding box, juntados na
   ordem de leitura (de cima para baixo, da esquerda para a direita). Um
   balão de instrumento ISA traz as letras numa linha e o número na
   outra — "PI" em cima, "0013" embaixo. Pegar só um dos dois destrói a
   TAG: "0013" sozinho não permite deduzir nem a Descrição nem o Grupo.

2. Para os equipamentos que sobraram sem nenhum texto interno, o texto
   mais próximo dentro de um raio proporcional ao tamanho da própria
   caixa — e apenas UM. Fora da caixa não há garantia de que dois textos
   vizinhos pertençam ao mesmo equipamento, então juntá-los seria chute.

O raio é relativo, e não fixo em pixels, para que um símbolo pequeno não
"puxe" a tag de um vizinho e um símbolo grande não fique sem tag em
diagramas de alta resolução.

Cada texto serve a no máximo um equipamento. Quando um texto cai dentro de
mais de uma caixa (símbolo dentro de símbolo), fica com aquela de centro
mais próximo.
"""

from __future__ import annotations

import math

from app import config


def associar(
    deteccoes: list[dict],
    textos: list[dict],
    raio_relativo: float | None = None,
    confianca_minima: float | None = None,
) -> list[dict]:
    """
    Devolve uma cópia de `deteccoes`, cada uma acrescida de:

        "tag":           texto associado ("" se nenhum)
        "confianca_ocr": confiança média dos textos usados (0.0 se nenhum)

    `textos` são os dicionários produzidos por
    app/services/image_service.py: {"texto", "confianca", "x", "y", "w", "h"}.
    """
    if raio_relativo is None:
        raio_relativo = config.ASSOCIACAO_RAIO_RELATIVO

    if confianca_minima is None:
        confianca_minima = config.OCR_CONFIANCA_MINIMA

    resultado = [
        {**deteccao, "tag": "", "confianca_ocr": 0.0}
        for deteccao in deteccoes
    ]

    candidatos = _filtrar(textos, confianca_minima)

    usados: set[int] = set()

    _atribuir_internos(resultado, candidatos, usados)
    _atribuir_vizinho(resultado, candidatos, usados, raio_relativo)

    return resultado


def _filtrar(textos: list[dict], confianca_minima: float) -> list[dict]:
    """
    Descarta o que não tem chance de ser TAG.

    O MSER encontra traços do próprio desenho do símbolo (o "X" de uma
    válvula, o círculo de um balão) e o Tesseract devolve letras a partir
    deles — "(X)", "Oo", "SH". Esses fragmentos vêm com confiança baixa, e
    barrá-los aqui evita que sejam colados no meio de uma TAG legítima na
    passada 1.
    """
    return [
        texto
        for texto in textos
        if (texto.get("texto") or "").strip()
        and float(texto.get("confianca", 0.0)) >= confianca_minima
    ]


def _atribuir_internos(
    deteccoes: list[dict],
    textos: list[dict],
    usados: set[int],
) -> None:
    """Junta todos os textos internos de cada equipamento."""
    por_deteccao: dict[int, list[int]] = {}

    for indice_texto, texto in enumerate(textos):
        centro_x, centro_y = _centro(texto)

        dono = None
        menor_distancia = float("inf")

        for indice_deteccao, deteccao in enumerate(deteccoes):
            if not _dentro_da_caixa(deteccao, centro_x, centro_y):
                continue

            distancia = math.dist(
                (centro_x, centro_y),
                (deteccao["centro_x"], deteccao["centro_y"]),
            )

            if distancia < menor_distancia:
                menor_distancia = distancia
                dono = indice_deteccao

        if dono is not None:
            por_deteccao.setdefault(dono, []).append(indice_texto)
            usados.add(indice_texto)

    for indice_deteccao, indices in por_deteccao.items():
        escolhidos = [textos[i] for i in indices]

        deteccoes[indice_deteccao]["tag"] = _juntar(escolhidos)
        deteccoes[indice_deteccao]["confianca_ocr"] = _confianca_media(
            escolhidos
        )


def _atribuir_vizinho(
    deteccoes: list[dict],
    textos: list[dict],
    usados: set[int],
    raio_relativo: float,
) -> None:
    """Um único texto vizinho para cada equipamento ainda sem TAG."""
    pares = []

    for indice_deteccao, deteccao in enumerate(deteccoes):
        if deteccao["tag"]:
            continue

        raio = _raio(deteccao, raio_relativo)

        for indice_texto, texto in enumerate(textos):
            if indice_texto in usados:
                continue

            distancia = math.dist(
                _centro(texto),
                (deteccao["centro_x"], deteccao["centro_y"]),
            )

            if distancia <= raio:
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


def _juntar(textos: list[dict]) -> str:
    """
    Concatena os textos na ordem de leitura.

    O agrupamento por linha usa a altura do próprio texto como tolerância:
    dois fragmentos lado a lado na mesma linha têm centros verticais quase
    iguais, mas raramente idênticos, e ordenar só por Y os embaralharia.
    """
    if len(textos) == 1:
        return textos[0]["texto"].strip()

    altura_media = sum(t["h"] for t in textos) / len(textos)
    tolerancia = max(altura_media, 1.0)

    ordenados = sorted(
        textos,
        key=lambda t: (
            round(_centro(t)[1] / tolerancia),
            _centro(t)[0],
        ),
    )

    return " ".join(t["texto"].strip() for t in ordenados)


def _confianca_media(textos: list[dict]) -> float:
    if not textos:
        return 0.0

    return sum(float(t.get("confianca", 0.0)) for t in textos) / len(textos)


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
