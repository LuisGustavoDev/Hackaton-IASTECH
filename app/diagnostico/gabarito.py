"""
Leitura do gabarito anotado (Pascal VOC) para as ferramentas de medição.

O gabarito NÃO é copiado para o banco: os .xml em dataset/original já
estão versionados, e duplicá-los criaria duas fontes da verdade que
poderiam divergir.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


def indexar(raiz: str | Path) -> dict[str, Path]:
    """
    Mapeia o nome do arquivo de imagem para o .xml correspondente.

    A chave é o nome da imagem (ex.: "101.jpg") porque é o que fica
    gravado em `execucoes.arquivo_nome`.
    """
    raiz = Path(raiz)
    indice: dict[str, Path] = {}

    for xml_path in raiz.rglob("*.xml"):
        for extensao in (".jpg", ".jpeg", ".png", ".bmp"):
            imagem = xml_path.with_suffix(extensao)
            if imagem.is_file():
                indice[imagem.name] = xml_path
                break

    return indice


def caixas(xml_path: str | Path) -> list[dict]:
    """Anotações de um .xml, em xyxy absoluto."""
    raiz = ET.parse(xml_path).getroot()

    anotacoes = []

    for objeto in raiz.findall("object"):
        nome = objeto.find("name")
        caixa = objeto.find("bndbox")

        if nome is None or caixa is None or not nome.text:
            continue

        x1 = float(caixa.find("xmin").text)
        y1 = float(caixa.find("ymin").text)
        x2 = float(caixa.find("xmax").text)
        y2 = float(caixa.find("ymax").text)

        if x2 <= x1 or y2 <= y1:
            continue

        anotacoes.append(
            {"classe": nome.text.strip(), "x1": x1, "y1": y1, "x2": x2, "y2": y2}
        )

    return anotacoes


def iou(a: dict, b: dict) -> float:
    """Intersection over Union entre duas caixas em xyxy."""
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersecao = (x2 - x1) * (y2 - y1)

    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])

    uniao = area_a + area_b - intersecao

    return intersecao / uniao if uniao > 0 else 0.0
