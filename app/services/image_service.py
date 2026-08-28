from pathlib import Path

from app.pipeline.preprocessing import (
    carregar_imagem,
    grayscale
)

from app.pipeline.text_detection import (
    detectar_mser
)

from app.pipeline.text_grouping import (
    remover_duplicadas,
    agrupar_caracteres,
    mesclar_grupos
)

from app.pipeline.ocr import (
    realizar_ocr
)


def processar_imagem(caminho):
    imagem = carregar_imagem(caminho)

    gray = grayscale(imagem)

    candidatos = detectar_mser(gray)

    candidatos = remover_duplicadas(
        candidatos
    )

    grupos = agrupar_caracteres(
        candidatos
    )

    grupos = mesclar_grupos(
        grupos
    )

    resultados = []

    for indice, grupo in enumerate(
        grupos,
        start=1
    ):

        texto, confianca, crop = realizar_ocr(
            imagem,
            grupo
        )

        resultados.append({
            "id": indice,
            "texto": texto,
            "confianca": confianca,
            "x": grupo["x"],
            "y": grupo["y"],
            "w": grupo["w"],
            "h": grupo["h"],
            "crop": crop
        })

    return resultados