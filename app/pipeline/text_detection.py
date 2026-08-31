"""
MSER
LÓGICA: detectar regiões, aplicar filtros geométricos e filtrar pela proporção de pixels escuros.
"""

import cv2

def detectar_mser(imagem):
    mser = cv2.MSER_create()

    regioes, _ = mser.detectRegions(imagem)

    candidatos = []

    for regiao in regioes:

        x, y, w, h = cv2.boundingRect(regiao)

        if w < 2 or h < 5:
            continue

        if w > 60 or h > 60:
            continue

        if w * h < 20:
            continue

        proporcao = w / h

        if proporcao > 8 or proporcao < 0.1:
            continue

        roi = imagem[y:y+h, x:x+w]

        if roi.size == 0:
            continue

        pixels_escuros = cv2.countNonZero(
            cv2.inRange(roi, 0, 100)
        )

        proporcao_escura = pixels_escuros / roi.size

        if proporcao_escura < 0.08:
            continue

        candidatos.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h
        })

    return candidatos