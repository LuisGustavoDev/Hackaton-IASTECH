"""
Agrupamento de texto
"""

import cv2
import pytesseract
import sys
from pathlib import Path


def calcular_iou(a, b):
    """
    Calcula Intersection over Union entre dois retângulos.
    """

    ax1 = a["x"]
    ay1 = a["y"]
    ax2 = a["x"] + a["w"]
    ay2 = a["y"] + a["h"]

    bx1 = b["x"]
    by1 = b["y"]
    bx2 = b["x"] + b["w"]
    by2 = b["y"] + b["h"]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    area_intersecao = (
        (inter_x2 - inter_x1)
        * (inter_y2 - inter_y1)
    )

    area_a = a["w"] * a["h"]
    area_b = b["w"] * b["h"]

    area_uniao = (
        area_a
        + area_b
        - area_intersecao
    )

    if area_uniao == 0:
        return 0.0

    return area_intersecao / area_uniao

def remover_duplicadas(candidatos):
    """
    Remove regiões MSER duplicadas ou contidas umas nas outras.
    """

    candidatos = sorted(
        candidatos,
        key=lambda r: r["w"] * r["h"],
        reverse=True
    )

    mantidas = []

    for candidato in candidatos:

        cx = candidato["x"]
        cy = candidato["y"]
        cw = candidato["w"]
        ch = candidato["h"]

        area_candidato = cw * ch

        duplicada = False

        for mantida in mantidas:

            mx = mantida["x"]
            my = mantida["y"]
            mw = mantida["w"]
            mh = mantida["h"]

            x1 = max(cx, mx)
            y1 = max(cy, my)

            x2 = min(
                cx + cw,
                mx + mw
            )

            y2 = min(
                cy + ch,
                my + mh
            )

            if x2 <= x1 or y2 <= y1:
                continue

            area_intersecao = (
                (x2 - x1)
                * (y2 - y1)
            )

            area_mantida = mw * mh

            menor_area = min(
                area_candidato,
                area_mantida
            )

            proporcao_sobreposicao = (
                area_intersecao
                / menor_area
            )

            if proporcao_sobreposicao > 0.80:
                duplicada = True
                break

        if not duplicada:
            mantidas.append(candidato)

    return mantidas

def agrupar_caracteres(candidatos):
    """
    Agrupa caracteres detectados pelo MSER
    em regiões de texto.
    """

    candidatos = sorted(
        candidatos,
        key=lambda r: (
            r["y"],
            r["x"]
        )
    )

    grupos = []

    for caractere in candidatos:

        x = caractere["x"]
        y = caractere["y"]
        w = caractere["w"]
        h = caractere["h"]

        centro_y = y + h / 2

        melhor_grupo = None
        menor_distancia = float("inf")

        for grupo in grupos:

            gx = grupo["x"]
            gy = grupo["y"]
            gw = grupo["w"]
            gh = grupo["h"]

            centro_y_grupo = (
                gy + gh / 2
            )

            altura_media = (
                grupo["altura_media"]
            )

            diferenca_y = abs(
                centro_y
                - centro_y_grupo
            )

            fim_grupo = gx + gw

            distancia_x = (
                x - fim_grupo
            )

            mesma_linha = (
                diferenca_y
                <= altura_media * 0.6
            )

            altura_parecida = (
                altura_media * 0.5
                <= h
                <= altura_media * 1.5
            )

            distancia_pequena = (
                -5
                <= distancia_x
                <= altura_media * 2
            )

            if (
                mesma_linha
                and altura_parecida
                and distancia_pequena
            ):

                if (
                    distancia_x
                    < menor_distancia
                ):
                    menor_distancia = (
                        distancia_x
                    )
                    melhor_grupo = grupo

        if melhor_grupo is None:

            grupos.append({
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "altura_media": float(h),
                "caracteres": [caractere],
            })

        else:

            melhor_grupo[
                "caracteres"
            ].append(caractere)

            caracteres = (
                melhor_grupo[
                    "caracteres"
                ]
            )

            xs = [
                c["x"]
                for c in caracteres
            ]

            ys = [
                c["y"]
                for c in caracteres
            ]

            x2s = [
                c["x"] + c["w"]
                for c in caracteres
            ]

            y2s = [
                c["y"] + c["h"]
                for c in caracteres
            ]

            melhor_grupo["x"] = min(xs)
            melhor_grupo["y"] = min(ys)

            melhor_grupo["w"] = (
                max(x2s) - min(xs)
            )

            melhor_grupo["h"] = (
                max(y2s) - min(ys)
            )

            melhor_grupo[
                "altura_media"
            ] = (
                sum(
                    c["h"]
                    for c in caracteres
                )
                / len(caracteres)
            )

    return grupos

def mesclar_grupos(grupos):
    """
    Mescla grupos de texto que pertencem à mesma linha.

    Isso corrige casos como:

        Pneumati + tic -> Pneumatic
        Contro + ller   -> Controller

    Também trata grupos parcialmente sobrepostos.
    """

    mudou = True

    while mudou:

        mudou = False
        resultado = []

        grupos = sorted(
            grupos,
            key=lambda g: (
                g["y"],
                g["x"]
            )
        )

        usados = [False] * len(grupos)

        for i, grupo_a in enumerate(grupos):

            if usados[i]:
                continue

            grupo_atual = grupo_a.copy()

            usados[i] = True

            j = 0

            while j < len(grupos):

                if usados[j] or j == i:
                    j += 1
                    continue

                grupo_b = grupos[j]

                # -----------------------------------------
                # CENTRO VERTICAL
                # -----------------------------------------

                centro_a = (
                    grupo_atual["y"]
                    + grupo_atual["h"] / 2
                )

                centro_b = (
                    grupo_b["y"]
                    + grupo_b["h"] / 2
                )

                altura_media = (
                    grupo_atual["altura_media"]
                    + grupo_b["altura_media"]
                ) / 2

                diferenca_vertical = abs(
                    centro_a - centro_b
                )

                # -----------------------------------------
                # MESMA LINHA
                # -----------------------------------------

                mesma_linha = (
                    diferenca_vertical
                    <= altura_media * 0.7
                )

                if not mesma_linha:
                    j += 1
                    continue

                # -----------------------------------------
                # DISTÂNCIA HORIZONTAL
                # -----------------------------------------

                fim_a = (
                    grupo_atual["x"]
                    + grupo_atual["w"]
                )

                inicio_a = grupo_atual["x"]

                fim_b = (
                    grupo_b["x"]
                    + grupo_b["w"]
                )

                inicio_b = grupo_b["x"]

                # Distância entre os grupos.
                #
                # Se houver sobreposição, o valor será negativo.
                #

                if inicio_b >= fim_a:
                    distancia_horizontal = (
                        inicio_b - fim_a
                    )

                elif inicio_a >= fim_b:
                    distancia_horizontal = (
                        inicio_a - fim_b
                    )

                else:
                    # Existe sobreposição
                    distancia_horizontal = 0

                # -----------------------------------------
                # LIMITE
                # -----------------------------------------

                limite_horizontal = (
                    altura_media * 1.5
                )

                grupos_proximos = (
                    distancia_horizontal
                    <= limite_horizontal
                )

                if not grupos_proximos:
                    j += 1
                    continue

                # -----------------------------------------
                # MESCLAR
                # -----------------------------------------

                caracteres = (
                    grupo_atual["caracteres"]
                    + grupo_b["caracteres"]
                )

                xs = [
                    c["x"]
                    for c in caracteres
                ]

                ys = [
                    c["y"]
                    for c in caracteres
                ]

                x2s = [
                    c["x"] + c["w"]
                    for c in caracteres
                ]

                y2s = [
                    c["y"] + c["h"]
                    for c in caracteres
                ]

                grupo_atual["x"] = min(xs)
                grupo_atual["y"] = min(ys)

                grupo_atual["w"] = (
                    max(x2s) - min(xs)
                )

                grupo_atual["h"] = (
                    max(y2s) - min(ys)
                )

                grupo_atual["caracteres"] = caracteres

                grupo_atual["altura_media"] = (
                    sum(
                        c["h"]
                        for c in caracteres
                    )
                    / len(caracteres)
                )

                usados[j] = True

                mudou = True

                # Recomeça a procura porque o grupo
                # acabou de aumentar.
                j = 0

            resultado.append(
                grupo_atual
            )

        grupos = resultado

    return grupos
