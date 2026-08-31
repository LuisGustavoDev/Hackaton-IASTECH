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

def realizar_ocr(imagem, grupo):
    """
    Recorta uma região de texto, amplia,
    aplica threshold e executa o Tesseract.

    Retorna:
        texto
        confiança média
        imagem processada
    """

    x = grupo["x"]
    y = grupo["y"]
    w = grupo["w"]
    h = grupo["h"]

    # -----------------------------------------
    # MARGEM
    # -----------------------------------------

    margem = 10

    altura, largura = imagem.shape[:2]

    x1 = max(
        0,
        x - margem
    )

    y1 = max(
        0,
        y - margem
    )

    x2 = min(
        largura,
        x + w + margem
    )

    y2 = min(
        altura,
        y + h + margem
    )

    recorte = imagem[
        y1:y2,
        x1:x2
    ]

    if recorte.size == 0:
        return "", 0.0, None

    # -----------------------------------------
    # ESCALA DE CINZA
    # -----------------------------------------

    gray = cv2.cvtColor(
        recorte,
        cv2.COLOR_BGR2GRAY
    )

    # -----------------------------------------
    # AMPLIAÇÃO
    # -----------------------------------------

    escala = 3

    ampliada = cv2.resize(
        gray,
        None,
        fx=escala,
        fy=escala,
        interpolation=cv2.INTER_CUBIC
    )

    # -----------------------------------------
    # THRESHOLD
    # -----------------------------------------

    _, thresh = cv2.threshold(
        ampliada,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )

    # -----------------------------------------
    # OCR
    # -----------------------------------------

    dados = pytesseract.image_to_data(
        thresh,
        lang="eng",
        config="--psm 7",
        output_type=pytesseract.Output.DICT
    )

    textos = []
    confiancas = []

    for i, texto in enumerate(
        dados["text"]
    ):

        texto = texto.strip()

        if not texto:
            continue

        try:
            confianca = float(
                dados["conf"][i]
            )
        except (ValueError, TypeError):
            continue

        if confianca < 0:
            continue

        textos.append(texto)
        confiancas.append(
            confianca
        )

    if not textos:
        return "", 0.0, thresh

    texto_final = " ".join(textos)

    confianca_media = (
        sum(confiancas)
        / len(confiancas)
    )

    return (
        texto_final,
        confianca_media,
        thresh
    )


def main():

    entrada = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(
            "dataset/original/diagramas/"
            "qtd_baixa/qld_alta/148.jpg"
        )
    )

    saida = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else Path(
            "dataset/processado/"
            "text_regions.png"
        )
    )

    if not entrada.is_file():
        raise FileNotFoundError(
            f"Imagem não encontrada: {entrada}"
        )

    imagem = cv2.imread(
        str(entrada)
    )

    if imagem is None:
        raise ValueError(
            f"Não foi possível abrir a imagem: "
            f"{entrada}"
        )

    # ==================================================
    # ESCALA DE CINZA
    # ==================================================

    gray = cv2.cvtColor(
        imagem,
        cv2.COLOR_BGR2GRAY
    )

    # ==================================================
    # MSER
    # ==================================================

    mser = cv2.MSER_create()

    regioes, _ = mser.detectRegions(
        gray
    )

    candidatos = []

    for regiao in regioes:

        x, y, w, h = cv2.boundingRect(
            regiao
        )

        # ----------------------------------------------
        # FILTROS GEOMÉTRICOS
        # ----------------------------------------------

        if w < 2 or h < 5:
            continue

        if w > 60 or h > 60:
            continue

        area = w * h

        if area < 20:
            continue

        proporcao = w / h

        if (
            proporcao > 8
            or proporcao < 0.1
        ):
            continue

        # ----------------------------------------------
        # FILTRO DE COR
        # ----------------------------------------------

        roi_gray = gray[
            y:y + h,
            x:x + w
        ]

        if roi_gray.size == 0:
            continue

        pixels_escuros = cv2.countNonZero(
            cv2.inRange(
                roi_gray,
                0,
                100
            )
        )

        proporcao_escura = (
            pixels_escuros
            / roi_gray.size
        )

        if proporcao_escura < 0.08:
            continue

        candidatos.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
        })

    print(
        f"\nRegiões MSER filtradas: "
        f"{len(candidatos)}"
    )

    # ==================================================
    # REMOVER DUPLICADAS
    # ==================================================

    candidatos = remover_duplicadas(
        candidatos
    )

    print(
        f"Após remover duplicadas: "
        f"{len(candidatos)}"
    )

    # ==================================================
    # AGRUPAR
    # ==================================================

    grupos = agrupar_caracteres(
        candidatos
    )

    print(
        f"Grupos iniciais: "
        f"{len(grupos)}"
    )

    grupos = mesclar_grupos(
        grupos
    )

    print(
        f"Grupos após mesclagem: "
        f"{len(grupos)}\n"
    )

    # ==================================================
    # DIRETÓRIO DOS CROPS
    # ==================================================

    pasta_crops = Path(
        "dataset/processado/ocr_regions"
    )

    pasta_crops.mkdir(
        parents=True,
        exist_ok=True
    )

    # ==================================================
    # RESULTADO
    # ==================================================

    resultado = imagem.copy()

    resultados_ocr = []

    for indice, grupo in enumerate(
        grupos,
        start=1
    ):

        x = grupo["x"]
        y = grupo["y"]
        w = grupo["w"]
        h = grupo["h"]

        quantidade = len(
            grupo["caracteres"]
        )

        # -----------------------------------------
        # OCR
        # -----------------------------------------

        texto, confianca, crop = (
            realizar_ocr(
                imagem,
                grupo
            )
        )

        # -----------------------------------------
        # SALVAR CROP
        # -----------------------------------------

        if crop is not None:

            caminho_crop = (
                pasta_crops
                / f"region_{indice:03d}.png"
            )

            cv2.imwrite(
                str(caminho_crop),
                crop
            )

        # -----------------------------------------
        # GUARDAR RESULTADO
        # -----------------------------------------

        resultados_ocr.append({
            "id": indice,
            "texto": texto,
            "confianca": confianca,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
        })

        # -----------------------------------------
        # TERMINAL
        # -----------------------------------------

        print(
            f'Grupo {indice:03d}: '
            f'"{texto}" '
            f'conf={confianca:.1f} '
            f'pos=({x}, {y}) '
            f'tam=({w}x{h}) '
            f'caracteres={quantidade}'
        )

        # -----------------------------------------
        # RETÂNGULO
        # -----------------------------------------

        cv2.rectangle(
            resultado,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        # -----------------------------------------
        # TEXTO OCR NA IMAGEM
        # -----------------------------------------

        if texto:

            texto_exibicao = texto

            # Evita texto gigante na imagem
            if len(texto_exibicao) > 25:
                texto_exibicao = (
                    texto_exibicao[:22]
                    + "..."
                )

            cv2.putText(
                resultado,
                texto_exibicao,
                (
                    x,
                    max(y - 5, 15)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 0, 0),
                1,
                cv2.LINE_AA
            )

    # ==================================================
    # SALVAR RESULTADO
    # ==================================================

    saida.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(saida),
        resultado
    )

    print(
        f"\nResultado salvo em: {saida}"
    )

    print(
        f"Crops salvos em: {pasta_crops}"
    )


if __name__ == "__main__":
    main()