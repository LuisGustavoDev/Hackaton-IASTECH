import cv2
import pytesseract

def realizar_ocr(imagem, grupo, escala=3):
    x = grupo["x"]
    y = grupo["y"]
    w = grupo["w"]
    h = grupo["h"]

    margem = 10

    altura, largura = imagem.shape[:2]

    x1 = max(0, x - margem)
    y1 = max(0, y - margem)
    x2 = min(largura, x + w + margem)
    y2 = min(altura, y + h + margem)

    recorte = imagem[y1:y2, x1:x2]

    if recorte.size == 0:
        return "", 0.0, None

    gray = cv2.cvtColor(
        recorte,
        cv2.COLOR_BGR2GRAY
    )

    ampliada = cv2.resize(
        gray,
        None,
        fx=escala,
        fy=escala,
        interpolation=cv2.INTER_CUBIC
    )

    _, thresh = cv2.threshold(
        ampliada,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    dados = pytesseract.image_to_data(
        thresh,
        lang="eng",
        config="--psm 7",
        output_type=pytesseract.Output.DICT
    )

    textos = []
    confiancas = []

    for i, texto in enumerate(dados["text"]):

        texto = texto.strip()

        if not texto:
            continue

        try:
            confianca = float(dados["conf"][i])
        except (ValueError, TypeError):
            continue

        if confianca < 0:
            continue

        textos.append(texto)
        confiancas.append(confianca)

    if not textos:
        return "", 0.0, thresh

    texto_final = " ".join(textos)

    confianca_media = (
        sum(confiancas) / len(confiancas)
    )

    return texto_final, confianca_media, thresh
