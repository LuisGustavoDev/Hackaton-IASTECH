"""
Responsável somente pelo tratamento da imagem
"""

import cv2


def carregar_imagem(caminho):
    imagem = cv2.imread(str(caminho))

    if imagem is None:
        raise FileNotFoundError(
            f"Imagem não encontrada ou inválida: {caminho}"
        )

    return imagem


def grayscale(imagem):
    return cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)


def resize(imagem, escala=2, metodo=cv2.INTER_CUBIC):
    return cv2.resize(
        imagem,
        None,
        fx=escala,
        fy=escala,
        interpolation=metodo
    )


def otsu(imagem):
    _, resultado = cv2.threshold(
        imagem,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return resultado


def adaptive_threshold(imagem):
    return cv2.adaptiveThreshold(
        imagem,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        5
    )