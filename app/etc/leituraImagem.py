import cv2
import pytesseract
import sys
from pathlib import Path

def main():
    entrada = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "dataset/processado/pid_102_otsu.png"
    )

    if not entrada.is_file():
        raise FileNotFoundError(f"Imagem não encontrada: {entrada}")

    imagem = cv2.imread(str(entrada))
    if imagem is None:
        raise ValueError(f"Não foi possível abrir a imagem: {entrada}")

    dados = pytesseract.image_to_data(
        imagem,
        lang="eng",
        config="--psm 11",
        output_type = pytesseract.Output.DICT
    )

    encontrados = []

    for i, texto in enumerate(dados["text"]):
        texto = texto.strip()

        if not texto:
            continue

        try:
            confianca = float(dados["conf"][i])
        except ValueError:
            continue

        if confianca < 30:
            continue

        x = dados["left"][i]
        y = dados["top"][i]
        w = dados["width"][i]
        h = dados["height"][i]

        encontrados.append({
            "texto": texto,
            "confianca": confianca,
            "x": x,
            "y": y,
            "largura": w,
            "altura": h,
        })

    print("\n=== Textos Encontrados ===\n")

    for item in encontrados:
        print(
            f'{item["texto"]:<20} '
            f'conf={item["confianca"]:.1f} '
            f'pos=({item["x"]}, {item["y"]})'
            f'tam=({item["largura"]}x{item["altura"]})'
        )
    

if __name__ == "__main__":
    main()
    # The previous code has been replaced by the main function implementation.