import cv2
import sys
from pathlib import Path

def main():
    entrada = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "dataset/original/diagramas/qtd_baixa/qld_alta/148.jpg"
    )

    saida = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
        "dataset/processado/text_regions.png"
    )

    if not entrada.is_file():
        raise FileNotFoundError(f"Imagem não encontrada: {entrada}")

    imagem = cv2.imread(str(entrada))

    if imagem is None:
        raise ValueError(f"Não foi possível abrir a imagem: {entrada}")

    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # DETECTOR MSER
    mser = cv2.MSER_create()

    regioes, _ = mser.detectRegions(gray)

    resultado = imagem.copy()

    candidatos = []

    for regiao in regioes:

        x, y, w, h = cv2.boundingRect(regiao)

        # Filtros básicos
        if w < 2 or h < 5:
            continue

        if w > 100 or h > 100:
            continue

        area = w * h

        if area < 20:
            continue

        # Proporção
        proporcao = w / h

        if proporcao > 15 or proporcao < 0.1:
            continue

        candidatos.append((x, y, w, h))

    print(f"\nRegiões detectadas: {len(candidatos)}\n")

    for x, y, w, h in candidatos:
        cv2.rectangle(
            resultado,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            1
        )

    saida.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(saida), resultado)

    print(f"Resultado salvo em: {saida}")

if __name__ == "__main__":
    main()