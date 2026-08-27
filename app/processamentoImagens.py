import cv2
import sys
from pathlib import Path

# GRAYSCALE
entrada = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
	"dataset/original/diagramas/qtd_media/qld_alta/153.jpg"
)
saida = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
	"dataset/processado/pid_153_gray.png"
)

imagem = cv2.imread(str(entrada))
if imagem is None:
	raise FileNotFoundError(f"Imagem de entrada não encontrada ou inválida: {entrada}")

cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

saida.parent.mkdir(parents=True, exist_ok=True)
if not cv2.imwrite(str(saida), cinza):
	raise OSError(f"Não foi possível salvar a imagem processada: {saida}")