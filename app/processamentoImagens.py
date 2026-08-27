import cv2
import numpy as np
import sys
from pathlib import Path

entrada = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
	"dataset/original/diagramas/qtd_baixa/qld_baixa/102.jpg"
)
saida_grayscale = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
	"dataset/processado/pid_102_gray.png"
)
saida_resized_cubic = Path(
	"dataset/processado/pid_102_resized_cubic.png"
)
saida_resized_linear = Path(
	"dataset/processado/pid_102_resized_linear.png"
)
saida_resized_lanczo = Path(
	"dataset/processado/pid_102_resized_lanczo.png"
)
saida_resized_nearest = Path(
	"dataset/processado/pid_102_resized_nearest.png"
)
saida_contrast = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(
	"dataset/processado/pid_102_contrast.png"
)
saida_gausianBlur = Path(
	"dataset/processado/pid_102_gausianBlur.png"
)
saida_medianBlur = Path(
	"dataset/processado/pid_102_medianBlur.png"
)
saida_bilateralFilter = Path(
	"dataset/processado/pid_102_bilateralFilter.png"
)
saida_binary = Path(
	"dataset/processado/pid_102_binary.png"
)
saida_otsu = Path(
	"dataset/processado/pid_102_otsu.png"
)
saida_adaptative = Path(
	"dataset/processado/pid_102_adaptative.png"
)
saida_erosion = Path(
	"dataset/processado/pid_102_erosion.png"
)
saida_dilation = Path(
	"dataset/processado/pid_102_dilation.png"
)
saida_opening = Path(
	"dataset/processado/pid_102_opening.png"
)
saida_closing = Path(
	"dataset/processado/pid_102_closing.png"
)
saida_lines = Path(
	"dataset/processado/pid_102_lines.png"
)
saida_without_lines = Path(
	"dataset/processado/pid_102_without_lines.png"
)

imagem = cv2.imread(str(entrada))
if imagem is None:
	raise FileNotFoundError(f"Imagem de entrada não encontrada ou inválida: {entrada}")


# Redimensionamento
"""
MÉTODOS DE INTERPOLAÇÃO:
    -   cv2.INTER_NEAREST
    -   cv2.INTER_LINEAR
    -   cv2.INTER_CUBIC
    -   cv2.INTER_LANCZOS4
"""
"""imagem_2x_cubic = cv2.resize(
	imagem,
	None,
	fx=2,
	fy=2,
	interpolation = cv2.INTER_CUBIC
)

imagem_2x_linear = cv2.resize(
	imagem,
	None,
	fx=2,
	fy=2,
	interpolation = cv2.INTER_LINEAR
)

imagem_2x_lanczo = cv2.resize(
	imagem,
	None,
	fx=2,
	fy=2,
	interpolation = cv2.INTER_LANCZOS4
)

imagem_2x_nearest = cv2.resize(
	imagem,
	None,
	fx=4,
	fy=4,
	interpolation = cv2.INTER_NEAREST
)

saida_resized_cubic.parent.mkdir(parents=True, exist_ok=True)
if not cv2.imwrite(str(saida_resized_cubic), imagem_2x_cubic):
	raise OSError(f"Não foi possível salvar a imagem processada: {saida_resized_cubic}")

saida_resized_linear.parent.mkdir(parents=True, exist_ok=True)
if not cv2.imwrite(str(saida_resized_linear), imagem_2x_linear):
	raise OSError(f"Não foi possível salvar a imagem processada: {saida_resized_linear}")

saida_resized_lanczo.parent.mkdir(parents=True, exist_ok=True)
if not cv2.imwrite(str(saida_resized_lanczo), imagem_2x_lanczo):
	raise OSError(f"Não foi possível salvar a imagem processada: {saida_resized_lanczo}")

saida_resized_nearest.parent.mkdir(parents=True, exist_ok=True)
if not cv2.imwrite(str(saida_resized_nearest), imagem_2x_nearest):
	raise OSError(f"Não foi possível salvar a imagem processada: {saida_resized_nearest}")"""


# Grayscale
"""cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

saida_grayscale.parent.mkdir(parents=True, exist_ok=True)
if not cv2.imwrite(str(saida_grayscale), cinza):
	raise OSError(f"Não foi possível salvar a imagem processada: {saida_grayscale}")"""


# Contraste
"""
VALORES DE CONTRASTE:
    clipLimit = 1.0
	clipLimit = 2.0
	clipLimit = 3.0
	clipLimit = 4.0
"""
"""clahe = cv2.createCLAHE(
	clipLimit = 2.0,
	tileGridSize = (8, 8)
)

contraste = clahe.apply(cinza)

saida_contrast.parent.mkdir(parents=True, exist_ok=True)
if not cv2.imwrite(str(saida_contrast), contraste):
	raise OSError(f"Não foi possível salvar a imagem com contraste: {saida_contrast}")"""


# Redução de ruído
"""
FILTROS:
    - Gaussian Blur
	- Median Blur
	- Bilateral Blur
"""
"""gausianBlur = cv2.GaussianBlur(
	cinza,
	(5, 5),
	0
)

medianBlur = cv2.medianBlur(
	cinza,
	5
)

bilateralFilter = cv2.bilateralFilter(
	cinza,
	9,
	75,
	75
)

saida_gausianBlur.parent.mkdir(parents=True, exist_ok=True)
if not cv2.imwrite(str(saida_gausianBlur), gausianBlur):
	raise OSError(f"Não foi possível salvar a imagem com contraste: {saida_gausianBlur}")

saida_medianBlur.parent.mkdir(parents=True, exist_ok=True)
if not cv2.imwrite(str(saida_medianBlur), medianBlur):
	raise OSError(f"Não foi possível salvar a imagem com contraste: {saida_medianBlur}")

saida_bilateralFilter.parent.mkdir(parents=True, exist_ok=True)
if not cv2.imwrite(str(saida_bilateralFilter), bilateralFilter):
	raise OSError(f"Não foi possível salvar a imagem com contraste: {saida_bilateralFilter}")"""

# Binarização
"""
TIPOS DE BINARIZAÇÃO:
    - Threshold manual (binary)
	- Threshold automático (otsu)
	- Adaptive
"""
"""threshold_value, binary = cv2.threshold(
	cinza,
	200,
	255,
	cv2.THRESH_BINARY
)

threshold_value, otsu = cv2.threshold(
	cinza,
	0,
	255,
	cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

adaptive = cv2.adaptiveThreshold(
	cinza,
	255,
	cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
	cv2.THRESH_BINARY,
	15,
	5
)

saida_binary.parent.mkdir(parents=True, exist_ok=True)
if not cv2.imwrite(str(saida_binary), binary):
	raise OSError(f"Não foi possível salvar a imagem processada: {saida_binary}")

saida_otsu.parent.mkdir(parents=True, exist_ok=True)
if not cv2.imwrite(str(saida_otsu), otsu):
	raise OSError(f"Não foi possível salvar a imagem processada: {saida_otsu}")

saida_adaptative.parent.mkdir(parents=True, exist_ok=True)
if not cv2.imwrite(str(saida_adaptative), adaptive):
	raise OSError(f"Não foi possível salvar a imagem processada: {saida_adaptative}")"""


# Operações Morfológicas
"""
PRINCIPAIS OPERAÇÕES:
    - Erosion
        Pode remover pequenos elementos.
	- Dilation
        Expande regiões.
	- Opening
        Pode ser útil para remover pequenos ruídos.
	- Closing
        Pode ajudar a fechar pequenas interrupções.
"""
"""kernel = cv2.getStructuringElement(
	cv2.MORPH_RECT,
	(3, 3)
)

eroded = cv2.erode(
	binary,
	kernel,
	iterations=1
)

dilated = cv2.dilate(
	binary,
	kernel,
	iterations=1
)

opening = cv2.morphologyEx(
	binary,
	cv2.MORPH_OPEN,
	kernel
)

closing = cv2.morphologyEx(
	binary,
	cv2.MORPH_CLOSE,
	kernel
)

saida_erosion.parent.mkdir(parents=True, exist_ok=True)
if not cv2.imwrite(str(saida_erosion), eroded):
	raise OSError(f"Não foi possível salvar a imagem processada: {saida_erosion}")

saida_dilation.parent.mkdir(parents=True, exist_ok=True)
if not cv2.imwrite(str(saida_dilation), dilated):
	raise OSError(f"Não foi possível salvar a imagem processada: {saida_dilation}")

saida_opening.parent.mkdir(parents=True, exist_ok=True)
if not cv2.imwrite(str(saida_opening), opening):
	raise OSError(f"Não foi possível salvar a imagem processada: {saida_opening}")

saida_closing.parent.mkdir(parents=True, exist_ok=True)
if not cv2.imwrite(str(saida_closing), closing):
	raise OSError(f"Não foi possível salvar a imagem processada: {saida_closing}")"""


# Detecção e remoção de linhas
"""edges = cv2.Canny(
	cinza,
	50,
	150
)

lines = cv2.HoughLinesP(
	edges,
	1,
	3.14159 / 180,
	threshold = 100,
	minLineLength = 100,
	maxLineGap = 10
)

imagem_com_linhas = imagem.copy()
mascara_linhas = np.zeros(cinza.shape, dtype=np.uint8)
if lines is not None:
	for line in lines:
		x1, y1, x2, y2 = line[0]
		cv2.line(imagem_com_linhas, (x1, y1), (x2, y2), (0, 0, 255), 2)
		cv2.line(mascara_linhas, (x1, y1), (x2, y2), 255, 2)

resultado = cv2.inpaint(
	imagem,
	mascara_linhas,
	3,
	cv2.INPAINT_TELEA
)

saida_lines.parent.mkdir(parents=True, exist_ok=True)
if not cv2.imwrite(str(saida_lines), imagem_com_linhas):
	raise OSError(f"Não foi possível salvar a imagem processada: {saida_lines}")

saida_without_lines.parent.mkdir(parents=True, exist_ok=True)
if not cv2.imwrite(str(saida_without_lines), resultado):
	raise OSError(f"Não foi possível salvar a imagem processada: {saida_without_lines}")"""
