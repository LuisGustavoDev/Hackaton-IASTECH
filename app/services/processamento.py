from pathlib import Path
import uuid


TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)


def processar_imagem(imagem: bytes) -> Path:

    imagem_path = salvar_temporariamente(imagem)

    imagem_processada = preprocessar(imagem_path)

    regioes = detectar_texto(imagem_processada)

    textos = executar_ocr(regioes)

    elementos = identificar_elementos(textos)

    csv_path = gerar_csv(elementos)

    return csv_path