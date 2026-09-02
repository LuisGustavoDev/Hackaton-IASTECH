"""
Verificação do detector na máquina de inferência.

Ferramenta de diagnóstico para responder, sem subir a API e sem depender
do Tesseract, três perguntas na ordem em que elas aparecem depois de
copiar um checkpoint recém-treinado para o notebook:

    1. O checkpoint é válido e o que tem dentro dele?
    2. Ele carrega e roda nesta máquina, em CPU?
    3. O que ele detecta numa imagem de verdade?

USO
---
    # 1 e 2: só inspeciona o checkpoint
    python -m app.detection.verificar

    # 3: roda a detecção numa imagem
    python -m app.detection.verificar --imagem dataset/original/testes/diagramas/qtd_baixa/qld_alta/101.jpg

    # com imagem anotada de saída, para olhar as caixas
    python -m app.detection.verificar --imagem 101.jpg --limiar 0.3 --salvar saida.png
"""

from __future__ import annotations

import argparse
import sys
import time
import unicodedata
from pathlib import Path

import torch

from app import config
from app.detection.checkpoint import carregar_checkpoint
from app.detection.detector import DetectorEquipamentos

LINHA = "=" * 72


def inspecionar(caminho: Path) -> dict:
    """Imprime o conteúdo do checkpoint sem construir o modelo."""
    dados = carregar_checkpoint(caminho)

    tamanho_mb = caminho.stat().st_size / (1024 * 1024)

    print(LINHA)
    print(" CHECKPOINT")
    print(LINHA)
    print(f"arquivo       : {caminho.resolve()}")
    print(f"tamanho       : {tamanho_mb:.1f} MB")
    print(f"formato       : versão {dados['formato_versao']}")
    print(f"arquitetura   : {dados['arquitetura']}")
    print(f"classes       : {dados['num_classes']}")

    for indice, nome in enumerate(dados["classes"]):
        # label = índice + 1; o 0 é o background
        print(f"    label {indice + 1:>2} -> {nome}")

    metadados = dados.get("metadados") or {}

    if metadados:
        print("metadados     :")
        for chave, valor in metadados.items():
            print(f"    {chave:<18} = {valor}")
    else:
        print("metadados     : (vazio)")

    return dados


def detectar(caminho_checkpoint: Path, imagem: Path, limiar: float) -> tuple:
    """Carrega o detector e roda numa imagem, medindo os tempos."""
    # Importado aqui para que a inspeção do checkpoint funcione mesmo numa
    # instalação sem opencv.
    from app.services.validacao_imagem import decodificar_imagem

    print()
    print(LINHA)
    print(" AMBIENTE")
    print(LINHA)
    print(f"torch         : {torch.__version__}")
    print(f"CUDA          : {torch.cuda.is_available()} (a produção roda em CPU)")

    inicio = time.perf_counter()
    detector = DetectorEquipamentos(caminho_checkpoint, limiar=limiar)
    tempo_carga = time.perf_counter() - inicio

    print(f"carga         : {tempo_carga:.1f}s")
    print(f"limiar        : {limiar}")

    array = decodificar_imagem(imagem.read_bytes())
    altura, largura = array.shape[:2]

    print()
    print(LINHA)
    print(" DETECÇÃO")
    print(LINHA)
    print(f"imagem        : {imagem}  ({largura}x{altura})")

    inicio = time.perf_counter()
    deteccoes = detector.detectar(array)
    tempo_inferencia = time.perf_counter() - inicio

    print(f"inferência    : {tempo_inferencia:.1f}s")
    print(f"detecções     : {len(deteccoes)} acima do limiar")

    if not deteccoes:
        print()
        print(
            "Nenhuma detecção. Se o modelo foi treinado por poucas épocas, "
            "tente um limiar menor (--limiar 0.1) antes de concluir que "
            "algo está errado."
        )
        return detector, array, deteccoes

    contagem: dict[str, int] = {}
    for deteccao in deteccoes:
        contagem[deteccao["classe"]] = contagem.get(deteccao["classe"], 0) + 1

    print()
    print("por classe:")
    for classe, quantidade in sorted(
        contagem.items(), key=lambda item: -item[1]
    ):
        print(f"    {quantidade:>4}x  {classe}")

    print()
    print("mais confiantes:")
    for deteccao in deteccoes[:10]:
        print(
            f"    {deteccao['score']:.3f}  {deteccao['classe']:<20} "
            f"centro=({deteccao['centro_x']}, {deteccao['centro_y']})"
        )

    return detector, array, deteccoes


def salvar_anotada(array, deteccoes: list[dict], destino: Path) -> Path:
    """Grava a imagem com as bounding boxes desenhadas."""
    import cv2

    anotada = array.copy()

    for deteccao in deteccoes:
        cv2.rectangle(
            anotada,
            (deteccao["x1"], deteccao["y1"]),
            (deteccao["x2"], deteccao["y2"]),
            (0, 0, 255),
            2,
        )

        cv2.putText(
            anotada,
            f"{_sem_acento(deteccao['classe'])} {deteccao['score']:.2f}",
            (deteccao["x1"], max(deteccao["y1"] - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    destino.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(destino), anotada):
        raise SystemExit(f"Não foi possível gravar {destino}")

    return destino


def _sem_acento(texto: str) -> str:
    """
    As fontes do OpenCV não têm acentuação: "Válvula" sairia desenhado
    como "V?lvula" na imagem anotada.
    """
    normalizado = unicodedata.normalize("NFKD", texto)
    return normalizado.encode("ascii", "ignore").decode("ascii")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Padrão: DETECTOR_CHECKPOINT_PATH ou data/models/faster_rcnn.pt",
    )
    parser.add_argument(
        "--imagem",
        default=None,
        help="Roda a detecção nesta imagem. Sem isso, só inspeciona o checkpoint.",
    )
    parser.add_argument("--limiar", type=float, default=None)
    parser.add_argument(
        "--salvar",
        default=None,
        help="Grava uma cópia da imagem com as caixas desenhadas.",
    )
    return parser.parse_args()


def _forcar_saida_utf8() -> None:
    """
    Imprime em UTF-8 mesmo no console do Windows.

    Sem isso, o console usa o codepage do sistema e os nomes de classe
    acentuados saem quebrados ("Conexão" -> "Conex?o") — justamente os
    nomes que esta ferramenta existe para mostrar. errors="replace" evita
    que um terminal limitado derrube o diagnóstico inteiro.
    """
    for fluxo in (sys.stdout, sys.stderr):
        try:
            fluxo.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def main() -> None:
    _forcar_saida_utf8()

    args = parse_args()

    caminho = Path(args.checkpoint or config.caminho_checkpoint())
    limiar = args.limiar if args.limiar is not None else config.limiar_deteccao()

    inspecionar(caminho)

    if args.imagem is None:
        print()
        print(
            "Checkpoint OK. Para testar a detecção numa imagem, rode de novo "
            "com --imagem <caminho>."
        )
        return

    _, array, deteccoes = detectar(caminho, Path(args.imagem), limiar)

    if args.salvar:
        destino = salvar_anotada(array, deteccoes, Path(args.salvar))
        print()
        print(f"imagem anotada: {destino.resolve()}")


if __name__ == "__main__":
    main()
