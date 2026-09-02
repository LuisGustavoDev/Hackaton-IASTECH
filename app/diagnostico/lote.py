"""
Roda o pipeline sobre uma pasta de imagens, gravando cada execução.

Serve para medir o modelo sobre o conjunto de teste inteiro em vez de uma
imagem por vez — é assim que se produz o baseline contra o qual o próximo
checkpoint será comparado.

Usa exatamente o mesmo `processar_imagem` que a API usa. Isso é
proposital: uma medição que passasse por um caminho paralelo mediria esse
caminho, não a produção.

USO
---
    docker compose run --rm app python -m app.diagnostico.lote \
        --imagens dataset/original/testes

O Tesseract e o dataset existem dentro do container (o compose monta
./dataset e ./data como volumes), e o resultado vai para o banco
apontado por DB_PATH.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.core.erros import IastechError
from app.detection.detector import obter_detector
from app.diagnostico.utf8 import forcar_utf8
from app.services.processamento import processar_imagem

EXTENSOES = (".jpg", ".jpeg", ".png", ".bmp")


def listar_imagens(raiz: Path) -> list[Path]:
    return sorted(
        caminho
        for caminho in raiz.rglob("*")
        if caminho.suffix.lower() in EXTENSOES
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--imagens", default="dataset/original/testes")
    parser.add_argument(
        "--limite",
        type=int,
        default=None,
        help="Processa no máximo N imagens (útil para um teste rápido).",
    )
    return parser.parse_args()


def main() -> None:
    forcar_utf8()

    args = parse_args()

    raiz = Path(args.imagens)

    if not raiz.is_dir():
        raise SystemExit(f"Pasta não encontrada: {raiz}")

    imagens = listar_imagens(raiz)

    if args.limite:
        imagens = imagens[: args.limite]

    if not imagens:
        raise SystemExit(f"Nenhuma imagem em {raiz}")

    # Carrega o modelo antes do laço: são segundos de carga que não devem
    # ser contados no tempo da primeira imagem.
    detector = obter_detector()

    print(f"checkpoint : {detector.caminho_checkpoint}")
    print(f"limiar     : {detector.limiar}")
    print(f"imagens    : {len(imagens)}")
    print()

    ids = []
    falhas = 0

    for indice, caminho in enumerate(imagens, start=1):
        try:
            resultado = processar_imagem(
                caminho.read_bytes(), arquivo_nome=caminho.name
            )
        except IastechError as erro:
            falhas += 1
            print(f"[{indice}/{len(imagens)}] {caminho.name}: FALHOU — {erro}")
            continue

        ids.append(resultado.execucao_id)

        com_tag = sum(1 for linha in resultado.linhas if linha["TAG"])

        print(
            f"[{indice}/{len(imagens)}] {caminho.name:<16} "
            f"execucao={resultado.execucao_id}  "
            f"deteccoes={len(resultado.linhas):<4} com TAG={com_tag}"
        )

    print()
    print(f"{len(ids)} execuções gravadas, {falhas} falha(s).")

    if ids:
        print(f"ids: {min(i for i in ids if i)}..{max(i for i in ids if i)}")
        print()
        print(
            "Para medir contra o gabarito:\n"
            "    python -m app.diagnostico.matriz_confusao --ultimas "
            f"{len(ids)}"
        )


if __name__ == "__main__":
    main()
