"""
Calibração do limiar de confiança do OCR.

`config.OCR_CONFIANCA_MINIMA` decide o que o MSER+Tesseract produziram
que merece virar candidato a TAG. O valor atual foi escolhido por
julgamento, não por medição — esta ferramenta troca o chute por dados.

Roda o OCR sobre um conjunto de imagens SEM filtro nenhum, classifica
cada fragmento lido em duas famílias e varre os limiares possíveis:

    "parece TAG" -> casa com o padrão ISA (letras + número); é o que NÃO
                    se pode perder
    "ruído"      -> o resto; em geral traço do próprio símbolo que o
                    Tesseract interpretou como letra

O limiar bom é o maior que ainda não descarta nenhum "parece TAG". Subir
além disso passa a custar tag de verdade.

USO
---
    docker compose run --rm app python -m app.diagnostico.calibrar_ocr \
        --imagens dataset/original/testes --limite 5

Precisa do Tesseract, que só existe dentro do container.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app import config
from app.diagnostico.utf8 import forcar_utf8
from app.services import image_service
from app.services.tag_service import analisar

EXTENSOES = (".jpg", ".jpeg", ".png", ".bmp")

LIMIARES = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90)


def coletar(imagens: list[Path]) -> list[tuple[str, float, bool]]:
    """(texto, confiança, parece_tag) de tudo que o OCR leu."""
    fragmentos = []

    for indice, caminho in enumerate(imagens, start=1):
        try:
            textos = image_service.processar_imagem(caminho)
        except Exception as erro:
            print(f"[{indice}/{len(imagens)}] {caminho.name}: FALHOU — {erro}")
            continue

        for item in textos:
            texto = (item.get("texto") or "").strip()

            if not texto:
                continue

            fragmentos.append(
                (
                    texto,
                    float(item.get("confianca", 0.0)),
                    bool(analisar(texto).tag),
                )
            )

        print(f"[{indice}/{len(imagens)}] {caminho.name:<16} {len(textos)} textos")

    return fragmentos


def relatorio(fragmentos: list[tuple[str, float, bool]]) -> None:
    tags = [f for f in fragmentos if f[2]]
    ruido = [f for f in fragmentos if not f[2]]

    print()
    print("=" * 70)
    print(" CALIBRAÇÃO DO LIMIAR DO OCR")
    print("=" * 70)
    print(f"fragmentos lidos : {len(fragmentos)}")
    print(f"  parecem TAG    : {len(tags)}")
    print(f"  ruído          : {len(ruido)}")
    print(f"limiar em uso    : {config.OCR_CONFIANCA_MINIMA}")

    if not tags:
        print()
        print(
            "Nenhum fragmento casou com o padrão ISA. Sem TAG legítima no "
            "conjunto não há o que calibrar — confira se as imagens "
            "escolhidas têm tags escritas."
        )
        return

    print()
    print("limiar   TAGs mantidas   TAGs perdidas   ruído mantido")
    print("-" * 56)

    for limiar in LIMIARES:
        mantidas = sum(1 for _, c, _ in tags if c >= limiar)
        ruido_mantido = sum(1 for _, c, _ in ruido if c >= limiar)

        marca = "  <- atual" if limiar == config.OCR_CONFIANCA_MINIMA else ""

        print(
            f"{limiar:>6}   {mantidas:>13}   {len(tags) - mantidas:>13}   "
            f"{ruido_mantido:>13}{marca}"
        )

    menor_confianca_tag = min(c for _, c, _ in tags)

    print()
    print(
        f"Menor confiança entre os fragmentos que parecem TAG: "
        f"{menor_confianca_tag:.1f}"
    )

    if config.OCR_CONFIANCA_MINIMA > menor_confianca_tag:
        perdidas = [
            (t, c) for t, c, _ in tags if c < config.OCR_CONFIANCA_MINIMA
        ]
        print(
            f"O limiar atual ({config.OCR_CONFIANCA_MINIMA}) está "
            f"DESCARTANDO {len(perdidas)} fragmento(s) que casam com o "
            f"padrão ISA:"
        )
        for texto, confianca in sorted(perdidas, key=lambda x: x[1])[:10]:
            print(f"    {confianca:>5.1f}  {texto!r}")
        print(
            f"Considere baixar OCR_CONFIANCA_MINIMA para algo abaixo de "
            f"{menor_confianca_tag:.0f}."
        )
    else:
        print(
            f"O limiar atual ({config.OCR_CONFIANCA_MINIMA}) não descarta "
            f"nenhuma TAG deste conjunto."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--imagens", default="dataset/original/testes")
    parser.add_argument("--limite", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    forcar_utf8()

    args = parse_args()

    raiz = Path(args.imagens)

    if not raiz.is_dir():
        raise SystemExit(f"Pasta não encontrada: {raiz}")

    imagens = sorted(
        c for c in raiz.rglob("*") if c.suffix.lower() in EXTENSOES
    )

    if args.limite:
        imagens = imagens[: args.limite]

    if not imagens:
        raise SystemExit(f"Nenhuma imagem em {raiz}")

    relatorio(coletar(imagens))


if __name__ == "__main__":
    main()
