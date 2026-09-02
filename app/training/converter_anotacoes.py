"""
Converte o dataset anotado em Pascal VOC (o que o Make Sense AI exporta)
para o formato COCO esperado pelo treino.

Adaptado de convert_annotations.py (repositório modelos_base_claude), com
duas diferenças pensadas para este projeto:

1. Lê direto a estrutura que a equipe já usa em dataset/original —
   treinamento/ e testes/, com as subpastas de quantidade e qualidade —
   em vez de exigir uma reorganização manual prévia.

2. Constrói a lista de classes a partir de TODOS os xml dos dois splits de
   uma vez. O conversor original derivava as classes só do split que
   estava convertendo, e era daí que vinha o id inconsistente entre
   train.json e val.json ("Bomba" = 3 no treino, 1 na validação). A
   canonicalização por nome no treino continua existindo como defesa, mas
   aqui o problema já nasce resolvido.

USO
---
    python -m app.training.converter_anotacoes \
        --treino dataset/original/treinamento \
        --validacao dataset/original/testes \
        --saida dataset/coco
"""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

EXTENSOES_IMAGEM = (".jpg", ".jpeg", ".png", ".bmp")


def listar_anotacoes(raiz: Path) -> list[Path]:
    """Todos os .xml sob a raiz, em qualquer nível de subpasta."""
    return sorted(raiz.rglob("*.xml"))


def localizar_imagem(xml_path: Path) -> Path | None:
    """
    Acha a imagem correspondente ao xml.

    Usa o nome do próprio arquivo xml em vez do campo <filename>: o campo
    guarda o caminho da máquina onde a anotação foi feita e nem sempre
    bate com o arquivo que está no disco.
    """
    for extensao in EXTENSOES_IMAGEM:
        candidato = xml_path.with_suffix(extensao)
        if candidato.is_file():
            return candidato
    return None


def coletar_classes(raizes) -> list[str]:
    """Nomes de classe de todos os xml de todos os splits, ordenados."""
    nomes: set[str] = set()

    for raiz in raizes:
        for xml_path in listar_anotacoes(raiz):
            for objeto in ET.parse(xml_path).getroot().findall("object"):
                nome = objeto.find("name")
                if nome is not None and nome.text:
                    nomes.add(nome.text.strip())

    return sorted(nomes)


def converter_split(
    raiz: Path,
    classes: list[str],
    destino_imagens: Path,
    destino_json: Path,
) -> tuple[int, int]:
    nome_para_id = {nome: i + 1 for i, nome in enumerate(classes)}
    categorias = [{"id": i + 1, "name": n} for i, n in enumerate(classes)]

    destino_imagens.mkdir(parents=True, exist_ok=True)

    imagens, anotacoes = [], []
    proximo_id = 1

    for img_id, xml_path in enumerate(listar_anotacoes(raiz), start=1):
        imagem = localizar_imagem(xml_path)

        if imagem is None:
            print(f"[AVISO] Sem imagem para {xml_path.name} — ignorado.")
            continue

        raiz_xml = ET.parse(xml_path).getroot()
        tamanho = raiz_xml.find("size")

        largura = int(tamanho.find("width").text)
        altura = int(tamanho.find("height").text)

        shutil.copy2(imagem, destino_imagens / imagem.name)

        imagens.append(
            {
                "id": img_id,
                "file_name": imagem.name,
                "width": largura,
                "height": altura,
            }
        )

        for objeto in raiz_xml.findall("object"):
            nome = objeto.find("name").text.strip()
            caixa = objeto.find("bndbox")

            xmin = float(caixa.find("xmin").text)
            ymin = float(caixa.find("ymin").text)
            xmax = float(caixa.find("xmax").text)
            ymax = float(caixa.find("ymax").text)

            largura_caixa = xmax - xmin
            altura_caixa = ymax - ymin

            if largura_caixa <= 0 or altura_caixa <= 0:
                continue

            anotacoes.append(
                {
                    "id": proximo_id,
                    "image_id": img_id,
                    "category_id": nome_para_id[nome],
                    "bbox": [xmin, ymin, largura_caixa, altura_caixa],
                    "area": largura_caixa * altura_caixa,
                    "iscrowd": 0,
                }
            )
            proximo_id += 1

    destino_json.parent.mkdir(parents=True, exist_ok=True)

    with open(destino_json, "w", encoding="utf-8") as arquivo:
        json.dump(
            {
                "images": imagens,
                "annotations": anotacoes,
                "categories": categorias,
            },
            arquivo,
            ensure_ascii=False,
            indent=2,
        )

    return len(imagens), len(anotacoes)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--treino", default="dataset/original/treinamento")
    parser.add_argument("--validacao", default="dataset/original/testes")
    parser.add_argument("--saida", default="dataset/coco")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raiz_treino = Path(args.treino)
    raiz_validacao = Path(args.validacao)
    saida = Path(args.saida)

    classes = coletar_classes([raiz_treino, raiz_validacao])

    if not classes:
        raise SystemExit(
            f"Nenhuma classe encontrada em {raiz_treino} / {raiz_validacao}. "
            f"Confira se os arquivos .xml estão nesses diretórios."
        )

    print(f"Classes ({len(classes)}): {classes}")

    for nome_split, raiz in (
        ("train", raiz_treino),
        ("val", raiz_validacao),
    ):
        n_imagens, n_anotacoes = converter_split(
            raiz,
            classes,
            saida / "images" / nome_split,
            saida / "annotations" / f"{nome_split}.json",
        )
        print(
            f"{nome_split}: {n_imagens} imagens, {n_anotacoes} anotações "
            f"-> {saida / 'annotations' / (nome_split + '.json')}"
        )


if __name__ == "__main__":
    main()
