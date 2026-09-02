"""
Leitura do dataset COCO para o treino.

Adaptado de benchmark_pid_models.py (repositório modelos_base_claude), que
continua existindo como ferramenta separada para comparar arquiteturas. A
lógica foi COPIADA, não importada: produção não deve depender de um script
de benchmark.

Duas armadilhas reais tratadas aqui, ambas já custaram bug no projeto:

1. Encoding: o pycocotools abre o json sem informar encoding, e no Windows
   isso usa o codepage do sistema — "Conexão" vira "ConexÃ£o" e o
   casamento de categorias entre splits quebra em silêncio.

2. Ids de categoria: o exportador COCO numera as classes de forma
   independente em cada split. No nosso dataset, "Bomba" é id 3 no
   train.json e id 1 no val.json. Casar por id levaria duas classes
   diferentes ao mesmo índice, e o modelo treinaria com os rótulos
   trocados sem nenhum erro visível.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


def load_coco_utf8(ann_file: str | Path) -> COCO:
    """
    Carrega um json COCO sempre como UTF-8 e devolve um objeto COCO já
    indexado.

    Lendo o json nós mesmos e entregando o dict pronto ao pycocotools, o
    resultado deixa de depender do locale da máquina.
    """
    with open(ann_file, encoding="utf-8") as arquivo:
        dataset = json.load(arquivo)

    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()

    return coco


def build_canonical_categories(ann_files) -> list[str]:
    """
    Une as categorias de vários arquivos COCO (train.json e val.json) e
    devolve a lista canônica de nomes de classe, em ordem alfabética.

    Casa as categorias pelo NOME, nunca pelo id numérico. Esta lista é a
    fonte da verdade dos rótulos do modelo — é ela que vai para dentro do
    checkpoint portátil, e é o índice dela (+1) que o modelo aprende.
    """
    nomes: list[str] = []
    vistos: set[str] = set()

    for ann_file in ann_files:
        with open(ann_file, encoding="utf-8") as arquivo:
            categorias = json.load(arquivo).get("categories", [])

        for categoria in categorias:
            if categoria["name"] not in vistos:
                vistos.add(categoria["name"])
                nomes.append(categoria["name"])

    return sorted(nomes)


def remap_coco_to_canonical(coco: COCO, canonical_classes: list[str]) -> None:
    """
    Substitui o category_id de todas as categorias e anotações pelo índice
    canônico (0..C-1, baseado no NOME) e reconstrói os índices do
    pycocotools.

    Depois disso, treino e validação "falam a mesma língua" de ids, mesmo
    que os jsons originais numerassem as classes de formas diferentes.
    """
    nome_para_canonico = {
        nome: indice for indice, nome in enumerate(canonical_classes)
    }

    id_antigo_para_nome = {
        categoria["id"]: categoria["name"]
        for categoria in coco.dataset["categories"]
    }

    for categoria in coco.dataset["categories"]:
        categoria["id"] = nome_para_canonico[categoria["name"]]

    for anotacao in coco.dataset["annotations"]:
        nome = id_antigo_para_nome[anotacao["category_id"]]
        anotacao["category_id"] = nome_para_canonico[nome]

    coco.createIndex()


class CocoDetectionRaw(Dataset):
    """
    Lê um dataset em formato COCO e devolve (PIL.Image, anotação crua).

    A anotação crua tem bbox em [x, y, w, h] e category_id JÁ remapeado
    para o índice canônico 0..C-1.
    """

    def __init__(
        self,
        images_dir: str | Path,
        ann_file: str | Path,
        canonical_classes: list[str],
    ) -> None:
        self.images_dir = Path(images_dir)
        self.coco = load_coco_utf8(ann_file)
        remap_coco_to_canonical(self.coco, canonical_classes)

        self.img_ids = sorted(self.coco.getImgIds())
        self.classes = list(canonical_classes)

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        imagem = Image.open(
            self.images_dir / img_info["file_name"]
        ).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)

        raw_target = {
            "image_id": img_id,
            "width": img_info["width"],
            "height": img_info["height"],
            "annotations": self.coco.loadAnns(ann_ids),
        }

        return imagem, raw_target


def default_collate(batch):
    """Mantém imagens e anotações como listas — o Faster R-CNN aceita
    imagens de tamanhos diferentes no mesmo batch."""
    imagens, raw_targets = zip(*batch)
    return list(imagens), list(raw_targets)


def to_torchvision_target(raw_target: dict) -> dict:
    """
    Converte a anotação crua para o formato do torchvision: boxes em xyxy
    absoluto e labels 1..C.

    O +1 no label é a convenção de background do Faster R-CNN — a mesma
    aplicada em app/detection/modelo.py na hora de traduzir de volta.
    """
    boxes, labels, areas, iscrowd = [], [], [], []

    for anotacao in raw_target["annotations"]:
        x, y, w, h = anotacao["bbox"]

        if w <= 0 or h <= 0:
            continue

        boxes.append([x, y, x + w, y + h])
        labels.append(anotacao["category_id"] + 1)
        areas.append(anotacao.get("area", w * h))
        iscrowd.append(anotacao.get("iscrowd", 0))

    return {
        "boxes": _tensor(boxes, torch.float32, (0, 4)),
        "labels": _tensor(labels, torch.int64, (0,)),
        "image_id": torch.tensor([raw_target["image_id"]]),
        "area": _tensor(areas, torch.float32, (0,)),
        "iscrowd": _tensor(iscrowd, torch.int64, (0,)),
    }


def _tensor(valores, dtype, shape_vazio):
    if not valores:
        return torch.zeros(shape_vazio, dtype=dtype)
    return torch.as_tensor(valores, dtype=dtype)
