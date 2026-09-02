"""
Canonicalização das categorias COCO (camada de treino).

Estes dois testes cobrem bugs reais já vividos no projeto:

- ids de categoria diferentes entre train.json e val.json para a mesma
  classe ("Bomba" é 3 num e 1 no outro no nosso dataset);
- nomes acentuados corrompidos por leitura sem encoding explícito no
  Windows ("Conexão" -> "ConexÃ£o").
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("pycocotools")
pytest.importorskip("torch")

from app.training.dataset_coco import (  # noqa: E402
    build_canonical_categories,
    load_coco_utf8,
    remap_coco_to_canonical,
)


def escrever_coco(caminho, categorias, anotacoes=()):
    conteudo = {
        "images": [{"id": 1, "file_name": "1.jpg", "width": 100, "height": 100}],
        "annotations": [
            {
                "id": i + 1,
                "image_id": 1,
                "category_id": category_id,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            }
            for i, category_id in enumerate(anotacoes)
        ],
        "categories": categorias,
    }

    with open(caminho, "w", encoding="utf-8") as arquivo:
        json.dump(conteudo, arquivo, ensure_ascii=False)

    return caminho


@pytest.fixture
def train_json(tmp_path):
    return escrever_coco(
        tmp_path / "train.json",
        [
            {"id": 1, "name": "Válvula"},
            {"id": 2, "name": "Conexão"},
            {"id": 3, "name": "Bomba"},
        ],
        anotacoes=[3, 1],
    )


@pytest.fixture
def val_json(tmp_path):
    # MESMAS classes, ids DIFERENTES — é assim que o exportador COCO
    # numera cada split de forma independente.
    return escrever_coco(
        tmp_path / "val.json",
        [
            {"id": 1, "name": "Bomba"},
            {"id": 2, "name": "Válvula"},
        ],
        anotacoes=[1],
    )


def test_classes_saem_ordenadas_e_sem_repeticao(train_json, val_json):
    classes = build_canonical_categories([train_json, val_json])

    assert classes == ["Bomba", "Conexão", "Válvula"]


def test_acentos_sobrevivem_a_leitura(train_json):
    classes = build_canonical_categories([train_json])

    assert "Conexão" in classes
    assert "ConexÃ£o" not in classes


def test_load_coco_utf8_preserva_acentos(train_json):
    coco = load_coco_utf8(train_json)

    nomes = {categoria["name"] for categoria in coco.dataset["categories"]}

    assert "Conexão" in nomes


def test_remap_faz_os_splits_concordarem(train_json, val_json):
    """
    "Bomba" é id 3 no treino e id 1 na validação. Depois do remap, os dois
    precisam apontar para o MESMO índice — senão o modelo aprende com um
    rótulo e é avaliado com outro.
    """
    classes = build_canonical_categories([train_json, val_json])

    coco_train = load_coco_utf8(train_json)
    coco_val = load_coco_utf8(val_json)

    remap_coco_to_canonical(coco_train, classes)
    remap_coco_to_canonical(coco_val, classes)

    def id_de(coco, nome):
        return next(
            categoria["id"]
            for categoria in coco.dataset["categories"]
            if categoria["name"] == nome
        )

    assert id_de(coco_train, "Bomba") == id_de(coco_val, "Bomba") == 0
    assert id_de(coco_train, "Válvula") == id_de(coco_val, "Válvula") == 2


def test_remap_atualiza_as_anotacoes(train_json, val_json):
    classes = build_canonical_categories([train_json, val_json])

    coco = load_coco_utf8(train_json)
    remap_coco_to_canonical(coco, classes)

    # Anotações do treino eram [3 = Bomba, 1 = Válvula].
    ids = [a["category_id"] for a in coco.dataset["annotations"]]

    assert ids == [0, 2]


def test_indice_canonico_mais_um_e_o_label_do_modelo(train_json, val_json):
    """
    Amarra a canonicalização à convenção de rótulos usada na inferência:
    classes[label - 1] tem de devolver o nome certo.
    """
    from app.detection.modelo import nome_da_classe

    classes = build_canonical_categories([train_json, val_json])

    coco = load_coco_utf8(train_json)
    remap_coco_to_canonical(coco, classes)

    indice_bomba = next(
        c["id"] for c in coco.dataset["categories"] if c["name"] == "Bomba"
    )

    assert nome_da_classe(indice_bomba + 1, classes) == "Bomba"
