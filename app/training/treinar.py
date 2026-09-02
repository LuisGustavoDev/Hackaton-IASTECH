"""
CLI de treino do detector de equipamentos (Faster R-CNN).

Roda na máquina forte (desktop com GPU NVIDIA / CUDA). O único produto que
precisa chegar à máquina de inferência é o checkpoint portátil gerado no
final — nem o dataset nem os arquivos de anotação COCO são necessários lá.

USO
---
    python -m app.training.treinar \
        --data-dir dataset/coco \
        --epochs 15 \
        --batch-size 4 \
        --saida data/models/faster_rcnn.pt

O dataset precisa estar no formato COCO:

    dataset/coco/
        images/train/*.jpg
        images/val/*.jpg
        annotations/train.json
        annotations/val.json

Se as anotações ainda estiverem em Pascal VOC (o que o Make Sense AI
exporta), converta antes com:

    python -m app.training.converter_anotacoes
"""

from __future__ import annotations

import argparse
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from app.detection.checkpoint import salvar_checkpoint
from app.detection.modelo import construir_faster_rcnn
from app.training.dataset_coco import (
    CocoDetectionRaw,
    build_canonical_categories,
    default_collate,
    to_torchvision_target,
)


def definir_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def treinar_uma_epoca(modelo, loader, otimizador, device) -> float:
    modelo.train()

    perda_total = 0.0
    lotes = 0

    for imagens, raw_targets in loader:
        tensores = [
            torch.from_numpy(
                np.ascontiguousarray(np.array(imagem))
            )
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .to(device)
            for imagem in imagens
        ]

        alvos = [
            {
                chave: valor.to(device)
                for chave, valor in to_torchvision_target(rt).items()
            }
            for rt in raw_targets
        ]

        perdas = modelo(tensores, alvos)
        perda = sum(perdas.values())

        otimizador.zero_grad()
        perda.backward()
        otimizador.step()

        perda_total += float(perda.item())
        lotes += 1

    return perda_total / max(lotes, 1)


@torch.no_grad()
def avaliar(modelo, dataset, device, limiar: float = 0.05) -> dict:
    """
    Avaliação COCO (mAP) no conjunto de validação.

    O import do pycocotools fica aqui dentro para que uma instalação sem
    ele ainda consiga treinar e salvar o checkpoint — a avaliação é
    diagnóstico, não pré-requisito do artefato de produção.
    """
    from pycocotools.cocoeval import COCOeval

    modelo.eval()

    predicoes = []

    for indice in range(len(dataset)):
        imagem, raw_target = dataset[indice]

        tensor = (
            torch.from_numpy(np.ascontiguousarray(np.array(imagem)))
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .to(device)
        )

        saida = modelo([tensor])[0]

        for box, score, label in zip(
            saida["boxes"].cpu().tolist(),
            saida["scores"].cpu().tolist(),
            saida["labels"].cpu().tolist(),
        ):
            if score < limiar or int(label) <= 0:
                continue

            x1, y1, x2, y2 = box

            predicoes.append(
                {
                    "image_id": int(raw_target["image_id"]),
                    "category_id": int(label) - 1,
                    "bbox": [x1, y1, max(x2 - x1, 0.0), max(y2 - y1, 0.0)],
                    "score": float(score),
                }
            )

    if not predicoes:
        print("[AVISO] Nenhuma detecção gerada — mAP não pôde ser calculado.")
        return {}

    coco_dt = dataset.coco.loadRes(predicoes)
    coco_eval = COCOeval(dataset.coco, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "mAP@[.5:.95]": float(coco_eval.stats[0]),
        "mAP@.5": float(coco_eval.stats[1]),
        "mAP@.75": float(coco_eval.stats[2]),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", default="dataset/coco")
    parser.add_argument("--saida", default="data/models/faster_rcnn.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sem-avaliacao",
        action="store_true",
        help="Pula o cálculo de mAP no fim do treino.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    if device.type == "cpu":
        print(
            "[AVISO] CUDA não disponível — o treino vai rodar em CPU e "
            "levar horas. Rode na máquina com GPU."
        )

    definir_seed(args.seed)

    data_dir = Path(args.data_dir)
    train_ann = data_dir / "annotations" / "train.json"
    val_ann = data_dir / "annotations" / "val.json"

    # Casa as categorias de treino e validação pelo NOME antes de montar
    # os datasets — ver dataset_coco.build_canonical_categories.
    classes = build_canonical_categories([train_ann, val_ann])

    ds_train = CocoDetectionRaw(
        data_dir / "images" / "train", train_ann, classes
    )
    ds_val = CocoDetectionRaw(data_dir / "images" / "val", val_ann, classes)

    print(f"Classes ({len(classes)}): {classes}")
    print(f"Treino: {len(ds_train)} imagens | Validação: {len(ds_val)} imagens")

    loader = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=default_collate,
        num_workers=args.num_workers,
    )

    modelo = construir_faster_rcnn(len(classes), pesos_pretreinados=True)
    modelo.to(device)

    parametros = [p for p in modelo.parameters() if p.requires_grad]
    otimizador = torch.optim.SGD(
        parametros, lr=args.lr, momentum=0.9, weight_decay=0.0005
    )

    inicio = time.perf_counter()
    perdas = []

    for epoca in range(args.epochs):
        perda = treinar_uma_epoca(modelo, loader, otimizador, device)
        perdas.append(perda)
        print(f"época {epoca + 1}/{args.epochs} - loss médio: {perda:.4f}")

    duracao = time.perf_counter() - inicio

    metricas = {}
    if not args.sem_avaliacao:
        metricas = avaliar(modelo, ds_val, device)

    caminho = salvar_checkpoint(
        args.saida,
        modelo,
        classes,
        metadados={
            "data_treino": datetime.now(timezone.utc).isoformat(),
            "epocas": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "device_treino": str(device),
            "torch_version": torch.__version__,
            "imagens_treino": len(ds_train),
            "imagens_val": len(ds_val),
            "loss_final": perdas[-1] if perdas else None,
            "tempo_treino_s": duracao,
            **metricas,
        },
    )

    tamanho_mb = caminho.stat().st_size / (1024 * 1024)

    print()
    print(f"Checkpoint portátil salvo em: {caminho.resolve()} ({tamanho_mb:.1f} MB)")
    print(
        "Copie SÓ esse arquivo para a máquina de inferência — o dataset e "
        "as anotações não são necessários lá."
    )


if __name__ == "__main__":
    main()
