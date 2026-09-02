"""
Empacota um state_dict solto no checkpoint portátil do projeto.

Serve para aproveitar os pesos que o benchmark_pid_models.py já produziu
(resultados_benchmark/faster_rcnn.pt é um state_dict puro, sem a lista de
classes) sem precisar treinar tudo de novo.

A lista de classes é reconstruída com build_canonical_categories() a
partir dos MESMOS arquivos de anotação usados no treino — é isso que
garante que o índice de cada classe aqui seja o mesmo que o modelo
aprendeu. Passar jsons diferentes dos do treino produz um checkpoint que
carrega sem erro e devolve as classes trocadas.

USO
---
    python -m app.training.empacotar_checkpoint \
        --pesos ../modelos_base_claude/resultados_benchmark/faster_rcnn.pt \
        --train-json ../modelos_base_claude/dataset/annotations/train.json \
        --val-json ../modelos_base_claude/dataset/annotations/val.json \
        --saida data/models/faster_rcnn.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from app.detection.checkpoint import salvar_checkpoint
from app.detection.modelo import construir_faster_rcnn
from app.training.dataset_coco import build_canonical_categories


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pesos", required=True)
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--val-json", required=True)
    parser.add_argument("--saida", default="data/models/faster_rcnn.pt")
    parser.add_argument(
        "--origem",
        default="benchmark_pid_models.py",
        help="Anotado nos metadados, só para rastreabilidade.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    classes = build_canonical_categories([args.train_json, args.val_json])
    print(f"Classes ({len(classes)}): {classes}")

    state_dict = torch.load(args.pesos, map_location="cpu", weights_only=True)

    if not isinstance(state_dict, dict) or "state_dict" in state_dict:
        raise SystemExit(
            f"{args.pesos} não parece ser um state_dict puro. Se ele já "
            f"estiver no formato portátil, não há nada para converter."
        )

    modelo = construir_faster_rcnn(len(classes), pesos_pretreinados=False)

    # Sem strict=False: se as classes reconstruídas não baterem com a
    # cabeça treinada, o load falha aqui e não em produção, com as classes
    # silenciosamente trocadas.
    modelo.load_state_dict(state_dict)

    caminho = salvar_checkpoint(
        args.saida,
        modelo,
        classes,
        metadados={
            "origem": args.origem,
            "pesos_originais": str(Path(args.pesos).name),
            "torch_version": torch.__version__,
            "observacao": (
                "Empacotado a partir de um state_dict do benchmark; "
                "as métricas de validação estão no relatório do benchmark."
            ),
        },
    )

    tamanho_mb = caminho.stat().st_size / (1024 * 1024)
    print(f"Checkpoint portátil salvo em: {caminho.resolve()} ({tamanho_mb:.1f} MB)")


if __name__ == "__main__":
    main()
