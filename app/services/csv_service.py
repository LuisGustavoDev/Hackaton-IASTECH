import csv
from pathlib import Path


def gerar_csv(resultados: list[dict], output_path: Path):

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as arquivo:

        writer = csv.DictWriter(
            arquivo,
            fieldnames=["equipamento", "quantidade"],
        )

        writer.writeheader()
        writer.writerows(resultados)