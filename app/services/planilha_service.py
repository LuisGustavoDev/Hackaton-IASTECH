"""
Exportação da planilha de resultados.

As colunas e a ordem são as fechadas na reunião de 14/08/2026 e no Plano
de Desenvolvimento — não devem ser alteradas sem alinhamento com o
cliente:

    TAG | Tipo | Descrição | Coordenada X | Coordenada Y | Grupo

Gera os dois formatos: .xlsx (o entregável pedido no plano, "arquivo Excel
pronto para utilização") e .csv (útil para conferência rápida e diff).
"""

from __future__ import annotations

import csv
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font

COLUNAS = [
    "TAG",
    "Tipo",
    "Descrição",
    "Coordenada X",
    "Coordenada Y",
    "Grupo",
]

_LARGURAS = {
    "TAG": 16,
    "Tipo": 20,
    "Descrição": 40,
    "Coordenada X": 14,
    "Coordenada Y": 14,
    "Grupo": 10,
}


def gerar_xlsx(linhas: list[dict], destino: str | Path) -> Path:
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)

    planilha = Workbook()
    aba = planilha.active
    aba.title = "Equipamentos"

    aba.append(COLUNAS)

    for celula in aba[1]:
        celula.font = Font(bold=True)

    for indice, coluna in enumerate(COLUNAS, start=1):
        aba.column_dimensions[
            aba.cell(row=1, column=indice).column_letter
        ].width = _LARGURAS[coluna]

    for linha in linhas:
        aba.append([linha.get(coluna, "") for coluna in COLUNAS])

    aba.freeze_panes = "A2"

    planilha.save(destino)

    return destino


def gerar_csv(linhas: list[dict], destino: str | Path) -> Path:
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)

    with destino.open("w", newline="", encoding="utf-8-sig") as arquivo:
        escritor = csv.DictWriter(
            arquivo,
            fieldnames=COLUNAS,
            extrasaction="ignore",
        )

        escritor.writeheader()

        for linha in linhas:
            escritor.writerow(
                {coluna: linha.get(coluna, "") for coluna in COLUNAS}
            )

    return destino


def gerar_planilha(
    linhas: list[dict],
    destino_xlsx: str | Path,
    destino_csv: str | Path,
) -> tuple[Path, Path]:
    """Gera os dois arquivos e devolve (caminho_xlsx, caminho_csv)."""
    return (
        gerar_xlsx(linhas, destino_xlsx),
        gerar_csv(linhas, destino_csv),
    )
