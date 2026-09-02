"""
Orquestração do processamento de um P&ID.

    bytes recebidos
        -> validação (é mesmo uma imagem?)
        -> detecção de equipamentos (Faster R-CNN)
        -> OCR das tags (pipeline MSER + Tesseract já existente)
        -> associação texto <-> equipamento
        -> interpretação da TAG (ISA-5.1)
        -> planilha (.xlsx + .csv)

Cada requisição trabalha numa pasta própria dentro de data/output, o que
mantém uploads simultâneos isolados e deixa os arquivos intermediários
disponíveis para conferência (data/output já está no .gitignore).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2

from app.core.erros import ProcessamentoError
from app.detection.detector import obter_detector
from app.services import image_service, planilha_service, tag_service
from app.services.associacao import associar
from app.services.validacao_imagem import decodificar_imagem

DIRETORIO_SAIDA = Path("data/output")


@dataclass(frozen=True)
class ResultadoProcessamento:
    """O que uma execução produziu."""

    linhas: list[dict]
    planilha_xlsx: Path
    planilha_csv: Path
    imagem: Path


def processar_imagem(imagem: bytes) -> ResultadoProcessamento:
    """
    Processa uma imagem de P&ID e devolve os caminhos das planilhas.

    Levanta ImagemInvalidaError se o arquivo não for uma imagem utilizável,
    CheckpointInvalidoError se o modelo não estiver disponível e
    ProcessamentoError para falhas no meio do caminho.
    """
    array = decodificar_imagem(imagem)

    pasta = DIRETORIO_SAIDA / uuid.uuid4().hex
    pasta.mkdir(parents=True, exist_ok=True)

    caminho_imagem = pasta / "entrada.png"

    if not cv2.imwrite(str(caminho_imagem), array):
        raise ProcessamentoError(
            f"Não foi possível gravar a imagem em {caminho_imagem}."
        )

    detector = obter_detector()

    try:
        deteccoes = detector.detectar(array)
    except Exception as erro:
        raise ProcessamentoError(
            f"Falha na detecção de equipamentos: {erro}"
        ) from erro

    try:
        textos = image_service.processar_imagem(caminho_imagem)
    except Exception as erro:
        raise ProcessamentoError(
            f"Falha no reconhecimento de texto (OCR): {erro}"
        ) from erro

    linhas = montar_linhas(associar(deteccoes, textos))

    caminho_xlsx, caminho_csv = planilha_service.gerar_planilha(
        linhas,
        pasta / "resultado.xlsx",
        pasta / "resultado.csv",
    )

    return ResultadoProcessamento(
        linhas=linhas,
        planilha_xlsx=caminho_xlsx,
        planilha_csv=caminho_csv,
        imagem=caminho_imagem,
    )


def montar_linhas(equipamentos: list[dict]) -> list[dict]:
    """
    Converte os equipamentos já associados às tags nas linhas da planilha.

    Coordenada X/Y são o centro da bounding box, em pixels da imagem
    original — é o ponto que localiza o equipamento no diagrama.
    """
    linhas = []

    for equipamento in equipamentos:
        analise = tag_service.analisar(
            equipamento.get("tag"),
            equipamento["classe"],
        )

        linhas.append(
            {
                "TAG": analise.tag,
                "Tipo": equipamento["classe"],
                "Descrição": analise.descricao,
                "Coordenada X": equipamento["centro_x"],
                "Coordenada Y": equipamento["centro_y"],
                "Grupo": analise.grupo,
            }
        )

    # Ordena por grupo e depois por TAG: é como a planilha fica legível
    # para quem confere o diagrama equipamento por equipamento.
    linhas.sort(key=lambda linha: (str(linha["Grupo"]), str(linha["TAG"])))

    return linhas
