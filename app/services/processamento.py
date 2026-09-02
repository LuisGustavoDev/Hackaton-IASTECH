"""
Orquestração do processamento de um P&ID.

    bytes recebidos
        -> validação (é mesmo uma imagem?)
        -> detecção de equipamentos (Faster R-CNN)
        -> OCR das tags (pipeline MSER + Tesseract já existente)
        -> associação texto <-> equipamento
        -> interpretação da TAG (ISA-5.1)
        -> planilha (.xlsx + .csv)
        -> registro da execução no banco

Cada requisição trabalha numa pasta própria dentro de data/output, o que
mantém uploads simultâneos isolados e deixa os arquivos intermediários
disponíveis para conferência (data/output já está no .gitignore).
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2

from app.core.erros import ProcessamentoError
from app.detection.detector import obter_detector
from app.models import execucoes
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
    execucao_id: int | None = None


def processar_imagem(
    imagem: bytes,
    arquivo_nome: str | None = None,
) -> ResultadoProcessamento:
    """
    Processa uma imagem de P&ID e devolve os caminhos das planilhas.

    Levanta ImagemInvalidaError se o arquivo não for uma imagem utilizável,
    CheckpointInvalidoError se o modelo não estiver disponível e
    ProcessamentoError para falhas no meio do caminho.
    """
    inicio = time.perf_counter()

    array = decodificar_imagem(imagem)
    altura, largura = array.shape[:2]

    pasta = DIRETORIO_SAIDA / uuid.uuid4().hex
    pasta.mkdir(parents=True, exist_ok=True)

    caminho_imagem = pasta / "entrada.png"

    if not cv2.imwrite(str(caminho_imagem), array):
        raise ProcessamentoError(
            f"Não foi possível gravar a imagem em {caminho_imagem}."
        )

    detector = obter_detector()

    marco = time.perf_counter()
    try:
        deteccoes = detector.detectar(array)
    except Exception as erro:
        raise ProcessamentoError(
            f"Falha na detecção de equipamentos: {erro}"
        ) from erro
    tempo_deteccao = (time.perf_counter() - marco) * 1000

    marco = time.perf_counter()
    try:
        textos = image_service.processar_imagem(caminho_imagem)
    except Exception as erro:
        raise ProcessamentoError(
            f"Falha no reconhecimento de texto (OCR): {erro}"
        ) from erro
    tempo_ocr = (time.perf_counter() - marco) * 1000

    equipamentos = associar(deteccoes, textos)
    linhas = montar_linhas(equipamentos)

    caminho_xlsx, caminho_csv = planilha_service.gerar_planilha(
        linhas,
        pasta / "resultado.xlsx",
        pasta / "resultado.csv",
    )

    tempo_total = (time.perf_counter() - inicio) * 1000

    execucao_id = _registrar(
        largura=largura,
        altura=altura,
        equipamentos=equipamentos,
        pasta=pasta,
        arquivo_nome=arquivo_nome,
        detector=detector,
        tempo_deteccao_ms=tempo_deteccao,
        tempo_ocr_ms=tempo_ocr,
        tempo_total_ms=tempo_total,
    )

    return ResultadoProcessamento(
        linhas=linhas,
        planilha_xlsx=caminho_xlsx,
        planilha_csv=caminho_csv,
        imagem=caminho_imagem,
        execucao_id=execucao_id,
    )


def montar_linhas(equipamentos: list[dict]) -> list[dict]:
    """
    Converte os equipamentos já associados às tags nas linhas da planilha.

    Enriquece cada equipamento, no caminho, com o resultado da leitura da
    TAG (`tag_normalizada`, `descricao`, `grupo`). É desses mesmos campos
    que a persistência lê depois — assim o que fica gravado no banco é
    exatamente o que foi entregue ao cliente, sem uma segunda derivação
    que pudesse divergir.

    Coordenada X/Y são o centro da bounding box, em pixels da imagem
    original — é o ponto que localiza o equipamento no diagrama.
    """
    linhas = []

    for equipamento in equipamentos:
        analise = tag_service.analisar(
            equipamento.get("tag"),
            equipamento["classe"],
        )

        equipamento["tag_normalizada"] = analise.tag
        equipamento["descricao"] = analise.descricao
        equipamento["grupo"] = analise.grupo

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


def _registrar(
    *,
    largura: int,
    altura: int,
    equipamentos: list[dict],
    pasta: Path,
    arquivo_nome: str | None,
    detector,
    tempo_deteccao_ms: float,
    tempo_ocr_ms: float,
    tempo_total_ms: float,
) -> int | None:
    """
    Grava a execução, sem deixar que uma falha de banco derrube o
    processamento.

    O registro é telemetria: serve para medir o modelo depois, não é o
    produto que o usuário pediu. Se o SQLite estiver mal configurado ou o
    disco cheio, é melhor entregar a planilha e devolver execucao_id=None
    do que negar o resultado. Só sqlite3.Error é engolido — um TypeError
    aqui seria bug de verdade e precisa aparecer.
    """
    try:
        return execucoes.registrar(
            imagem_largura=largura,
            imagem_altura=altura,
            equipamentos=equipamentos,
            arquivo_nome=arquivo_nome,
            checkpoint=str(detector.caminho_checkpoint),
            limiar=detector.limiar,
            pasta=str(pasta),
            tempo_deteccao_ms=tempo_deteccao_ms,
            tempo_ocr_ms=tempo_ocr_ms,
            tempo_total_ms=tempo_total_ms,
        )
    except sqlite3.Error:
        return None
