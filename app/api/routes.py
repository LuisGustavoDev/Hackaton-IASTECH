from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from app.core.erros import (
    CheckpointInvalidoError,
    ImagemInvalidaError,
    ProcessamentoError,
)
from app.models import execucoes
from app.services.processamento import processar_imagem

router = APIRouter(prefix="/api")

MEDIA_TYPE_XLSX = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

FORMATOS = {
    "xlsx": ("resultado.xlsx", MEDIA_TYPE_XLSX),
    "csv": ("resultado.csv", "text/csv"),
}


@router.get("/health")
def health():
    return {
        "status": "ok",
        "service": "iastech-api",
    }


@router.post("/process")
def process(
    file: UploadFile = File(...),
    formato: str = Query(
        "xlsx",
        pattern="^(xlsx|csv|json)$",
        description=(
            "xlsx/csv devolvem o arquivo direto; json devolve as linhas "
            "para pré-visualização mais os links de download."
        ),
    ),
):
    """
    Recebe uma imagem de P&ID e devolve a planilha de equipamentos.

    O content-type informado no upload não é usado como critério: ele vem
    do cliente e um PDF renomeado chega como "image/png". Quem decide se o
    arquivo é uma imagem é app/services/validacao_imagem.py, olhando o
    conteúdo.

    `formato=json` existe para a interface gráfica: ela precisa mostrar a
    quantidade encontrada e uma pré-visualização da lista ANTES de o
    usuário baixar o Excel, e um corpo binário não serve para isso.

    Rota síncrona de propósito. `processar_imagem` gasta segundos de CPU
    na inferência do Faster R-CNN; num `async def` isso rodaria dentro do
    event loop e travaria o servidor inteiro durante o processamento —
    requisições simultâneas serializariam e até o /api/health ficaria
    pendurado. Declarada como `def`, o FastAPI a executa no threadpool.
    """
    conteudo = file.file.read()

    try:
        resultado = processar_imagem(conteudo, arquivo_nome=file.filename)

    except ImagemInvalidaError as erro:
        # Erro do arquivo enviado.
        raise HTTPException(status_code=400, detail=str(erro))

    except CheckpointInvalidoError as erro:
        # O sistema está no ar, mas sem modelo para trabalhar.
        raise HTTPException(status_code=503, detail=str(erro))

    except ProcessamentoError as erro:
        raise HTTPException(status_code=500, detail=str(erro))

    except Exception as erro:
        raise HTTPException(
            status_code=500,
            detail=f"Erro durante o processamento: {erro}",
        )

    if formato == "json":
        return {
            "execucao_id": resultado.execucao_id,
            "arquivo": file.filename,
            "quantidade": len(resultado.linhas),
            "linhas": resultado.linhas,
            "download": _links(resultado.execucao_id),
        }

    if formato == "csv":
        return FileResponse(
            path=resultado.planilha_csv,
            media_type="text/csv",
            filename="resultado.csv",
        )

    return FileResponse(
        path=resultado.planilha_xlsx,
        media_type=MEDIA_TYPE_XLSX,
        filename="resultado.xlsx",
    )


@router.get("/resultado/{execucao_id}")
def baixar_resultado(
    execucao_id: int,
    formato: str = Query("xlsx", pattern="^(xlsx|csv)$"),
):
    """
    Devolve a planilha de uma execução já processada.

    Permite que a interface gráfica separe as duas ações que o Plano de
    Desenvolvimento pede: processar e pré-visualizar primeiro, exportar o
    Excel depois — sem reprocessar a imagem, que custa segundos de CPU.
    """
    resumo = execucoes.obter(execucao_id)

    if resumo is None:
        raise HTTPException(
            status_code=404,
            detail=f"Execução {execucao_id} não encontrada.",
        )

    if not resumo.pasta:
        raise HTTPException(
            status_code=404,
            detail=(
                f"A execução {execucao_id} foi registrada antes de o "
                f"sistema passar a guardar o local dos arquivos. "
                f"Reprocesse a imagem."
            ),
        )

    nome, media_type = FORMATOS[formato]
    caminho = Path(resumo.pasta) / nome

    if not caminho.is_file():
        raise HTTPException(
            status_code=410,
            detail=(
                f"Os arquivos da execução {execucao_id} não estão mais no "
                f"disco. Reprocesse a imagem."
            ),
        )

    return FileResponse(
        path=caminho,
        media_type=media_type,
        filename=nome,
    )


@router.get("/execucoes")
def listar_execucoes(limite: int = Query(20, ge=1, le=200)):
    """Histórico de processamentos, do mais recente para o mais antigo."""
    return [
        {
            "execucao_id": resumo.id,
            "criado_em": resumo.criado_em,
            "arquivo": resumo.arquivo_nome,
            "quantidade": resumo.qtd_deteccoes,
            "tags_lidas": resumo.qtd_tags_lidas,
            "tempo_total_ms": resumo.tempo_total_ms,
            "download": _links(resumo.id) if resumo.pasta else None,
        }
        for resumo in execucoes.listar(limite=limite)
    ]


def _links(execucao_id: int | None) -> dict | None:
    if execucao_id is None:
        return None

    return {
        "xlsx": f"/api/resultado/{execucao_id}?formato=xlsx",
        "csv": f"/api/resultado/{execucao_id}?formato=csv",
    }
