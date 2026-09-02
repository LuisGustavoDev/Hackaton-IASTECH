from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core.erros import (
    CheckpointInvalidoError,
    ImagemInvalidaError,
    ProcessamentoError,
)
from app.services.processamento import processar_imagem

router = APIRouter(prefix="/api")

MEDIA_TYPE_XLSX = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)


@router.get("/health")
def health():
    return {
        "status": "ok",
        "service": "iastech-api",
    }


@router.post("/process")
async def process(file: UploadFile = File(...)):
    """
    Recebe uma imagem de P&ID e devolve a planilha de equipamentos.

    O content-type informado no upload não é usado como critério: ele vem
    do cliente e um PDF renomeado chega como "image/png". Quem decide se o
    arquivo é uma imagem é app/services/validacao_imagem.py, olhando o
    conteúdo.
    """
    conteudo = await file.read()

    try:
        resultado = processar_imagem(conteudo)

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

    return FileResponse(
        path=resultado.planilha_xlsx,
        media_type=MEDIA_TYPE_XLSX,
        filename="resultado.xlsx",
    )
