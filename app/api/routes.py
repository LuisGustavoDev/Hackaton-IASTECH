from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from app.services.processamento import processar_imagem


router = APIRouter(prefix="/api")


@router.get("/health")
def health():
    return {
        "status": "ok",
        "service": "iastech-api",
    }


@router.post("/process")
async def process(file: UploadFile = File(...)):

    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="Tipo de arquivo não informado."
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="O arquivo enviado precisa ser uma imagem."
        )

    imagem = await file.read()

    try:
        csv_path = processar_imagem(imagem)

        return FileResponse(
            path=csv_path,
            media_type="text/csv",
            filename="resultado.csv",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro durante o processamento: {str(e)}",
        )