from fastapi import FastAPI

from app.api.routes import router


app = FastAPI(
    title="IASTECH - Análise de Fluxogramas",
    description="API para processamento de fluxogramas industriais.",
    version="1.0.0",
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "message": "IASTECH API",
        "status": "online",
    }



