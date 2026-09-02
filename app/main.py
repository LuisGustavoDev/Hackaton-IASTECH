from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.api.routes import router


app = FastAPI(
    title="IASTECH - Análise de Fluxogramas",
    description="API para processamento de fluxogramas industriais.",
    version="1.0.0",
)

# Sem CORS, um frontend no navegador não consegue nem chamar a API: a
# requisição é bloqueada pelo próprio navegador antes de sair. O curl não
# passa por isso, o que faz o problema aparecer só quando a interface
# entra em cena.
#
# expose_headers é igualmente necessário e menos óbvio: por padrão o
# JavaScript só enxerga um punhado de cabeçalhos da resposta, e
# Content-Disposition não está entre eles. Sem expô-lo, o frontend recebe
# o arquivo mas não consegue ler o nome sugerido para o download.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.origens_cors(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "message": "IASTECH API",
        "status": "online",
    }
