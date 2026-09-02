"""
Exceções de domínio da aplicação.

Ficam separadas do FastAPI de propósito: os serviços e o pipeline não
conhecem HTTP. Quem traduz cada exceção em um status code é a camada de
rotas (app/api/routes.py), que é o único lugar do sistema que sabe que
existe um cliente HTTP do outro lado.
"""

from __future__ import annotations


class IastechError(Exception):
    """Base de todos os erros previstos da aplicação."""


class ImagemInvalidaError(IastechError):
    """
    O arquivo enviado não é uma imagem utilizável: formato não suportado,
    arquivo corrompido/truncado, conteúdo vazio ou dimensões fora dos
    limites aceitos.

    Vira HTTP 400 na API — é erro do que foi enviado, não falha do sistema.
    """


class CheckpointInvalidoError(IastechError):
    """
    O checkpoint do detector não existe, não pôde ser lido ou não segue o
    formato portátil esperado (ver app/detection/checkpoint.py).

    Vira HTTP 503 na API — o sistema está no ar, mas não tem modelo para
    trabalhar; é problema de instalação/configuração, não do envio.
    """


class ProcessamentoError(IastechError):
    """
    Falha durante o processamento de uma imagem que já passou pela
    validação.

    Vira HTTP 500 na API.
    """
