"""
Formato do checkpoint portátil do detector.

O treino roda numa máquina com GPU (desktop, CUDA) e a inferência num
notebook sem GPU. Para que APENAS o arquivo .pt precise ser copiado entre
as duas, ele carrega, além dos pesos, tudo que é necessário para
reconstruir o modelo e traduzir as predições em nomes de equipamento:

    {
        "formato_versao": 1,
        "arquitetura":    "fasterrcnn_resnet50_fpn_v2",
        "classes":        ["Acumulador", ..., "Válvula"],  # ordem canônica
        "num_classes":    24,        # sem contar o background
        "state_dict":     {...},     # pesos
        "metadados":      {...},     # informativo (data, épocas, mAP...)
    }

Por que `classes` e não os ids do json de anotação: ferramentas de
exportação COCO numeram a MESMA classe com ids diferentes em cada split
(no nosso dataset, "Bomba" é id 3 no train.json e id 1 no val.json). A
lista canônica vem de build_canonical_categories(), que casa as
categorias pelo NOME. Guardar os nomes aqui — e não uma referência ao
dataset — é justamente o que permite jogar o dataset fora depois do
treino.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from app.core.erros import CheckpointInvalidoError
from app.detection.modelo import ARQUITETURA

FORMATO_VERSAO = 1

_CHAVES_OBRIGATORIAS = (
    "formato_versao",
    "arquitetura",
    "classes",
    "num_classes",
    "state_dict",
)


def salvar_checkpoint(
    caminho: str | Path,
    modelo,
    classes: list[str],
    metadados: dict[str, Any] | None = None,
) -> Path:
    """
    Grava o checkpoint portátil.

    `classes` precisa estar na MESMA ordem usada durante o treino: é ela
    que define o índice de cada classe, e portanto o label que o modelo
    aprendeu (índice + 1).
    """
    if not classes:
        raise ValueError("A lista de classes não pode ser vazia.")

    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    conteudo = {
        "formato_versao": FORMATO_VERSAO,
        "arquitetura": ARQUITETURA,
        "classes": list(classes),
        "num_classes": len(classes),
        # .cpu() aqui para que um checkpoint treinado na GPU não guarde
        # tensores presos ao device cuda:0 — sem isso, carregar na máquina
        # sem CUDA depende de map_location e falha se alguém esquecer.
        "state_dict": {
            chave: valor.cpu()
            for chave, valor in modelo.state_dict().items()
        },
        "metadados": _sanitizar(metadados or {}),
    }

    torch.save(conteudo, caminho)

    return caminho


def _sanitizar(valor: Any) -> Any:
    """
    Reduz os metadados a tipos primitivos.

    O checkpoint é lido com weights_only=True, que só aceita tipos
    básicos. Um valor aparentemente inocente quebra a leitura: o
    `torch.__version__` que o treino grava não é uma str, é um
    TorchVersion, e o torch.load recusa o arquivo inteiro por causa dele.
    Como os metadados são informativos, converter o que não for primitivo
    para texto é preferível a produzir um checkpoint que não carrega.

    A comparação é por tipo EXATO, não isinstance: TorchVersion é uma
    subclasse de str, passaria por um isinstance(valor, str) e continuaria
    sendo serializada como TorchVersion.
    """
    if isinstance(valor, dict):
        return {str(chave): _sanitizar(item) for chave, item in valor.items()}

    if isinstance(valor, (list, tuple)):
        return [_sanitizar(item) for item in valor]

    if valor is None or type(valor) in (str, int, float, bool):
        return valor

    return str(valor)


def carregar_checkpoint(caminho: str | Path) -> dict[str, Any]:
    """
    Lê e valida o checkpoint portátil, sempre em CPU.

    map_location="cpu" é obrigatório: o checkpoint pode ter sido gerado
    numa máquina com CUDA, e sem isso o torch tentaria alocar os tensores
    numa GPU que não existe no notebook de produção.
    """
    caminho = Path(caminho)

    if not caminho.is_file():
        raise CheckpointInvalidoError(
            f"Checkpoint do detector não encontrado: {caminho}. "
            f"Copie o arquivo .pt gerado pelo treino para esse caminho ou "
            f"aponte a variável de ambiente DETECTOR_CHECKPOINT_PATH para "
            f"onde ele está."
        )

    try:
        dados = torch.load(caminho, map_location="cpu", weights_only=True)
    except Exception as erro:
        raise CheckpointInvalidoError(
            f"Não foi possível ler o checkpoint {caminho}: {erro}"
        ) from erro

    _validar(dados, caminho)

    return dados


def _validar(dados: Any, origem: Path) -> None:
    if not isinstance(dados, dict):
        raise CheckpointInvalidoError(
            f"O arquivo {origem} não está no formato portátil do projeto "
            f"(esperado um dicionário, encontrado {type(dados).__name__}). "
            f"Se for um state_dict puro — como o gerado pelo "
            f"benchmark_pid_models.py — converta com "
            f"'python -m app.training.empacotar_checkpoint'."
        )

    faltando = [c for c in _CHAVES_OBRIGATORIAS if c not in dados]

    if faltando:
        raise CheckpointInvalidoError(
            f"Checkpoint {origem} incompleto: faltam as chaves "
            f"{', '.join(faltando)}. "
            f"Se ele veio do benchmark, converta com "
            f"'python -m app.training.empacotar_checkpoint'."
        )

    if dados["formato_versao"] != FORMATO_VERSAO:
        raise CheckpointInvalidoError(
            f"Checkpoint {origem} está no formato versão "
            f"{dados['formato_versao']}, mas esta versão do sistema lê o "
            f"formato {FORMATO_VERSAO}."
        )

    if dados["arquitetura"] != ARQUITETURA:
        raise CheckpointInvalidoError(
            f"Checkpoint {origem} foi gerado para a arquitetura "
            f"'{dados['arquitetura']}', mas a produção usa '{ARQUITETURA}'."
        )

    classes = dados["classes"]

    if not isinstance(classes, list) or not classes:
        raise CheckpointInvalidoError(
            f"Checkpoint {origem} não traz uma lista de classes válida."
        )

    if dados["num_classes"] != len(classes):
        raise CheckpointInvalidoError(
            f"Checkpoint {origem} inconsistente: num_classes="
            f"{dados['num_classes']} mas a lista tem {len(classes)} classes."
        )
