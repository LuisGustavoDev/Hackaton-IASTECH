"""
Interpretação da TAG lida pelo OCR.

Preenche duas das seis colunas da planilha, seguindo o que ficou decidido
na reunião de 14/08/2026 com Ricardo Sandrini (IASTECH):

- **Descrição**: decomposição das letras da TAG pela norma ISA-5.1
  (FT-210 -> "Transmissor de Vazão"; PI-101 -> "Indicador de Pressão").
  O dicionário de letras vem da tabela `isa_letras`, populada por
  app/models/isa_seed.py.

- **Grupo**: o primeiro dígito do número da TAG. Da ata: "o primeiro
  número da tag geralmente indica a qual conjunto/equipamento (ex.:
  reator 1, reator 2) o item pertence". FT-210 -> grupo "2".

Quando a TAG não é lida ou as letras não constam na tabela ISA, a
Descrição cai para o nome da classe detectada pelo Faster R-CNN — assim a
linha nunca sai sem informação de tipo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.models.database import Database
from app.models.isa_seed import CategoriasISA, popular_isa_letras

# Duas letras + número é o padrão descrito na ata, mas o dataset também
# traz tags de 1 letra (V-12) e de 3 (PIT-101), com ou sem separador.
_PADRAO_TAG = re.compile(r"([A-Za-z]{1,4})\s*[-_/.]?\s*(\d{1,6})")

# Como cada função ISA é ESCRITA na descrição.
#
# A tabela isa_letras guarda a ação ("Transmitir", "Indicar"); a coluna
# Descrição pede o agente ("Transmissor de Vazão", "Indicador de
# Pressão") — é a forma usada nos exemplos que o Ricardo passou. Este
# mapa é só apresentação: não altera nem substitui a norma, e letras que
# não estiverem aqui usam o texto da tabela como veio.
_FUNCAO_COMO_SUBSTANTIVO = {
    "Transmitir": "Transmissor",
    "Indicar": "Indicador",
    "Controle": "Controlador",
    "Registrar": "Registrador",
    "Integrar, totalizar": "Totalizador",
    "Chave/Comutador": "Chave",
    "Sensor, elemento primário": "Sensor",
    "Visor, indicador de vidro": "Visor",
    "Orifício, restrição": "Orifício",
    "Luz/Indicador luminoso": "Indicador luminoso",
    "Poço (termopar)": "Poço",
    "Válvula, registro, veneziana": "Válvula",
    "Acionador, atuador, elemento final de controle": "Atuador",
    "Dispositivos auxiliares": "Dispositivo auxiliar",
    "Estação de controle": "Estação de controle",
}

# Significados que não acrescentam nada à descrição.
_SEM_CONTEUDO = {"Livre escolha", "Não classificado"}

_ORDEM_FUNCOES = (
    CategoriasISA.FUNCAO_LEITURA,
    CategoriasISA.FUNCAO_SAIDA,
)


@dataclass(frozen=True)
class AnaliseTag:
    """Resultado da leitura de uma TAG."""

    tag: str
    letras: str
    numero: str
    grupo: str
    descricao: str


_dicionario_isa: dict[str, dict[str, str]] | None = None


def obter_dicionario_isa(
    db: Database | None = None,
) -> dict[str, dict[str, str]]:
    """
    Carrega a tabela ISA-5.1 no formato {letra: {categoria: significado}}.

    Popula o banco na primeira chamada (popular_isa_letras é idempotente)
    e guarda o resultado em memória: são ~90 linhas fixas, relê-las a cada
    equipamento de cada diagrama não faz sentido.
    """
    global _dicionario_isa

    if _dicionario_isa is not None and db is None:
        return _dicionario_isa

    popular_isa_letras(db)

    from app.models.database import get_db

    banco = db or get_db()

    dicionario: dict[str, dict[str, str]] = {}

    with banco.cursor() as cursor:
        cursor.execute("SELECT letra, categoria, significado FROM isa_letras")

        for letra, categoria, significado in cursor.fetchall():
            dicionario.setdefault(letra.upper(), {})[categoria] = significado

    if db is None:
        _dicionario_isa = dicionario

    return dicionario


def redefinir_dicionario_isa() -> None:
    """Descarta o dicionário em cache (usado pelos testes)."""
    global _dicionario_isa
    _dicionario_isa = None


def analisar(texto: str | None, tipo_detectado: str = "") -> AnaliseTag:
    """
    Interpreta o texto lido pelo OCR como uma TAG ISA.

    `tipo_detectado` é o nome da classe do Faster R-CNN, usado como
    descrição quando a TAG não permite chegar a nada melhor.
    """
    texto = (texto or "").strip()

    correspondencia = _PADRAO_TAG.search(texto)

    if correspondencia is None:
        return AnaliseTag(
            tag=texto,
            letras="",
            numero="",
            grupo="",
            descricao=tipo_detectado,
        )

    letras = correspondencia.group(1).upper()
    numero = correspondencia.group(2)

    descricao = _montar_descricao(letras) or tipo_detectado

    return AnaliseTag(
        tag=texto,
        letras=letras,
        numero=numero,
        grupo=numero[0],
        descricao=descricao,
    )


def _montar_descricao(letras: str) -> str:
    """
    Monta a descrição a partir das letras da TAG.

    Estrutura da norma: a 1ª letra é a variável medida; a 2ª pode ser um
    modificador dessa variável (PDT: P=Pressão, D=Diferencial); as demais
    são funções (leitura/saída) e modificadores de função (H=Alto).
    """
    isa = obter_dicionario_isa()

    primeira = letras[0]
    restantes = list(letras[1:])

    variavel = _significado(
        isa, primeira, CategoriasISA.VARIAVEL_MEDIDA
    )

    # A 2ª letra só é lida como modificador da variável quando ainda sobra
    # letra para ser a função — em "PD" o D é a função, em "PDT" é o
    # modificador.
    modificador_variavel = ""

    if len(restantes) >= 2:
        candidato = _significado(
            isa, restantes[0], CategoriasISA.MODIFICADOR_VARIAVEL
        )
        if candidato:
            modificador_variavel = candidato
            restantes.pop(0)

    funcoes = []
    modificadores_funcao = []

    for letra in restantes:
        funcao = ""

        for categoria in _ORDEM_FUNCOES:
            funcao = _significado(isa, letra, categoria)
            if funcao:
                break

        if funcao:
            funcoes.append(
                _FUNCAO_COMO_SUBSTANTIVO.get(funcao, funcao)
            )
            continue

        modificador = _significado(
            isa, letra, CategoriasISA.MODIFICADOR_FUNCAO
        )

        if modificador:
            modificadores_funcao.append(modificador)

    return _escrever(
        funcoes, variavel, modificador_variavel, modificadores_funcao
    )


def _escrever(
    funcoes: list[str],
    variavel: str,
    modificador_variavel: str,
    modificadores_funcao: list[str],
) -> str:
    if modificador_variavel and variavel:
        variavel = f"{variavel} {modificador_variavel}"
    elif modificador_variavel:
        variavel = modificador_variavel

    if funcoes and variavel:
        descricao = f"{' e '.join(funcoes)} de {variavel}"
    elif funcoes:
        descricao = " e ".join(funcoes)
    else:
        descricao = variavel

    if descricao and modificadores_funcao:
        descricao = f"{descricao} ({', '.join(modificadores_funcao)})"

    return descricao


def _significado(
    isa: dict[str, dict[str, str]],
    letra: str,
    categoria: str,
) -> str:
    significado = isa.get(letra.upper(), {}).get(categoria, "")

    if significado in _SEM_CONTEUDO:
        return ""

    return significado
