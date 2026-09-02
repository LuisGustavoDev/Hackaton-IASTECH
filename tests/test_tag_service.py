"""
Interpretação da TAG: coluna Descrição (ISA-5.1) e coluna Grupo.

Os exemplos abaixo são os que Ricardo Sandrini apresentou na reunião de
14/08/2026 e que constam na ata.
"""

from __future__ import annotations

import pytest

from app.services import tag_service


@pytest.mark.parametrize(
    "texto, descricao_esperada",
    [
        ("FT-210", "Transmissor de Vazão"),
        ("TI-101", "Indicador de Temperatura"),
        ("PI-101", "Indicador de Pressão"),
        ("TT-305", "Transmissor de Temperatura"),
        ("WT-402", "Transmissor de Peso, força"),
        ("LC-110", "Controlador de Nível"),
    ],
)
def test_descricao_vem_da_decomposicao_isa(texto, descricao_esperada):
    assert tag_service.analisar(texto).descricao == descricao_esperada


def test_tres_letras_combinam_as_funcoes():
    # PIT = P(Pressão) + I(Indicar) + T(Transmitir)
    analise = tag_service.analisar("PIT-101")

    assert analise.descricao == "Indicador e Transmissor de Pressão"


def test_segunda_letra_pode_modificar_a_variavel():
    # PDT = P(Pressão) + D(Diferencial) + T(Transmitir)
    analise = tag_service.analisar("PDT-220")

    assert analise.descricao == "Transmissor de Pressão Diferencial, desvio"


@pytest.mark.parametrize(
    "texto, grupo_esperado",
    [
        ("FT-210", "2"),
        ("PI-101", "1"),
        ("TT305", "3"),
        ("XV 203", "2"),
        ("LC-9", "9"),
        # Zero à esquerda é largura de escrita, não grupo.
        ("PI-0013", "1"),
        ("FT 007", "7"),
        ("XV-0000", "0"),
    ],
)
def test_grupo_e_o_primeiro_digito_do_numero(texto, grupo_esperado):
    """
    Da ata: "o primeiro número da tag geralmente indica a qual
    conjunto/equipamento (ex.: reator 1, reator 2) o item pertence".
    """
    assert tag_service.analisar(texto).grupo == grupo_esperado


@pytest.mark.parametrize("texto", ["FT-210", "FT210", "FT 210", "ft.210"])
def test_separador_entre_letras_e_numero_e_opcional(texto):
    analise = tag_service.analisar(texto)

    assert analise.letras == "FT"
    assert analise.numero == "210"


def test_texto_sem_tag_cai_para_a_classe_detectada():
    analise = tag_service.analisar("ruido do ocr", tipo_detectado="Bomba")

    assert analise.descricao == "Bomba"
    assert analise.grupo == ""
    assert analise.letras == ""


def test_texto_vazio_nao_quebra():
    analise = tag_service.analisar("", tipo_detectado="Válvula")

    assert analise.tag == ""
    assert analise.descricao == "Válvula"
    assert analise.grupo == ""


def test_tag_none_nao_quebra():
    assert tag_service.analisar(None, tipo_detectado="Tanque").descricao == "Tanque"


def test_letra_sem_significado_util_cai_para_a_classe():
    # N é "Livre escolha" em todas as categorias — não descreve nada.
    analise = tag_service.analisar("NN-100", tipo_detectado="Outro")

    assert analise.descricao == "Outro"
    assert analise.grupo == "1"


def test_dicionario_isa_traz_as_letras_da_norma():
    isa = tag_service.obter_dicionario_isa()

    assert isa["F"]["variavel_medida"] == "Vazão"
    assert isa["T"]["funcao_saida"] == "Transmitir"
    assert len(isa) == 26


# ---------------------------------------------------------------------
# Filtro da coluna TAG
#
# O Tesseract produz "letras" a partir do traço dos próprios símbolos.
# Esses valores foram observados numa saída real do sistema e não podem
# chegar à planilha do cliente disfarçados de identificador.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("ruido", ["(X)", "Oo", "SH", "ro.", "y", "|", "---"])
def test_ruido_do_ocr_nao_vira_tag(ruido):
    analise = tag_service.analisar(ruido, tipo_detectado="Válvula")

    assert analise.tag == ""
    assert analise.texto_bruto == ruido
    assert analise.descricao == "Válvula"


@pytest.mark.parametrize(
    "texto, tag_esperada",
    [
        ("PI 0013", "PI-0013"),
        ("PI-0013", "PI-0013"),
        ("PI0013", "PI-0013"),
        ("ft 210", "FT-210"),
    ],
)
def test_tag_e_normalizada_para_letras_hifen_numero(texto, tag_esperada):
    """
    O mesmo instrumento aparece escrito de formas diferentes conforme o
    desenho e o OCR; a planilha precisa de uma forma só para o cliente
    ordenar e filtrar.
    """
    assert tag_service.analisar(texto).tag == tag_esperada


def test_texto_bruto_preserva_o_que_o_ocr_leu():
    analise = tag_service.analisar("PI 0013")

    assert analise.tag == "PI-0013"
    assert analise.texto_bruto == "PI 0013"


def test_numero_sozinho_nao_vira_tag():
    """
    Era o sintoma do balão ISA lido pela metade: "0013" sem o "PI" não
    permite deduzir Descrição nem Grupo. A correção está na associação
    (juntar as duas linhas); aqui só se garante que o meio-resultado não
    vaza para a planilha como se fosse uma TAG.
    """
    analise = tag_service.analisar("0013", tipo_detectado="Instrumento")

    assert analise.tag == ""
    assert analise.grupo == ""
    assert analise.descricao == "Instrumento"


def test_balao_isa_completo_produz_descricao_e_grupo():
    analise = tag_service.analisar("PI 0013", tipo_detectado="Instrumento")

    assert analise.tag == "PI-0013"
    assert analise.descricao == "Indicador de Pressão"
    assert analise.grupo == "1"
