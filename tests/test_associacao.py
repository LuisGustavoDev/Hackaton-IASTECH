"""
Casamento entre os textos do OCR e os equipamentos detectados.
"""

from __future__ import annotations

from app.services.associacao import associar


def deteccao(x1, y1, x2, y2, classe="Instrumento"):
    return {
        "classe": classe,
        "score": 0.9,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "centro_x": (x1 + x2) // 2,
        "centro_y": (y1 + y2) // 2,
    }


def texto(conteudo, x, y, w=40, h=12, confianca=90.0):
    return {
        "texto": conteudo,
        "confianca": confianca,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
    }


def test_texto_dentro_da_caixa_vira_tag():
    equipamentos = associar(
        [deteccao(100, 100, 200, 200)],
        [texto("FT-210", 130, 140)],
    )

    assert equipamentos[0]["tag"] == "FT-210"
    assert equipamentos[0]["confianca_ocr"] == 90.0


def test_texto_distante_nao_e_associado():
    equipamentos = associar(
        [deteccao(100, 100, 140, 140)],
        [texto("FT-210", 3000, 3000)],
    )

    assert equipamentos[0]["tag"] == ""
    assert equipamentos[0]["confianca_ocr"] == 0.0


def test_texto_proximo_e_associado_na_segunda_passada():
    # Fora da caixa (100..140), mas dentro do raio relativo.
    equipamentos = associar(
        [deteccao(100, 100, 140, 140)],
        [texto("PI-101", 145, 110, w=30, h=10)],
    )

    assert equipamentos[0]["tag"] == "PI-101"


def test_texto_interno_tem_prioridade_sobre_texto_vizinho():
    """
    O texto de fora está mais PERTO do centro do que o de dentro, mas quem
    está dentro da caixa vence: é a passada 1 que roda primeiro.
    """
    equipamentos = associar(
        [deteccao(0, 0, 200, 200)],
        [
            texto("DENTRO", 150, 150, w=20, h=10),
            texto("FORA", 205, 95, w=20, h=10),
        ],
    )

    assert equipamentos[0]["tag"] == "DENTRO"


def test_cada_texto_serve_a_um_unico_equipamento():
    equipamentos = associar(
        [
            deteccao(0, 0, 100, 100),
            deteccao(100, 0, 200, 100),
        ],
        [
            texto("FT-210", 40, 40, w=20, h=10),
            texto("PI-101", 140, 40, w=20, h=10),
        ],
    )

    tags = {e["tag"] for e in equipamentos}

    assert tags == {"FT-210", "PI-101"}


def test_balao_isa_de_duas_linhas_vira_uma_tag_so():
    """
    Um balão de instrumento traz as letras em cima e o número embaixo.
    Pegar só uma das linhas destrói a TAG: "0013" sozinho não permite
    deduzir nem a Descrição nem o Grupo.
    """
    equipamentos = associar(
        [deteccao(100, 100, 220, 220)],
        [
            texto("PI", 145, 130, w=30, h=18),
            texto("0013", 130, 170, w=60, h=18),
        ],
    )

    assert equipamentos[0]["tag"] == "PI 0013"


def test_fragmentos_da_mesma_linha_saem_da_esquerda_para_a_direita():
    equipamentos = associar(
        [deteccao(0, 0, 200, 200)],
        [
            texto("210", 120, 98, w=30, h=12),
            texto("FT", 60, 100, w=25, h=12),
        ],
    )

    assert equipamentos[0]["tag"] == "FT 210"


def test_todos_os_textos_internos_entram_na_tag():
    equipamentos = associar(
        [deteccao(0, 0, 200, 300)],
        [
            texto("A", 90, 30, w=20, h=15),
            texto("B", 90, 130, w=20, h=15),
            texto("C", 90, 230, w=20, h=15),
        ],
    )

    assert equipamentos[0]["tag"] == "A B C"


def test_confianca_e_a_media_dos_textos_usados():
    equipamentos = associar(
        [deteccao(0, 0, 200, 200)],
        [
            texto("PI", 90, 60, w=20, h=15, confianca=80.0),
            texto("101", 90, 120, w=30, h=15, confianca=60.0),
        ],
    )

    assert equipamentos[0]["confianca_ocr"] == 70.0


def test_texto_de_baixa_confianca_e_descartado():
    """
    O MSER acha traços do desenho da válvula e o Tesseract devolve letras
    a partir deles. Esse ruído vem com confiança baixa e não pode ser
    colado no meio de uma TAG legítima.
    """
    equipamentos = associar(
        [deteccao(0, 0, 200, 200)],
        [
            texto("PI", 90, 60, w=20, h=15, confianca=95.0),
            texto("(X)", 90, 120, w=20, h=15, confianca=8.0),
        ],
    )

    assert equipamentos[0]["tag"] == "PI"


def test_texto_em_caixas_aninhadas_fica_com_a_de_centro_mais_proximo():
    equipamentos = associar(
        [
            deteccao(0, 0, 400, 400, classe="Tanque"),
            deteccao(80, 80, 140, 140, classe="Instrumento"),
        ],
        [texto("FT-210", 100, 104, w=20, h=12)],
    )

    tags = {e["classe"]: e["tag"] for e in equipamentos}

    assert tags["Instrumento"] == "FT-210"
    assert tags["Tanque"] == ""


def test_vizinho_externo_traz_um_texto_so():
    """
    Dentro da caixa dá para presumir que os textos são do mesmo
    equipamento. Fora dela, não — juntar dois vizinhos seria chute.
    """
    # Os dois estão fora da caixa e dentro do raio; PI-101 é o mais
    # próximo do centro (32px contra 39px).
    equipamentos = associar(
        [deteccao(100, 100, 140, 140)],
        [
            texto("PI-101", 142, 112, w=20, h=10),
            texto("2\"", 145, 132, w=20, h=10),
        ],
    )

    assert equipamentos[0]["tag"] == "PI-101"


def test_textos_em_branco_sao_ignorados():
    equipamentos = associar(
        [deteccao(0, 0, 100, 100)],
        [texto("   ", 40, 40), texto("TI-101", 45, 45)],
    )

    assert equipamentos[0]["tag"] == "TI-101"


def test_sem_textos_todas_as_tags_ficam_vazias():
    equipamentos = associar([deteccao(0, 0, 100, 100)], [])

    assert equipamentos[0]["tag"] == ""


def test_deteccoes_originais_nao_sao_modificadas():
    original = deteccao(0, 0, 100, 100)

    associar([original], [texto("FT-210", 40, 40)])

    assert "tag" not in original
