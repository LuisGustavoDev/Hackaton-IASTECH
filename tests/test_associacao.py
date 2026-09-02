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


def test_texto_mais_proximo_do_centro_vence_o_empate():
    equipamentos = associar(
        [deteccao(0, 0, 200, 200)],
        [
            texto("LONGE", 10, 10, w=20, h=10),
            texto("PERTO", 95, 95, w=20, h=10),
        ],
    )

    assert equipamentos[0]["tag"] == "PERTO"


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
