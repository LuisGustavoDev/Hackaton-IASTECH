"""
Matriz de confusão: casamento por IoU e contagem de VP/FP/FN.

O casamento é o coração do entregável de validação — se ele errar, todos
os números do relatório mentem juntos e de forma plausível.
"""

from __future__ import annotations

import pytest

from app.diagnostico import gabarito
from app.diagnostico.matriz_confusao import SEM_PAR, casar, relatorio


def caixa(classe, x1, y1, x2, y2, score=0.9):
    return {
        "classe": classe,
        "score": score,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }


# ---------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------


def test_iou_de_caixas_identicas_e_um():
    a = caixa("Válvula", 0, 0, 10, 10)

    assert gabarito.iou(a, a) == pytest.approx(1.0)


def test_iou_de_caixas_disjuntas_e_zero():
    assert gabarito.iou(
        caixa("Válvula", 0, 0, 10, 10),
        caixa("Válvula", 50, 50, 60, 60),
    ) == 0.0


def test_iou_de_caixas_que_so_se_tocam_e_zero():
    assert gabarito.iou(
        caixa("Válvula", 0, 0, 10, 10),
        caixa("Válvula", 10, 0, 20, 10),
    ) == 0.0


def test_iou_de_sobreposicao_parcial():
    # Interseção 5x10=50; união 100+100-50=150.
    valor = gabarito.iou(
        caixa("Válvula", 0, 0, 10, 10),
        caixa("Válvula", 5, 0, 15, 10),
    )

    assert valor == pytest.approx(50 / 150)


# ---------------------------------------------------------------------
# Casamento
# ---------------------------------------------------------------------


def test_predicao_certa_no_lugar_certo_vira_par():
    pares, fp, fn = casar(
        [caixa("Válvula", 0, 0, 10, 10)],
        [caixa("Válvula", 0, 0, 10, 10)],
        0.5,
    )

    assert pares == [("Válvula", "Válvula")]
    assert fp == []
    assert fn == []


def test_predicao_sem_anotacao_e_falso_positivo():
    pares, fp, fn = casar([caixa("Válvula", 0, 0, 10, 10)], [], 0.5)

    assert pares == []
    assert fp == ["Válvula"]
    assert fn == []


def test_anotacao_sem_predicao_e_falso_negativo():
    pares, fp, fn = casar([], [caixa("Vaso", 0, 0, 10, 10)], 0.5)

    assert pares == []
    assert fp == []
    assert fn == ["Vaso"]


def test_caixa_certa_com_classe_errada_vira_confusao():
    """
    Não pode virar FP + FN: isso esconderia que o modelo ACHOU o
    equipamento e só errou o rótulo, que é um erro diferente e com
    solução diferente.
    """
    pares, fp, fn = casar(
        [caixa("Outro", 0, 0, 10, 10)],
        [caixa("Válvula", 0, 0, 10, 10)],
        0.5,
    )

    assert pares == [("Válvula", "Outro")]
    assert fp == []
    assert fn == []


def test_sobreposicao_abaixo_do_limiar_nao_casa():
    pares, fp, fn = casar(
        [caixa("Válvula", 0, 0, 10, 10)],
        [caixa("Válvula", 9, 0, 19, 10)],
        0.5,
    )

    assert pares == []
    assert fp == ["Válvula"]
    assert fn == ["Válvula"]


def test_cada_anotacao_e_reivindicada_uma_unica_vez():
    """Duas predições sobre o mesmo equipamento: uma casa, a outra é FP."""
    pares, fp, fn = casar(
        [
            caixa("Válvula", 0, 0, 10, 10, score=0.9),
            caixa("Válvula", 1, 1, 11, 11, score=0.7),
        ],
        [caixa("Válvula", 0, 0, 10, 10)],
        0.5,
    )

    assert len(pares) == 1
    assert fp == ["Válvula"]
    assert fn == []


def test_predicao_mais_confiante_escolhe_primeiro():
    """
    A de score maior fica com a anotação que cobre melhor; a de score
    menor sobra. Sem essa ordem, o resultado dependeria da ordem em que
    as detecções chegaram.
    """
    pares, fp, _ = casar(
        [
            caixa("Bomba", 0, 0, 10, 10, score=0.5),
            caixa("Válvula", 0, 0, 10, 10, score=0.95),
        ],
        [caixa("Válvula", 0, 0, 10, 10)],
        0.5,
    )

    assert pares == [("Válvula", "Válvula")]
    assert fp == ["Bomba"]


def test_varios_equipamentos_casam_com_os_seus():
    pares, fp, fn = casar(
        [
            caixa("Válvula", 0, 0, 10, 10),
            caixa("Bomba", 100, 100, 110, 110),
        ],
        [
            caixa("Bomba", 100, 100, 110, 110),
            caixa("Válvula", 0, 0, 10, 10),
        ],
        0.5,
    )

    assert sorted(pares) == [("Bomba", "Bomba"), ("Válvula", "Válvula")]
    assert fp == []
    assert fn == []


# ---------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------


def test_relatorio_soma_vp_fp_fn_e_trocas(capsys):
    from collections import Counter

    matriz = Counter(
        {
            ("Válvula", "Válvula"): 8,
            ("Vaso", "Vaso"): 1,
            (SEM_PAR, "Outro"): 3,       # 3 falsos positivos
            ("Tanque", SEM_PAR): 2,      # 2 falsos negativos
            ("Válvula", "Outro"): 4,     # 4 classes trocadas
        }
    )

    relatorio(matriz, imagens_medidas=5, ignoradas=[], limiar_iou=0.5)

    saida = capsys.readouterr().out

    assert "VP (acertou classe e posição) : 9" in saida
    assert "FP (previu onde não havia)    : 3" in saida
    assert "FN (deixou de ver)            : 2" in saida
    assert "Classe trocada (casou a caixa): 4" in saida


def test_relatorio_nao_finge_ter_vn(capsys):
    from collections import Counter

    relatorio(
        Counter({("Válvula", "Válvula"): 1}),
        imagens_medidas=1,
        ignoradas=[],
        limiar_iou=0.5,
    )

    assert "não aplicável" in capsys.readouterr().out


def test_relatorio_lista_as_classes_confundidas(capsys):
    from collections import Counter

    relatorio(
        Counter({("Válvula", "Outro"): 7, ("Válvula", "Válvula"): 1}),
        imagens_medidas=1,
        ignoradas=[],
        limiar_iou=0.5,
    )

    saida = capsys.readouterr().out

    assert "CLASSES MAIS CONFUNDIDAS" in saida
    assert "Válvula  ->  Outro" in saida


def test_relatorio_com_matriz_vazia_nao_divide_por_zero(capsys):
    from collections import Counter

    relatorio(Counter(), imagens_medidas=0, ignoradas=[], limiar_iou=0.5)

    saida = capsys.readouterr().out

    assert "Precisão : 0.000" in saida
    assert "F1       : 0.000" in saida


# ---------------------------------------------------------------------
# Gabarito
# ---------------------------------------------------------------------


def test_indexar_encontra_o_xml_pelo_nome_da_imagem():
    """
    A chave precisa ser o nome do arquivo de imagem, porque é o que fica
    gravado em execucoes.arquivo_nome.
    """
    indice = gabarito.indexar("dataset/original/testes")

    if not indice:
        pytest.skip("dataset não presente nesta cópia do repositório")

    assert "101.jpg" in indice
    assert indice["101.jpg"].suffix == ".xml"


def test_caixas_le_as_anotacoes_do_xml():
    indice = gabarito.indexar("dataset/original/testes")

    if not indice:
        pytest.skip("dataset não presente nesta cópia do repositório")

    anotacoes = gabarito.caixas(indice["101.jpg"])

    assert anotacoes
    for anotacao in anotacoes:
        assert anotacao["classe"]
        assert anotacao["x2"] > anotacao["x1"]
        assert anotacao["y2"] > anotacao["y1"]
