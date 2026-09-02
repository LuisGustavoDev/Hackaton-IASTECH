"""
Persistência das execuções.

Base da matriz de confusão (VP, FP, VN, FN) e dos indicadores da Etapa 05:
sem guardar o que o modelo previu em cada rodada, não há como comparar
duas versões do checkpoint.
"""

from __future__ import annotations

import pytest

from app.models import execucoes
from app.models.database import Database


@pytest.fixture
def db(tmp_path):
    banco = Database(str(tmp_path / "teste.db"))
    yield banco
    banco.close()


def equipamento(
    classe="Válvula",
    tag_normalizada="",
    descricao="Válvula",
    grupo="",
    centro=(50, 60),
):
    return {
        "classe": classe,
        "score": 0.87,
        "x1": 10,
        "y1": 20,
        "x2": 90,
        "y2": 100,
        "centro_x": centro[0],
        "centro_y": centro[1],
        "tag": "PI 0013",
        "tag_normalizada": tag_normalizada,
        "confianca_ocr": 89.5,
        "descricao": descricao,
        "grupo": grupo,
    }


def test_registrar_devolve_o_id_da_execucao(db):
    execucao_id = execucoes.registrar(
        imagem_largura=1030,
        imagem_altura=748,
        equipamentos=[equipamento()],
        db=db,
    )

    assert isinstance(execucao_id, int)
    assert execucao_id > 0


def test_execucao_guarda_o_contexto_da_rodada(db):
    """
    Checkpoint e limiar precisam ficar gravados: a pergunta que estas
    tabelas respondem é comparativa ("250 épocas ficaram melhores que
    15?"), e sem saber qual modelo gerou cada linha os números de duas
    rodadas não são comparáveis.
    """
    execucao_id = execucoes.registrar(
        imagem_largura=1030,
        imagem_altura=748,
        equipamentos=[equipamento()],
        arquivo_nome="101.jpg",
        checkpoint="data/models/faster_rcnn_250.pt",
        limiar=0.5,
        tempo_deteccao_ms=2800.0,
        tempo_ocr_ms=1200.0,
        tempo_total_ms=4300.0,
        db=db,
    )

    resumo = execucoes.obter(execucao_id, db=db)

    assert resumo.arquivo_nome == "101.jpg"
    assert resumo.checkpoint == "data/models/faster_rcnn_250.pt"
    assert resumo.limiar == 0.5
    assert resumo.imagem_largura == 1030
    assert resumo.imagem_altura == 748
    assert resumo.tempo_deteccao_ms == 2800.0
    assert resumo.tempo_ocr_ms == 1200.0
    assert resumo.tempo_total_ms == 4300.0


def test_qtd_tags_lidas_conta_so_as_tags_validas(db):
    """
    Insumo da "taxa de acerto do OCR" da Etapa 05: quantos equipamentos
    saíram com TAG contra quantos foram detectados.
    """
    execucao_id = execucoes.registrar(
        imagem_largura=100,
        imagem_altura=100,
        equipamentos=[
            equipamento(tag_normalizada="PI-0013", centro=(10, 10)),
            equipamento(tag_normalizada="", centro=(20, 20)),
            equipamento(tag_normalizada="FT-210", centro=(30, 30)),
        ],
        db=db,
    )

    resumo = execucoes.obter(execucao_id, db=db)

    assert resumo.qtd_deteccoes == 3
    assert resumo.qtd_tags_lidas == 2


def test_deteccoes_guardam_caixa_classe_e_tag(db):
    execucao_id = execucoes.registrar(
        imagem_largura=100,
        imagem_altura=100,
        equipamentos=[
            equipamento(
                classe="Instrumento",
                tag_normalizada="PI-0013",
                descricao="Indicador de Pressão",
                grupo="0",
            )
        ],
        db=db,
    )

    (deteccao,) = execucoes.deteccoes_de(execucao_id, db=db)

    assert deteccao["classe"] == "Instrumento"
    assert deteccao["tag"] == "PI-0013"
    assert deteccao["descricao"] == "Indicador de Pressão"
    assert deteccao["grupo"] == "0"
    assert (deteccao["x1"], deteccao["y1"]) == (10, 20)
    assert (deteccao["x2"], deteccao["y2"]) == (90, 100)


def test_texto_bruto_do_ocr_e_preservado(db):
    """
    A TAG entregue ao cliente é filtrada pelo padrão ISA. O que o OCR leu
    de fato fica guardado à parte: é o que permite diagnosticar depois se
    um falso negativo veio do detector ou do OCR.
    """
    execucao_id = execucoes.registrar(
        imagem_largura=100,
        imagem_altura=100,
        equipamentos=[equipamento(tag_normalizada="PI-0013")],
        db=db,
    )

    (deteccao,) = execucoes.deteccoes_de(execucao_id, db=db)

    assert deteccao["tag"] == "PI-0013"
    assert deteccao["texto_bruto"] == "PI 0013"


def test_execucao_sem_deteccoes_e_registrada(db):
    execucao_id = execucoes.registrar(
        imagem_largura=100, imagem_altura=100, equipamentos=[], db=db
    )

    assert execucoes.obter(execucao_id, db=db).qtd_deteccoes == 0
    assert execucoes.deteccoes_de(execucao_id, db=db) == []


def test_listar_traz_as_mais_recentes_primeiro(db):
    for nome in ("a.jpg", "b.jpg", "c.jpg"):
        execucoes.registrar(
            imagem_largura=100,
            imagem_altura=100,
            equipamentos=[],
            arquivo_nome=nome,
            db=db,
        )

    nomes = [e.arquivo_nome for e in execucoes.listar(db=db)]

    assert nomes == ["c.jpg", "b.jpg", "a.jpg"]


def test_listar_respeita_o_limite(db):
    for _ in range(5):
        execucoes.registrar(
            imagem_largura=100, imagem_altura=100, equipamentos=[], db=db
        )

    assert len(execucoes.listar(limite=2, db=db)) == 2


def test_obter_execucao_inexistente_devolve_none(db):
    assert execucoes.obter(999, db=db) is None


def test_duas_rodadas_do_mesmo_arquivo_sao_comparaveis(db):
    """O caso de uso central: mesma imagem, dois checkpoints."""
    for checkpoint, quantos in (("15_epocas.pt", 1), ("250_epocas.pt", 3)):
        execucoes.registrar(
            imagem_largura=1030,
            imagem_altura=748,
            equipamentos=[
                equipamento(centro=(i, i)) for i in range(quantos)
            ],
            arquivo_nome="101.jpg",
            checkpoint=checkpoint,
            db=db,
        )

    por_checkpoint = {
        e.checkpoint: e.qtd_deteccoes for e in execucoes.listar(db=db)
    }

    assert por_checkpoint == {"15_epocas.pt": 1, "250_epocas.pt": 3}


def test_persistencia_sobrevive_a_reabertura_do_banco(tmp_path):
    """
    Com DB_PATH apontando para arquivo, os dados precisam continuar lá
    depois do processo cair — é o que diferencia isto do :memory:.
    """
    caminho = str(tmp_path / "persistente.db")

    with Database(caminho) as banco:
        execucoes.registrar(
            imagem_largura=100,
            imagem_altura=100,
            equipamentos=[equipamento()],
            arquivo_nome="101.jpg",
            db=banco,
        )

    with Database(caminho) as reaberto:
        assert len(execucoes.listar(db=reaberto)) == 1
        assert execucoes.listar(db=reaberto)[0].arquivo_nome == "101.jpg"


def test_banco_antigo_ganha_a_coluna_nova_sem_perder_dados(tmp_path):
    """
    CREATE TABLE IF NOT EXISTS não altera tabela existente. Como DB_PATH
    aponta para arquivo em produção, um banco já povoado é a regra: ele
    precisa ganhar a coluna nova por ALTER TABLE, preservando o histórico.
    """
    import sqlite3

    caminho = tmp_path / "antigo.db"

    # Simula o schema anterior: execucoes sem a coluna `pasta`.
    con = sqlite3.connect(caminho)
    con.executescript(
        """
        CREATE TABLE execucoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            criado_em TEXT NOT NULL,
            arquivo_nome TEXT,
            imagem_largura INTEGER NOT NULL,
            imagem_altura INTEGER NOT NULL,
            checkpoint TEXT,
            limiar REAL,
            qtd_deteccoes INTEGER NOT NULL,
            qtd_tags_lidas INTEGER NOT NULL,
            tempo_deteccao_ms REAL,
            tempo_ocr_ms REAL,
            tempo_total_ms REAL
        );
        INSERT INTO execucoes (criado_em, arquivo_nome, imagem_largura,
                               imagem_altura, qtd_deteccoes, qtd_tags_lidas)
        VALUES ('2026-09-01T00:00:00Z', 'antiga.jpg', 100, 100, 5, 1);
        """
    )
    con.commit()
    con.close()

    with Database(str(caminho)) as banco:
        colunas = {
            linha[1]
            for linha in banco.conn.execute("PRAGMA table_info(execucoes)")
        }
        assert "pasta" in colunas

        antigas = execucoes.listar(db=banco)
        assert len(antigas) == 1
        assert antigas[0].arquivo_nome == "antiga.jpg"
        assert antigas[0].pasta is None

        novo_id = execucoes.registrar(
            imagem_largura=100,
            imagem_altura=100,
            equipamentos=[],
            arquivo_nome="nova.jpg",
            pasta="data/output/abc",
            db=banco,
        )

        assert execucoes.obter(novo_id, db=banco).pasta == "data/output/abc"
        assert len(execucoes.listar(db=banco)) == 2


def test_migracao_e_idempotente(tmp_path):
    """Abrir o mesmo banco duas vezes não pode tentar recriar a coluna."""
    caminho = str(tmp_path / "repetido.db")

    with Database(caminho) as banco:
        execucoes.registrar(
            imagem_largura=10,
            imagem_altura=10,
            equipamentos=[],
            pasta="x",
            db=banco,
        )

    with Database(caminho) as reaberto:
        assert execucoes.listar(db=reaberto)[0].pasta == "x"
