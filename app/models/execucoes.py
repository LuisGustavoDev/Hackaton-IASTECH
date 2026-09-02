"""
Persistência das execuções do pipeline.

Guarda, a cada imagem processada, o que o modelo previu e quanto tempo
levou. A planilha em data/output responde "o que tem neste diagrama?";
estas tabelas respondem "o modelo está melhorando?", que é uma pergunta
que exige comparar rodadas e por isso não sobrevive num arquivo que é
regravado a cada execução.

O destino disso é a matriz de confusão (VP, FP, VN, FN) pedida na reunião
de 14/08/2026 e os indicadores da Etapa 05 do Plano de Desenvolvimento. O
gabarito NÃO é duplicado aqui: os .xml em dataset/original já estão
versionados, e a matriz sai do cruzamento entre eles e a tabela
`deteccoes`, casando por IoU e classe.

> Sobre o "VN" da matriz: em detecção de objetos não existe verdadeiro
> negativo natural — não há um conjunto finito de "caixas que poderiam
> ter sido detectadas e corretamente não foram". VP, FP e FN saem direto
> do cruzamento; o VN precisa de uma definição de negócio (por exemplo,
> classes do catálogo ausentes no diagrama e corretamente não previstas).
> Fica registrado aqui para não ser inventado na hora de calcular.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from app.models.database import Database, get_db


@dataclass(frozen=True)
class ResumoExecucao:
    """Uma linha da tabela `execucoes`, sem as detecções."""

    id: int
    criado_em: str
    arquivo_nome: str | None
    imagem_largura: int
    imagem_altura: int
    checkpoint: str | None
    limiar: float | None
    qtd_deteccoes: int
    qtd_tags_lidas: int
    tempo_deteccao_ms: float | None
    tempo_ocr_ms: float | None
    tempo_total_ms: float | None
    pasta: str | None = None


def registrar(
    *,
    imagem_largura: int,
    imagem_altura: int,
    equipamentos: list[dict],
    arquivo_nome: str | None = None,
    checkpoint: str | None = None,
    limiar: float | None = None,
    tempo_deteccao_ms: float | None = None,
    tempo_ocr_ms: float | None = None,
    tempo_total_ms: float | None = None,
    pasta: str | None = None,
    db: Database | None = None,
) -> int:
    """
    Grava uma execução e suas detecções. Devolve o id da execução.

    `equipamentos` são as detecções já enriquecidas por
    app/services/processamento.py:montar_linhas() — cada uma carrega,
    além da caixa e da classe, a TAG normalizada, a Descrição e o Grupo
    que foram para a planilha. Ler tudo do mesmo dicionário evita ter de
    reconciliar detecção com linha da planilha depois, que é onde
    apareceria divergência silenciosa entre o que foi entregue ao cliente
    e o que foi medido.

    Tudo numa transação só: uma execução gravada pela metade poluiria a
    matriz de confusão com falsos negativos que nunca existiram.
    """
    banco = db or get_db()

    qtd_tags = sum(1 for e in equipamentos if e.get("tag_normalizada"))

    with banco.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO execucoes (
                criado_em, arquivo_nome, imagem_largura, imagem_altura,
                checkpoint, limiar, qtd_deteccoes, qtd_tags_lidas,
                tempo_deteccao_ms, tempo_ocr_ms, tempo_total_ms, pasta
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                arquivo_nome,
                imagem_largura,
                imagem_altura,
                checkpoint,
                limiar,
                len(equipamentos),
                qtd_tags,
                tempo_deteccao_ms,
                tempo_ocr_ms,
                tempo_total_ms,
                pasta,
            ),
        )

        execucao_id = cursor.lastrowid

        cursor.executemany(
            """
            INSERT INTO deteccoes (
                execucao_id, classe, score, x1, y1, x2, y2,
                centro_x, centro_y, tag, texto_bruto, confianca_ocr,
                descricao, grupo
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    execucao_id,
                    e["classe"],
                    float(e["score"]),
                    e["x1"],
                    e["y1"],
                    e["x2"],
                    e["y2"],
                    e["centro_x"],
                    e["centro_y"],
                    e.get("tag_normalizada", ""),
                    e.get("tag", ""),
                    float(e.get("confianca_ocr", 0.0)),
                    e.get("descricao", ""),
                    e.get("grupo", ""),
                )
                for e in equipamentos
            ],
        )

    return execucao_id


def listar(limite: int = 50, db: Database | None = None) -> list[ResumoExecucao]:
    """Execuções mais recentes primeiro."""
    banco = db or get_db()

    with banco.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, criado_em, arquivo_nome, imagem_largura,
                   imagem_altura, checkpoint, limiar, qtd_deteccoes,
                   qtd_tags_lidas, tempo_deteccao_ms, tempo_ocr_ms,
                   tempo_total_ms, pasta
              FROM execucoes
             ORDER BY id DESC
             LIMIT ?
            """,
            (limite,),
        )

        return [ResumoExecucao(**dict(linha)) for linha in cursor.fetchall()]


def obter(execucao_id: int, db: Database | None = None) -> ResumoExecucao | None:
    banco = db or get_db()

    with banco.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, criado_em, arquivo_nome, imagem_largura,
                   imagem_altura, checkpoint, limiar, qtd_deteccoes,
                   qtd_tags_lidas, tempo_deteccao_ms, tempo_ocr_ms,
                   tempo_total_ms, pasta
              FROM execucoes
             WHERE id = ?
            """,
            (execucao_id,),
        )

        linha = cursor.fetchone()

    return ResumoExecucao(**dict(linha)) if linha else None


def deteccoes_de(execucao_id: int, db: Database | None = None) -> list[dict]:
    """As detecções de uma execução, prontas para cruzar com o gabarito."""
    banco = db or get_db()

    with banco.cursor() as cursor:
        cursor.execute(
            """
            SELECT classe, score, x1, y1, x2, y2, centro_x, centro_y,
                   tag, texto_bruto, confianca_ocr, descricao, grupo
              FROM deteccoes
             WHERE execucao_id = ?
             ORDER BY score DESC
            """,
            (execucao_id,),
        )

        return [dict(linha) for linha in cursor.fetchall()]
