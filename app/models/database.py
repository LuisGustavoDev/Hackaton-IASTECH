from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Iterator

DB_PATH_ENV = "DB_PATH"
DEFAULT_DB_PATH = ":memory:"

SCHEMA = """
-- Tabela de REFERÊNCIA (não é resultado do pipeline): dicionário de
-- letras de identificação da norma ISA-5.1. A IA consulta essa tabela
-- pra decompor um TAG lido pelo OCR (ex: "FT210" -> F, T) e descobrir
-- o significado de cada letra conforme a posição em que ela aparece.
CREATE TABLE IF NOT EXISTS isa_letras (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    letra        TEXT NOT NULL,   -- A-Z
    categoria    TEXT NOT NULL,   -- ver CategoriasISA em isa_seed.py
    significado  TEXT NOT NULL,   -- ex: "Vazão", "Transmitir"
    UNIQUE(letra, categoria, significado)
);

CREATE INDEX IF NOT EXISTS idx_isa_letras_letra ON isa_letras(letra);

-- Tabelas de RESULTADO: uma linha por imagem processada, e uma linha por
-- equipamento detectado nela.
--
-- Existem para viabilizar a matriz de confusão (VP, FP, VN, FN) e os
-- indicadores da Etapa 05 do Plano de Desenvolvimento — tempo médio de
-- processamento, taxa de acerto do OCR, número de falsos positivos,
-- quantidade encontrada x quantidade real. Nada disso é calculável a
-- partir da planilha, que é sobrescrita a cada execução: é preciso
-- guardar o que o modelo previu, para depois cruzar com o gabarito
-- (os .xml em dataset/original, que já estão versionados e por isso NÃO
-- são duplicados aqui).
--
-- O checkpoint e o limiar ficam gravados junto porque a pergunta que
-- essas tabelas respondem é comparativa: "as 250 épocas melhoraram em
-- relação às 15?". Sem saber qual modelo gerou cada linha, os números de
-- duas rodadas não são comparáveis.
CREATE TABLE IF NOT EXISTS execucoes (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    criado_em           TEXT    NOT NULL,  -- ISO-8601 UTC
    arquivo_nome        TEXT,              -- nome enviado pelo usuário
    imagem_largura      INTEGER NOT NULL,
    imagem_altura       INTEGER NOT NULL,
    checkpoint          TEXT,              -- qual modelo gerou este resultado
    limiar              REAL,
    qtd_deteccoes       INTEGER NOT NULL,
    qtd_tags_lidas      INTEGER NOT NULL,  -- detecções que receberam TAG válida
    tempo_deteccao_ms   REAL,
    tempo_ocr_ms        REAL,
    tempo_total_ms      REAL,
    pasta               TEXT               -- onde a planilha desta execução ficou
);

CREATE INDEX IF NOT EXISTS idx_execucoes_criado_em
    ON execucoes(criado_em);

CREATE TABLE IF NOT EXISTS deteccoes (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    execucao_id    INTEGER NOT NULL,
    classe         TEXT    NOT NULL,  -- Tipo previsto pelo Faster R-CNN
    score          REAL    NOT NULL,
    x1             INTEGER NOT NULL,
    y1             INTEGER NOT NULL,
    x2             INTEGER NOT NULL,
    y2             INTEGER NOT NULL,
    centro_x       INTEGER NOT NULL,
    centro_y       INTEGER NOT NULL,
    tag            TEXT,              -- TAG normalizada ("" se não houve)
    texto_bruto    TEXT,              -- o que o OCR leu, antes do filtro ISA
    confianca_ocr  REAL,
    descricao      TEXT,
    grupo          TEXT,
    FOREIGN KEY (execucao_id) REFERENCES execucoes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_deteccoes_execucao
    ON deteccoes(execucao_id);

CREATE INDEX IF NOT EXISTS idx_deteccoes_classe
    ON deteccoes(classe);
"""


# Colunas acrescentadas ao SCHEMA depois da primeira versão. Bancos já
# existentes recebem cada uma via ALTER TABLE — ver Database._migrar().
# (tabela, coluna, tipo)
COLUNAS_ADICIONADAS = [
    ("execucoes", "pasta", "TEXT"),
]


class Database:
    """
    Wrapper fino sobre sqlite3.

    Mantém UMA conexão viva durante todo o ciclo de vida da aplicação.
    Isso é importante especialmente no modo ":memory:": se a conexão
    fechar, o banco em memória desaparece.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or os.environ.get(DB_PATH_ENV, DEFAULT_DB_PATH)

        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON;")

        # A conexão é uma só e compartilhada entre threads. Enquanto o
        # banco só era lido uma vez por processo (o dicionário ISA), isso
        # era inofensivo; agora que cada requisição grava uma execução, e
        # que as rotas rodam no threadpool do FastAPI, duas gravações
        # simultâneas dividiriam o mesmo commit/rollback. RLock (e não
        # Lock) para que um cursor aninhado não trave o processo.
        self._lock = RLock()

        self._init_schema()

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.executescript(SCHEMA)
            self._migrar()

    def _migrar(self) -> None:
        """
        Acrescenta colunas que entraram no SCHEMA depois que o banco já
        existia.

        `CREATE TABLE IF NOT EXISTS` não altera tabela existente: num banco
        já criado, a coluna nova simplesmente não apareceria e toda
        consulta que a mencionasse quebraria. Como DB_PATH aponta para
        arquivo em produção, esse caso é a regra, não a exceção.
        """
        for tabela, coluna, tipo in COLUNAS_ADICIONADAS:
            existentes = {
                linha[1]
                for linha in self.conn.execute(
                    f"PRAGMA table_info({tabela})"
                )
            }

            if not existentes:
                continue  # a tabela ainda não existe; o SCHEMA já a criou

            if coluna not in existentes:
                self.conn.execute(
                    f"ALTER TABLE {tabela} ADD COLUMN {coluna} {tipo}"
                )

    @contextmanager
    def cursor(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cur = self.conn.cursor()
            try:
                yield cur
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
            finally:
                cur.close()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


_db_instance: Database | None = None


def get_db() -> Database:
    """Retorna a instância única (singleton) do banco para o processo atual."""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance


def redefinir_db() -> None:
    """
    Descarta a instância em cache (usado pelos testes).

    Mesmo papel de redefinir_detector() em app/detection/detector.py:
    sem isso, um teste que aponta DB_PATH para um arquivo temporário
    continuaria falando com o banco criado pelo teste anterior.
    """
    global _db_instance

    if _db_instance is not None:
        _db_instance.close()

    _db_instance = None
