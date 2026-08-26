from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

DB_PATH_ENV = "DB_PATH"
DEFAULT_DB_PATH = ":memory:"

# Só a tabela de referência ISA por enquanto. As tabelas de resultado
# (documentos, ferramentas etc.) entram aqui quando o schema delas
# for fechado.
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
"""


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
        self._init_schema()

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.executescript(SCHEMA)

    @contextmanager
    def cursor(self) -> Iterator[sqlite3.Cursor]:
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