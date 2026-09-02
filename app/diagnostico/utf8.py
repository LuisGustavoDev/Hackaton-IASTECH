"""Saída em UTF-8 no console do Windows."""

from __future__ import annotations

import sys


def forcar_utf8() -> None:
    """
    Sem isto o console usa o codepage do sistema e nomes de classe
    acentuados saem quebrados ("Conexão" -> "Conex?o") — justamente os
    nomes que estes relatórios existem para mostrar.
    """
    for fluxo in (sys.stdout, sys.stderr):
        try:
            fluxo.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
