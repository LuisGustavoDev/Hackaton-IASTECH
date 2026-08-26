"""
Dados de referência da norma ISA-5.1 (letras de identificação de TAG).

Fonte: tabela oficial compartilhada por Ricardo Sandrini (IASTECH) na
reunião de 14/08/2026 — confere com a tabela "Identification Letters"
do material do hackathon.

Cada letra pode ter significados diferentes dependendo da posição em
que aparece no TAG:

- Primeira letra:
    - VARIAVEL_MEDIDA      -> o que está sendo medido (ex: F = Vazão)
    - MODIFICADOR_VARIAVEL -> modifica a variável medida (ex: D = Diferencial)
- Letras seguintes:
    - FUNCAO_LEITURA     -> função de leitura/passiva (ex: I = Indicar)
    - FUNCAO_SAIDA       -> função de saída/ativa (ex: T = Transmitir)
    - MODIFICADOR_FUNCAO -> modifica a função (ex: H = Alto)

Exemplos confirmados na reunião com o Ricardo:
    FT  -> F(Vazão) + T(Transmitir)            = Transmissor de Vazão
    TI  -> T(Temperatura) + I(Indicar)          = Indicador de Temperatura
    PI  -> P(Pressão) + I(Indicar)              = Indicador de Pressão
    TT  -> T(Temperatura) + T(Transmitir)       = Transmissor de Temperatura
    WT  -> W(Peso) + T(Transmitir)              = Indicador/Transmissor de Peso
    PIT -> P(Pressão) + I(Indicar) + T(Transmitir)
"""

from __future__ import annotations

from app.models.database import Database, get_db


class CategoriasISA:
    VARIAVEL_MEDIDA = "variavel_medida"
    MODIFICADOR_VARIAVEL = "modificador_variavel"
    FUNCAO_LEITURA = "funcao_leitura"
    FUNCAO_SAIDA = "funcao_saida"
    MODIFICADOR_FUNCAO = "modificador_funcao"


# (letra, categoria, significado)
ISA_LETRAS: list[tuple[str, str, str]] = [
    ("A", CategoriasISA.VARIAVEL_MEDIDA, "Análise"),
    ("A", CategoriasISA.FUNCAO_LEITURA, "Alarme"),

    ("B", CategoriasISA.VARIAVEL_MEDIDA, "Queimador, combustão"),
    ("B", CategoriasISA.FUNCAO_LEITURA, "Livre escolha"),
    ("B", CategoriasISA.FUNCAO_SAIDA, "Livre escolha"),
    ("B", CategoriasISA.MODIFICADOR_FUNCAO, "Livre escolha"),

    ("C", CategoriasISA.VARIAVEL_MEDIDA, "Livre escolha"),
    ("C", CategoriasISA.FUNCAO_SAIDA, "Controle"),
    ("C", CategoriasISA.MODIFICADOR_FUNCAO, "Fechar"),

    ("D", CategoriasISA.VARIAVEL_MEDIDA, "Livre escolha"),
    ("D", CategoriasISA.MODIFICADOR_VARIAVEL, "Diferencial, desvio"),
    ("D", CategoriasISA.MODIFICADOR_FUNCAO, "Desvio"),

    ("E", CategoriasISA.VARIAVEL_MEDIDA, "Tensão/Voltagem"),
    ("E", CategoriasISA.FUNCAO_LEITURA, "Sensor, elemento primário"),

    ("F", CategoriasISA.VARIAVEL_MEDIDA, "Vazão"),
    ("F", CategoriasISA.MODIFICADOR_VARIAVEL, "Razão/Proporção"),

    ("G", CategoriasISA.VARIAVEL_MEDIDA, "Livre escolha"),
    ("G", CategoriasISA.FUNCAO_LEITURA, "Visor, indicador de vidro"),

    ("H", CategoriasISA.VARIAVEL_MEDIDA, "Manual"),
    ("H", CategoriasISA.MODIFICADOR_FUNCAO, "Alto"),

    ("I", CategoriasISA.VARIAVEL_MEDIDA, "Corrente elétrica"),
    ("I", CategoriasISA.FUNCAO_LEITURA, "Indicar"),

    ("J", CategoriasISA.VARIAVEL_MEDIDA, "Potência"),
    ("J", CategoriasISA.FUNCAO_LEITURA, "Varredura"),

    ("K", CategoriasISA.VARIAVEL_MEDIDA, "Tempo, programação"),
    ("K", CategoriasISA.MODIFICADOR_VARIAVEL, "Taxa de variação no tempo"),
    ("K", CategoriasISA.FUNCAO_SAIDA, "Estação de controle"),

    ("L", CategoriasISA.VARIAVEL_MEDIDA, "Nível"),
    ("L", CategoriasISA.FUNCAO_LEITURA, "Luz/Indicador luminoso"),
    ("L", CategoriasISA.MODIFICADOR_FUNCAO, "Baixo"),

    ("M", CategoriasISA.VARIAVEL_MEDIDA, "Livre escolha"),
    ("M", CategoriasISA.MODIFICADOR_FUNCAO, "Médio, intermediário"),

    ("N", CategoriasISA.VARIAVEL_MEDIDA, "Livre escolha"),
    ("N", CategoriasISA.FUNCAO_LEITURA, "Livre escolha"),
    ("N", CategoriasISA.FUNCAO_SAIDA, "Livre escolha"),
    ("N", CategoriasISA.MODIFICADOR_FUNCAO, "Livre escolha"),

    ("O", CategoriasISA.VARIAVEL_MEDIDA, "Livre escolha"),
    ("O", CategoriasISA.FUNCAO_LEITURA, "Orifício, restrição"),
    ("O", CategoriasISA.MODIFICADOR_FUNCAO, "Aberto"),

    ("P", CategoriasISA.VARIAVEL_MEDIDA, "Pressão"),
    ("P", CategoriasISA.FUNCAO_LEITURA, "Ponto de teste"),

    ("Q", CategoriasISA.VARIAVEL_MEDIDA, "Quantidade"),
    ("Q", CategoriasISA.MODIFICADOR_VARIAVEL, "Integrar, totalizar"),
    ("Q", CategoriasISA.FUNCAO_LEITURA, "Integrar, totalizar"),

    ("R", CategoriasISA.VARIAVEL_MEDIDA, "Radiação"),
    ("R", CategoriasISA.FUNCAO_LEITURA, "Registrar"),

    ("S", CategoriasISA.VARIAVEL_MEDIDA, "Velocidade, frequência"),
    ("S", CategoriasISA.MODIFICADOR_VARIAVEL, "Segurança"),
    ("S", CategoriasISA.FUNCAO_SAIDA, "Chave/Comutador"),

    ("T", CategoriasISA.VARIAVEL_MEDIDA, "Temperatura"),
    ("T", CategoriasISA.FUNCAO_SAIDA, "Transmitir"),

    ("U", CategoriasISA.VARIAVEL_MEDIDA, "Multivariável"),
    ("U", CategoriasISA.FUNCAO_LEITURA, "Multifunção"),
    ("U", CategoriasISA.FUNCAO_SAIDA, "Multifunção"),

    ("V", CategoriasISA.VARIAVEL_MEDIDA, "Vibração, análise mecânica"),
    ("V", CategoriasISA.FUNCAO_SAIDA, "Válvula, registro, veneziana"),

    ("W", CategoriasISA.VARIAVEL_MEDIDA, "Peso, força"),
    ("W", CategoriasISA.FUNCAO_LEITURA, "Poço (termopar)"),

    ("X", CategoriasISA.VARIAVEL_MEDIDA, "Não classificado"),
    ("X", CategoriasISA.MODIFICADOR_VARIAVEL, "Eixo X"),
    ("X", CategoriasISA.FUNCAO_LEITURA, "Não classificado"),
    ("X", CategoriasISA.FUNCAO_SAIDA, "Não classificado"),
    ("X", CategoriasISA.MODIFICADOR_FUNCAO, "Não classificado"),

    ("Y", CategoriasISA.VARIAVEL_MEDIDA, "Evento, estado, presença"),
    ("Y", CategoriasISA.MODIFICADOR_VARIAVEL, "Eixo Y"),
    ("Y", CategoriasISA.FUNCAO_SAIDA, "Dispositivos auxiliares"),

    ("Z", CategoriasISA.VARIAVEL_MEDIDA, "Posição, dimensão"),
    ("Z", CategoriasISA.MODIFICADOR_VARIAVEL, "Eixo Z"),
    ("Z", CategoriasISA.FUNCAO_SAIDA, "Acionador, atuador, elemento final de controle"),
]


def popular_isa_letras(db: Database | None = None) -> int:
    """
    Insere os dados de referência ISA no banco, se ainda não existirem.
    Idempotente: pode ser chamado toda vez que a aplicação sobe sem
    duplicar linhas (usa INSERT OR IGNORE + UNIQUE na tabela).

    Retorna quantas linhas existem na tabela após a operação.
    """
    db = db or get_db()
    with db.cursor() as cur:
        cur.executemany(
            """
            INSERT OR IGNORE INTO isa_letras (letra, categoria, significado)
            VALUES (?, ?, ?)
            """,
            ISA_LETRAS,
        )
        cur.execute("SELECT COUNT(*) FROM isa_letras")
        return cur.fetchone()[0]