"""
Matriz de confusão do detector (VP, FP, FN) e acurácia por classe.

Entregável pedido por Ricardo Sandrini na reunião de 14/08/2026 e listado
na Etapa 05 do Plano de Desenvolvimento.

Cruza o que o modelo previu (tabela `deteccoes`, gravada por
app/models/execucoes.py) com o gabarito anotado (os .xml em
dataset/original), casando as caixas por IoU.

COMO O CASAMENTO É FEITO
------------------------
Para cada imagem, as predições são percorridas da mais confiante para a
menos confiante e cada uma reivindica a anotação ainda livre de maior
IoU, desde que passe do limiar. Depois disso:

    predição casada, mesma classe     -> VP
    predição casada, classe diferente -> confusão (célula da matriz)
    predição sem anotação             -> FP  (previu onde não havia nada)
    anotação sem predição             -> FN  (deixou de ver)

O casamento é feito ANTES de comparar as classes, de propósito: assim uma
válvula rotulada como "Outro" aparece como confusão entre duas classes, e
não como um FP somado a um FN, que é o que esconderia o erro real.

SOBRE O "VN"
------------
Detecção de objetos não tem verdadeiro negativo natural: não existe um
conjunto finito de "caixas que poderiam ter sido previstas e corretamente
não foram", então não há o que contar. VP, FP e FN saem do cruzamento; o
VN precisa antes de uma definição de negócio (por exemplo: classes do
catálogo ausentes do diagrama e corretamente não previstas). Este
relatório diz isso explicitamente em vez de imprimir um zero que
pareceria uma medição.

USO
---
    python -m app.diagnostico.matriz_confusao --ultimas 21
    python -m app.diagnostico.matriz_confusao --execucoes 3,4,5
    python -m app.diagnostico.matriz_confusao --checkpoint data/models/faster_rcnn.pt
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from app.diagnostico import gabarito
from app.diagnostico.utf8 import forcar_utf8
from app.models import execucoes

SEM_PAR = "(nenhum)"


def casar_detalhado(
    predicoes: list[dict], anotacoes: list[dict], limiar_iou: float
):
    """
    Casamento com os objetos preservados, para o relatório detalhado.

    Devolve (pares, predicoes_sem_par, anotacoes_sem_par), onde cada par é
    a tupla (anotacao, predicao, iou). `casar()` é a versão resumida desta,
    que só devolve os nomes de classe.
    """
    livres = list(range(len(anotacoes)))
    pares = []
    sem_par_predicao = []

    for predicao in sorted(predicoes, key=lambda p: -p["score"]):
        melhor_indice = None
        melhor_iou = limiar_iou

        for indice in livres:
            valor = gabarito.iou(predicao, anotacoes[indice])
            if valor >= melhor_iou:
                melhor_iou = valor
                melhor_indice = indice

        if melhor_indice is None:
            sem_par_predicao.append(predicao)
            continue

        livres.remove(melhor_indice)
        pares.append((anotacoes[melhor_indice], predicao, melhor_iou))

    return pares, sem_par_predicao, [anotacoes[i] for i in livres]


def casar(predicoes: list[dict], anotacoes: list[dict], limiar_iou: float):
    """
    Devolve (pares, classes_previstas_sem_par, classes_anotadas_sem_par).

    `pares` são tuplas (classe_real, classe_prevista).
    """
    pares, fp, fn = casar_detalhado(predicoes, anotacoes, limiar_iou)

    return (
        [(anotacao["classe"], predicao["classe"]) for anotacao, predicao, _ in pares],
        [predicao["classe"] for predicao in fp],
        [anotacao["classe"] for anotacao in fn],
    )


def apurar(
    ids: list[int],
    indice_gabarito: dict,
    limiar_iou: float,
    limiar_score: float = 0.0,
):
    """
    `limiar_score` descarta predições gravadas com score abaixo dele.

    Filtrar aqui, e não na hora de detectar, é o que permite levantar a
    curva de precisão/revocação inteira a partir de UMA passada do
    detector: rodar o lote com o limiar no piso do modelo grava tudo, e
    cada ponto da curva sai de uma releitura do banco em milissegundos,
    em vez de outra rodada de inferência.
    """
    matriz: Counter = Counter()
    imagens_medidas = 0
    ignoradas = []

    for execucao_id in ids:
        resumo = execucoes.obter(execucao_id)

        if resumo is None:
            continue

        xml_path = indice_gabarito.get(resumo.arquivo_nome or "")

        if xml_path is None:
            ignoradas.append(resumo.arquivo_nome)
            continue

        predicoes = [
            p
            for p in execucoes.deteccoes_de(execucao_id)
            if p["score"] >= limiar_score
        ]
        anotacoes = gabarito.caixas(xml_path)

        pares, fp, fn = casar(predicoes, anotacoes, limiar_iou)

        for real, previsto in pares:
            matriz[(real, previsto)] += 1

        for classe in fp:
            matriz[(SEM_PAR, classe)] += 1

        for classe in fn:
            matriz[(classe, SEM_PAR)] += 1

        imagens_medidas += 1

    return matriz, imagens_medidas, ignoradas


def relatorio(matriz, imagens_medidas, ignoradas, limiar_iou):
    classes = sorted(
        {real for real, _ in matriz if real != SEM_PAR}
        | {previsto for _, previsto in matriz if previsto != SEM_PAR}
    )

    vp = sum(q for (r, p), q in matriz.items() if r == p and r != SEM_PAR)
    fp = sum(q for (r, _), q in matriz.items() if r == SEM_PAR)
    fn = sum(q for (_, p), q in matriz.items() if p == SEM_PAR)
    trocas = sum(
        q
        for (r, p), q in matriz.items()
        if r != p and r != SEM_PAR and p != SEM_PAR
    )

    print("=" * 78)
    print(" MATRIZ DE CONFUSAO")
    print("=" * 78)
    print(f"imagens medidas : {imagens_medidas}")
    print(f"limiar de IoU   : {limiar_iou}")

    if ignoradas:
        nomes = ", ".join(str(n) for n in ignoradas[:5])
        reticencias = "..." if len(ignoradas) > 5 else ""
        print(f"sem gabarito    : {len(ignoradas)} ({nomes}{reticencias})")

    print()
    print(f"VP (acertou classe e posição) : {vp}")
    print(f"FP (previu onde não havia)    : {fp}")
    print(f"FN (deixou de ver)            : {fn}")
    print(f"Classe trocada (casou a caixa): {trocas}")
    print("VN                            : não aplicável em detecção")

    # Classe trocada é erro dos dois lados: falta um acerto onde havia
    # equipamento e sobra uma predição que não corresponde a nada.
    precisao = vp / (vp + fp + trocas) if (vp + fp + trocas) else 0.0
    revocacao = vp / (vp + fn + trocas) if (vp + fn + trocas) else 0.0
    f1 = (
        2 * precisao * revocacao / (precisao + revocacao)
        if (precisao + revocacao)
        else 0.0
    )

    print()
    print(f"Precisão : {precisao:.3f}")
    print(f"Revocação: {revocacao:.3f}")
    print(f"F1       : {f1:.3f}")

    print()
    print("-" * 78)
    print(" POR CLASSE")
    print("-" * 78)

    cabecalho = (
        "classe".ljust(22)
        + "anotadas".rjust(9)
        + "VP".rjust(6)
        + "FP".rjust(6)
        + "FN".rjust(6)
        + "precisão".rjust(10)
        + "revocação".rjust(11)
    )
    print(cabecalho)

    for classe in classes:
        anotadas = sum(q for (r, _), q in matriz.items() if r == classe)
        previstas = sum(q for (_, p), q in matriz.items() if p == classe)
        acertos = matriz.get((classe, classe), 0)

        p = acertos / previstas if previstas else 0.0
        r = acertos / anotadas if anotadas else 0.0

        print(
            classe.ljust(22)
            + str(anotadas).rjust(9)
            + str(acertos).rjust(6)
            + str(previstas - acertos).rjust(6)
            + str(anotadas - acertos).rjust(6)
            + f"{p:.3f}".rjust(10)
            + f"{r:.3f}".rjust(11)
        )

    confusoes = sorted(
        (
            (q, real, previsto)
            for (real, previsto), q in matriz.items()
            if real != previsto and real != SEM_PAR and previsto != SEM_PAR
        ),
        reverse=True,
    )

    if confusoes:
        print()
        print("-" * 78)
        print(" CLASSES MAIS CONFUNDIDAS (caixa certa, rótulo errado)")
        print("-" * 78)
        for quantidade, real, previsto in confusoes[:10]:
            print(f"  {quantidade:>4}x  {real}  ->  {previsto}")


def curva(ids: list[int], indice_gabarito: dict, limiar_iou: float):
    """
    Precisão x revocação em cada limiar de score.

    Serve para escolher o ponto de operação com número em vez de
    intuição. Todos os pontos saem da MESMA passada do detector — o
    banco guarda o score de cada predição, então baixar o limiar é uma
    releitura, não uma nova inferência.
    """
    print("=" * 78)
    print(" CURVA DE PRECISÃO x REVOCAÇÃO POR LIMIAR DE SCORE")
    print("=" * 78)
    print(
        "limiar".rjust(7)
        + "previstas".rjust(11)
        + "VP".rjust(7)
        + "FP".rjust(7)
        + "FN".rjust(7)
        + "troca".rjust(7)
        + "precisão".rjust(10)
        + "revocação".rjust(11)
        + "F1".rjust(8)
    )

    for limiar in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        matriz, medidas, _ = apurar(
            ids, indice_gabarito, limiar_iou, limiar
        )

        if not medidas:
            continue

        vp = sum(q for (r, p), q in matriz.items() if r == p and r != SEM_PAR)
        fp = sum(q for (r, _), q in matriz.items() if r == SEM_PAR)
        fn = sum(q for (_, p), q in matriz.items() if p == SEM_PAR)
        trocas = sum(
            q
            for (r, p), q in matriz.items()
            if r != p and r != SEM_PAR and p != SEM_PAR
        )

        precisao = vp / (vp + fp + trocas) if (vp + fp + trocas) else 0.0
        revocacao = vp / (vp + fn + trocas) if (vp + fn + trocas) else 0.0
        f1 = (
            2 * precisao * revocacao / (precisao + revocacao)
            if (precisao + revocacao)
            else 0.0
        )

        print(
            f"{limiar:.2f}".rjust(7)
            + str(vp + fp + trocas).rjust(11)
            + str(vp).rjust(7)
            + str(fp).rjust(7)
            + str(fn).rjust(7)
            + str(trocas).rjust(7)
            + f"{precisao:.3f}".rjust(10)
            + f"{revocacao:.3f}".rjust(11)
            + f"{f1:.3f}".rjust(8)
        )


def resolver_ids(args) -> list[int]:
    if args.execucoes:
        return [int(x) for x in args.execucoes.split(",") if x.strip()]

    todas = execucoes.listar(limite=args.ultimas or 1000)

    if args.checkpoint:
        todas = [e for e in todas if e.checkpoint == args.checkpoint]

    return [e.id for e in todas]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--execucoes", help="ids separados por vírgula")
    parser.add_argument("--ultimas", type=int, help="as N execuções mais recentes")
    parser.add_argument("--checkpoint", help="filtra por checkpoint")
    parser.add_argument("--gabarito", default="dataset/original")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument(
        "--score",
        type=float,
        default=0.0,
        help="Descarta predições gravadas abaixo deste score.",
    )
    parser.add_argument(
        "--curva",
        action="store_true",
        help=(
            "Varre os limiares de score e mostra precisão/revocação "
            "em cada um, para escolher o ponto de operação."
        ),
    )
    return parser.parse_args()


def main() -> None:
    forcar_utf8()

    args = parse_args()

    indice = gabarito.indexar(Path(args.gabarito))

    if not indice:
        raise SystemExit(f"Nenhum .xml de gabarito em {args.gabarito}")

    ids = resolver_ids(args)

    if not ids:
        raise SystemExit(
            "Nenhuma execução encontrada. Rode "
            "'python -m app.diagnostico.lote' antes."
        )

    if args.curva:
        curva(ids, indice, args.iou)
        return

    matriz, medidas, ignoradas = apurar(
        ids, indice, args.iou, args.score
    )

    if not medidas:
        raise SystemExit(
            "Nenhuma execução tinha gabarito correspondente. Confira se "
            f"arquivo_nome bate com o nome das imagens em {args.gabarito}."
        )

    relatorio(matriz, medidas, ignoradas, args.iou)


if __name__ == "__main__":
    main()
