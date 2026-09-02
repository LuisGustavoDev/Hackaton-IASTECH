"""
Ferramentas de medição do pipeline (Etapa 05 — Testes e Validação).

Não fazem parte do fluxo que atende o usuário: são CLIs que se apoiam no
histórico de execuções (app/models/execucoes.py) e no gabarito anotado
(dataset/original) para responder "o modelo está melhorando?".

Precisam do dataset e, no caso do lote, do Tesseract. Ambos existem
dentro do container via volume, então o caminho normal é:

    docker compose run --rm app python -m app.diagnostico.lote ...
"""
