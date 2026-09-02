"""
Treino do detector de equipamentos (Faster R-CNN).

Este pacote roda SÓ na máquina forte (desktop com GPU NVIDIA, CUDA) e é
excluído da imagem Docker de produção pelo .dockerignore. Ele depende de
pycocotools e do dataset COCO — nada disso faz sentido no notebook
que roda a inferência.

A ponte entre os dois mundos é um único arquivo: o checkpoint portátil
gerado por app/detection/checkpoint.py:salvar_checkpoint().

Nomes em inglês neste pacote (load_coco_utf8, build_canonical_categories,
remap_coco_to_canonical, CocoDetectionRaw) são propositais: essa camada é
adaptada do benchmark_pid_models.py e manter os nomes originais deixa a
origem rastreável. O restante do projeto segue a nomenclatura em
português.
"""
