"""
Detecção de equipamentos em P&IDs com Faster R-CNN.

Este pacote é a metade de INFERÊNCIA do detector: roda na máquina fraca
(notebook, CPU) e depende apenas de torch/torchvision. Não importa nada
de app/training — o treino exige pycocotools, o dataset COCO e um
otimizador, que não fazem sentido em produção.
"""
