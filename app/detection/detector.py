"""
Detector de equipamentos em produção.

Recebe uma imagem já carregada (BGR, como o resto do pipeline usa) e
devolve a lista de equipamentos encontrados. Roda inteiramente em CPU e
não precisa de internet: os pesos vêm do checkpoint portátil.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from app import config
from app.detection.checkpoint import carregar_checkpoint
from app.detection.modelo import construir_faster_rcnn, nome_da_classe


class DetectorEquipamentos:
    """
    Faster R-CNN carregado a partir de um checkpoint portátil.

    Carregar o modelo custa alguns segundos, então a instância é feita
    para ser criada UMA vez e reaproveitada — use obter_detector().
    """

    def __init__(
        self,
        caminho_checkpoint: str | Path | None = None,
        device: str = "cpu",
        limiar: float | None = None,
    ) -> None:
        self.caminho_checkpoint = Path(
            caminho_checkpoint or config.caminho_checkpoint()
        )
        self.device = torch.device(device)
        self.limiar = (
            limiar if limiar is not None else config.limiar_deteccao()
        )

        checkpoint = carregar_checkpoint(self.caminho_checkpoint)

        self.classes: list[str] = checkpoint["classes"]
        self.metadados: dict = checkpoint.get("metadados", {})

        modelo = construir_faster_rcnn(
            checkpoint["num_classes"],
            pesos_pretreinados=False,
        )
        modelo.load_state_dict(checkpoint["state_dict"])
        modelo.to(self.device)
        modelo.eval()

        self.modelo = modelo

    @torch.no_grad()
    def detectar(self, imagem: np.ndarray) -> list[dict]:
        """
        Detecta os equipamentos de uma imagem BGR (o formato que o
        cv2.imread do pipeline devolve).

        Retorna uma lista de dicionários ordenada da detecção mais
        confiante para a menos confiante:

            {
                "classe":    "Válvula",
                "score":     0.87,
                "x1", "y1", "x2", "y2":  bounding box em pixels,
                "centro_x", "centro_y":  centro da bounding box,
            }
        """
        tensor = self._para_tensor(imagem)

        saidas = self.modelo([tensor])[0]

        boxes = saidas["boxes"].cpu()
        scores = saidas["scores"].cpu()
        labels = saidas["labels"].cpu()

        deteccoes = []

        for box, score, label in zip(boxes, scores, labels):
            score = float(score)

            if score < self.limiar:
                continue

            # Descarta o background por segurança: ele não deveria sair do
            # postprocess do torchvision, mas um checkpoint com a cabeça
            # errada faria isso passar despercebido.
            if int(label) <= 0:
                continue

            x1, y1, x2, y2 = (int(round(v)) for v in box.tolist())

            deteccoes.append(
                {
                    "classe": nome_da_classe(int(label), self.classes),
                    "score": score,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "centro_x": (x1 + x2) // 2,
                    "centro_y": (y1 + y2) // 2,
                }
            )

        deteccoes.sort(key=lambda d: d["score"], reverse=True)

        return deteccoes

    def _para_tensor(self, imagem: np.ndarray) -> torch.Tensor:
        """
        Converte BGR uint8 (HxWxC) no tensor RGB float [0,1] (CxHxW) que o
        torchvision espera.

        A conversão é feita à mão, sem cv2, para que o detector continue
        importável em ambientes onde só torch está instalado.
        """
        if not isinstance(imagem, np.ndarray) or imagem.ndim != 3:
            raise ValueError(
                "A imagem precisa ser um array BGR com 3 dimensões "
                "(altura, largura, canais)."
            )

        rgb = imagem[:, :, ::-1]

        tensor = torch.from_numpy(np.ascontiguousarray(rgb))
        tensor = tensor.permute(2, 0, 1).float().div(255.0)

        return tensor.to(self.device)


_detector: DetectorEquipamentos | None = None


def obter_detector() -> DetectorEquipamentos:
    """
    Retorna a instância única do detector para o processo atual.

    Mesmo padrão do get_db() em app/models/database.py: evita recarregar
    ~170 MB de pesos a cada requisição.
    """
    global _detector

    if _detector is None:
        _detector = DetectorEquipamentos()

    return _detector


def redefinir_detector() -> None:
    """Descarta a instância em cache (usado pelos testes)."""
    global _detector
    _detector = None
