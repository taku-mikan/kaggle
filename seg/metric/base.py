from abc import ABCMeta, abstractmethod
from typing import Dict
from unittest.mock import NonCallableMagicMock

import torch

class Metric(metaclass=ABCMeta):
    @abstractmethod
    def evaluate(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        pass

    @abstractmethod
    def score(self) -> Dict[str, float]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    def log(self) -> str:
        result = ""
        score  = self.score()

        for name, score in score.items():
            if isinstance(score, int):
                result += f"{name}: {score} "
            else:
                result += f"{name}: {score:.3f} "
        
        # delete final space
        return result[:-1]

