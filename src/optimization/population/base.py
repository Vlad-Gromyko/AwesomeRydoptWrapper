"""
Базовый интерфейс для популяционных оптимизаторов с ask/tell.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import numpy as np


class PopulationOptimizer(ABC):
    """Базовый интерфейс для всех популяционных оптимизаторов."""

    @abstractmethod
    def ask(self) -> np.ndarray:
        """Запросить следующую особь."""
        pass

    @abstractmethod
    def tell(self, solutions: List[Tuple[np.ndarray, float]]) -> None:
        """Сообщить значения fitness для кандидатов."""
        pass

    @abstractmethod
    def result(self) -> Tuple[np.ndarray, float]:
        """Вернуть лучшее решение."""
        pass

    @property
    @abstractmethod
    def population_size(self) -> int:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    def reset(self) -> None:
        """Сброс оптимизатора (опционально)."""
        pass


class BoundedOptimizer(PopulationOptimizer):
    """Расширенный интерфейс для оптимизаторов с поддержкой границ."""

    @abstractmethod
    def set_bounds(self, lower: np.ndarray, upper: np.ndarray) -> None:
        pass