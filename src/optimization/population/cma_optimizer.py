"""
CMA-ES оптимизатор с интерфейсом PopulationOptimizer.
"""

import numpy as np
from typing import Optional, Tuple, List
from cmaes import CMA

from .base import PopulationOptimizer, BoundedOptimizer


class CMAOptimizer(BoundedOptimizer):
    """CMA-ES оптимизатор с ручной обработкой границ."""

    def __init__(
        self,
        dimension: int,
        sigma0: float = 0.5,
        mean: Optional[np.ndarray] = None,  # Начальное среднее (можно передать хорошее решение)
        population_size: Optional[int] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        seed: Optional[int] = None,
    ):
        self._dimension = dimension
        self._sigma0 = sigma0
        self._seed = seed

        # Начальное среднее - если None, то нули, иначе переданное решение
        if mean is None:
            self._mean = np.zeros(dimension)
        else:
            self._mean = np.asarray(mean)
            assert len(self._mean) == dimension, f"Mean dimension mismatch: {len(self._mean)} vs {dimension}"

        if population_size is None:
            self._population_size = 4 + int(3 * np.log(dimension))
        else:
            self._population_size = population_size

        # Храним границы
        self._lower = None
        self._upper = None
        if bounds is not None:
            self._lower = np.asarray(bounds[0])
            self._upper = np.asarray(bounds[1])

        # Инициализируем CMA без bounds (будем обрабатывать в ask)
        self._optimizer = CMA(
            mean=self._mean,
            sigma=self._sigma0,
            bounds=None,
            population_size=self._population_size,
            seed=self._seed,
        )

        self._best_solution: Optional[Tuple[np.ndarray, float]] = None

    def ask(self) -> np.ndarray:
        x = self._optimizer.ask()
        if self._lower is not None:
            x = np.clip(x, self._lower, self._upper)
        return x

    def tell(self, solutions: List[Tuple[np.ndarray, float]]) -> None:
        for x, value in solutions:
            if self._best_solution is None or value < self._best_solution[1]:
                self._best_solution = (x.copy(), value)
        self._optimizer.tell(solutions)

    def result(self) -> Tuple[np.ndarray, float]:
        if self._best_solution is None:
            return self._mean.copy(), float('inf')
        return self._best_solution

    def set_bounds(self, lower: np.ndarray, upper: np.ndarray) -> None:
        self._lower = np.asarray(lower)
        self._upper = np.asarray(upper)

    @property
    def population_size(self) -> int:
        return self._population_size

    @property
    def dimension(self) -> int:
        return self._dimension

    def reset(self) -> None:
        self._best_solution = None
        self._optimizer = CMA(
            mean=self._mean,
            sigma=self._sigma0,
            bounds=None,
            population_size=self._population_size,
            seed=self._seed,
        )