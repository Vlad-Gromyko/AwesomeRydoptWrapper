"""
Differential Evolution (DE) optimizer with PopulationOptimizer interface.
"""

import numpy as np
from typing import Optional, Tuple, List
from .base import PopulationOptimizer, BoundedOptimizer


class DEOptimizer(BoundedOptimizer):
    """
    Differential Evolution optimizer with bounds.

    Parameters
    ----------
    dimension : int
        Problem dimension.
    bounds : Tuple[np.ndarray, np.ndarray]
        (lower, upper) bounds for each variable.
    population_size : int, optional
        Population size. Default: max(4, 10*dimension).
    mutation_factor : float, default 0.8
        Differential weight F.
    crossover_prob : float, default 0.9
        Crossover probability CR.
    seed : int, optional
        Random seed.
    mean : np.ndarray, optional
        Good initial solution. If provided, it will be included in the initial population
        (exactly one individual equals mean). If `spread` is also given, the whole population
        is initialized around mean with spread.
    spread : float, optional
        If given, the initial population is uniformly sampled in [mean - spread, mean + spread]
        (clipped to bounds). Otherwise, the population is uniform in the whole bounds,
        but the `mean` individual is guaranteed to be present.
    """

    def __init__(
        self,
        dimension: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        population_size: Optional[int] = None,
        mutation_factor: float = 0.8,
        crossover_prob: float = 0.9,
        seed: Optional[int] = None,
        mean: Optional[np.ndarray] = None,
        spread: Optional[float] = None,          # соответствует initial_spread
    ):
        self._dimension = dimension
        self._lower = np.asarray(bounds[0], dtype=float)
        self._upper = np.asarray(bounds[1], dtype=float)
        assert self._lower.shape == (dimension,)
        assert self._upper.shape == (dimension,)

        if population_size is None:
            self._population_size = max(4, 10 * dimension)
        else:
            self._population_size = population_size

        self._mutation_factor = mutation_factor
        self._crossover_prob = crossover_prob
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        self._mean = np.asarray(mean) if mean is not None else None
        self._spread = spread

        self._population = self._init_population()
        self._fitness = np.full(self._population_size, np.inf)
        self._best_solution: Optional[Tuple[np.ndarray, float]] = None
        self._current_idx = 0
        self._ready_for_evolution = False

    def _init_population(self) -> np.ndarray:
        """Create initial population according to rules."""
        if self._mean is None:
            # Uniform random in whole bounds
            return self._rng.uniform(
                low=self._lower, high=self._upper,
                size=(self._population_size, self._dimension)
            )
        elif self._spread is not None:
            # Localized initialization around mean with given spread
            half = self._spread
            pop = self._rng.uniform(
                low=self._mean - half,
                high=self._mean + half,
                size=(self._population_size, self._dimension)
            )
            np.clip(pop, self._lower, self._upper, out=pop)
            return pop
        else:
            # Uniform random but guarantee that mean is included
            pop = self._rng.uniform(
                low=self._lower, high=self._upper,
                size=(self._population_size, self._dimension)
            )
            mean_clipped = np.clip(self._mean, self._lower, self._upper)
            pop[0] = mean_clipped
            return pop

    def ask(self) -> np.ndarray:
        if self._current_idx >= self._population_size:
            self._evolve()
            self._current_idx = 0
        return self._population[self._current_idx].copy()

    def tell(self, solutions: List[Tuple[np.ndarray, float]]) -> None:
        for x, value in solutions:
            if self._current_idx >= self._population_size:
                raise RuntimeError("Too many tell calls")
            self._fitness[self._current_idx] = value
            if self._best_solution is None or value < self._best_solution[1]:
                self._best_solution = (x.copy(), value)
            self._current_idx += 1
        if self._current_idx == self._population_size:
            self._ready_for_evolution = True

    def _evolve(self):
        new_population = np.empty_like(self._population)
        for i in range(self._population_size):
            indices = [j for j in range(self._population_size) if j != i]
            r1, r2, r3 = self._rng.choice(indices, size=3, replace=False)
            mutant = (self._population[r1] +
                      self._mutation_factor * (self._population[r2] - self._population[r3]))
            cross_mask = self._rng.random(self._dimension) < self._crossover_prob
            j_rand = self._rng.integers(0, self._dimension)
            cross_mask[j_rand] = True
            trial = np.where(cross_mask, mutant, self._population[i])
            np.clip(trial, self._lower, self._upper, out=trial)
            new_population[i] = trial
        self._population = new_population
        self._fitness.fill(np.inf)
        self._ready_for_evolution = False

    def result(self) -> Tuple[np.ndarray, float]:
        if self._best_solution is None:
            return self._population[0].copy(), np.inf
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
        self._rng = np.random.default_rng(self._seed)
        self._population = self._init_population()
        self._fitness = np.full(self._population_size, np.inf)
        self._best_solution = None
        self._current_idx = 0
        self._ready_for_evolution = False