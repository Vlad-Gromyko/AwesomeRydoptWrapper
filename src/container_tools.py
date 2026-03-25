from rydopt.protocols import GateSystem
from rydopt.pulses.pulse_ansatz import PulseAnsatz
from rydopt.simulation.fidelity import process_fidelity
from rydopt.optimization.optimize import OptimizationResult
from typing import Any, List, Callable
import jax
import jax.numpy as jnp
import numpy as np


def sort_by_fidelity_function(
    gate: GateSystem,
    pulse: PulseAnsatz,
    candidates: Any,
    fidelity_function: Callable = process_fidelity,
    tol: float = 1e-7,
) -> List[Any]:
    # Приводим к списку
    if not isinstance(candidates, (list, tuple)):
        candidates = [candidates]

    # Извлекаем параметры
    if candidates and hasattr(candidates[0], "params"):
        params_list = [c.params for c in candidates]
    else:
        params_list = candidates

    # Проверяем, что все параметры имеют одинаковую структуру
    # (для vmap нужно, чтобы списки параметров были одинаковой формы)
    # Преобразуем в батч: каждый компонент кортежа -> JAX-массив
    # Для длительности, детюнинга, фазы, амплитуды
    durations = jnp.array([p[0] for p in params_list])
    # detuning: может быть списком чисел, превращаем в массив
    detunings = jnp.array([p[1] for p in params_list])
    phases = jnp.array([p[2] for p in params_list])
    rabis = jnp.array([p[3] for p in params_list])

    # Собираем батч параметров (каждый элемент - массив с первой осью по кандидатам)
    batch_params = (durations, detunings, phases, rabis)

    # Векторизуем функцию потерь по первой оси
    vmap_fidelity = jax.vmap(fidelity_function, in_axes=(None, None, 0, None))

    # Вычисляем фиделити для всех кандидатов
    fidelities = vmap_fidelity(gate, pulse, batch_params, tol)

    # Преобразуем в список Python для сортировки
    fidelities_list = [float(f) for f in fidelities]

    # Сортируем
    sorted_pairs = sorted(zip(fidelities_list, candidates), key=lambda x: x[0], reverse=True)
    return [pair[1] for pair in sorted_pairs]


def filter_by_fidelity_function(
    gate: GateSystem,
    pulse: PulseAnsatz,
    candidates: Any,
    fidelity_function: Callable = process_fidelity,
    tol: float = 1e-7,
) -> Any:
    sorted_list = sort_by_fidelity_function(gate, pulse, candidates, fidelity_function, tol)
    if sorted_list:
        return sorted_list[0]
    return None