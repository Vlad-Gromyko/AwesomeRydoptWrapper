import optax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Optional, Iterable

from src.optimization.optimize import *
from rydopt.optimization.optimize import optimize as ro_optimize
from rydopt.optimization.optimize import multi_start_optimize as ro_multi_start_optimize
from rydopt.simulation.fidelity import process_fidelity
from rydopt.protocols import GateSystem, PulseAnsatzLike
from rydopt.simulation.evolve import evolve
from rydopt.types import PulseParams

from itertools import cycle, islice
from src.container_tools import sort_by_fidelity_function, filter_by_fidelity_function


# -----------------------------------------------------------------------------
# Визуализация прогресса оптимизации
# -----------------------------------------------------------------------------

def plot_optimization_flow(steps: List[Dict[str, Any]], save_path: Optional[str] = None):
    """
    График прогресса infidelity.
    """
    if not steps:
        print("No steps to visualize")
        return

    print(f"Total steps: {len(steps)}")

    step_nums = []
    infidelities = []
    methods = []
    improvements = []

    prev_inf = None
    for i, step in enumerate(steps):
        step_nums.append(i + 1)
        metadata = step.get('metadata', {})

        inf = metadata.get('output_infidelity')
        if inf is None:
            print(f"WARNING: No output_infidelity in step {i+1}")
            continue
        infidelities.append(inf)

        method = step['method']
        if 'optimize' in method and 'population' not in method:
            methods.append('Gradient')
        elif 'population' in method:
            methods.append('CMA-ES')
        elif 'multi_start' in method:
            methods.append('Multi-start')
        elif 'sequence' in method:
            methods.append('Sequence')
        else:
            methods.append(method[:15])

        if prev_inf is not None:
            improvements.append(prev_inf - inf)
        else:
            improvements.append(0)
        prev_inf = inf

    if not infidelities:
        print("No infidelity data available")
        return

    color_map = {
        'Gradient': '#1f77b4',
        'CMA-ES': '#2ca02c',
        'Multi-start': '#ff7f0e',
        'Sequence': '#d62728',
    }
    colors = [color_map.get(m, '#888888') for m in methods]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.semilogy(step_nums[:len(infidelities)], infidelities, 'o-', linewidth=2,
                markersize=10, color='#333333', zorder=1, alpha=0.5)

    for i, (x, y, method, imp) in enumerate(zip(step_nums[:len(infidelities)], infidelities, methods, improvements)):
        ax.plot(x, y, 'o', markersize=12, color=colors[i], zorder=2,
                markeredgecolor='black', markeredgewidth=1.5)
        ax.annotate(method, (x, y), xytext=(0, 12), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color=colors[i])
        ax.annotate(f'{y:.1e}', (x, y), xytext=(8, 0), textcoords='offset points',
                    ha='left', va='center', fontsize=8, fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        if imp > 0 and i > 0:
            ax.annotate(f'▼ {imp:.1e}', (x, y), xytext=(0, -20), textcoords='offset points',
                        ha='center', va='top', fontsize=7, color='#2e7d32')

    ax.set_xlabel('Optimization Step', fontsize=12)
    ax.set_ylabel('Infidelity (log scale)', fontsize=12)
    ax.set_title('Optimization Progress - Infidelity', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.set_xlim(0.5, len(step_nums) + 0.5)

    table_data = [[f"{i+1}", methods[i], f"{infidelities[i]:.1e}", f"{improvements[i]:.1e}" if improvements[i] > 0 else '-']
                  for i in range(len(infidelities))]
    table = ax.table(cellText=table_data,
                     colLabels=['Step', 'Method', 'Infidelity', 'Improvement'],
                     loc='bottom', bbox=[0, -0.4, 1, 0.25],
                     cellLoc='center', fontsize=8)
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()


# -----------------------------------------------------------------------------
# Состояние AwesomeRydopt (логирование и переключение режимов)
# -----------------------------------------------------------------------------

class AwesomeState:
    def __init__(self):
        self.__imitate_rydopt = True
        self._steps = []          # история шагов оптимизации

    @property
    def imitate_rydopt(self):
        return self.__imitate_rydopt

    @imitate_rydopt.setter
    def imitate_rydopt(self, value: bool = True):
        self.__imitate_rydopt = value

    def log_step(self, method: str,
                 input_params, output_params,
                 input_infidelity: float, output_infidelity: float,
                 **extra):
        """Сохранить информацию о выполненном шаге оптимизации."""
        self._steps.append({
            'method': method,
            'input_params': input_params,
            'output_params': output_params,
            'input_infidelity': input_infidelity,
            'output_infidelity': output_infidelity,
            'extra': extra
        })

    def clear_history(self):
        self._steps.clear()

    def plot_history(self, save_path: Optional[str] = None):
        """Построить график прогресса infidelity по всем записанным шагам."""
        if not self._steps:
            print("No optimization steps recorded.")
            return
        steps_for_plot = []
        for step in self._steps:
            steps_for_plot.append({
                'method': step['method'],
                'metadata': {
                    'output_infidelity': step['output_infidelity'],
                    'input_infidelity': step.get('input_infidelity'),
                    **step.get('extra', {})
                }
            })
        plot_optimization_flow(steps_for_plot, save_path)


# Глобальный экземпляр состояния
awesome_state = AwesomeState()


def imitate_rydopt(condition: bool = True):
    awesome_state.imitate_rydopt = condition


def is_ask_tell(optimizer: Callable):
    condition = hasattr(optimizer, 'ask') and hasattr(optimizer.ask, 'tell')
    return condition


def adjust_sequence(iterable, n):
    length = len(iterable)
    if length == n:
        return iterable
    repeated = islice(cycle(iterable), n)
    if isinstance(iterable, tuple):
        return tuple(repeated)
    else:
        return list(repeated)


def make_full_specs(*args: Union[int, float, Callable] | Union[Iterable[int], Iterable[float], Iterable[Callable]]):
    processed = []
    has_sequence = False
    for arg in args:
        if isinstance(arg, (list, tuple)):
            elements = list(arg)
            length = len(elements)
            has_sequence = True
        else:
            elements = [arg]
            length = 1
        processed.append((elements, length))
    if not has_sequence:
        raise ValueError("Среди аргументов нет ни одной последовательности")
    max_len = max(length for _, length in processed)
    result = []
    for elements, length in processed:
        if length == max_len:
            result.append(elements)
        else:
            repeated = islice(cycle(elements), max_len)
            result.append(list(repeated))
    return result


def _to_jax_params(params: PulseParams) -> PulseParams:
    """Преобразовать параметры (кортеж из скаляров/списков) в кортеж JAX-массивов."""
    return tuple(jnp.asarray(p) if isinstance(p, (np.ndarray, list, tuple)) else p for p in params)


def optimize(
        gate: GateSystem,
        pulse: PulseAnsatz,
        initial_params: PulseParams,
        fixed_initial_params: FixedPulseParams | None = None,
        min_initial_params: PulseParams = None,
        max_initial_params: PulseParams = None,
        *,
        num_steps: int = 1000,
        learning_rate: float = 0.005,
        tol: float = 1e-7,
        return_history: bool = False,
        verbose: bool = False,
        method: Callable = None,
        fidelity_type: Callable = None,
        apply_bounds: bool = True,
        return_best: bool = True
) -> OptimizationResult[PulseParams, float, np.ndarray | None]:

    if fidelity_type is None:
        fidelity_type = process_fidelity
    if method is None:
        method = optax.adam

    # Вычисляем начальную infidelity
    params_jax = _to_jax_params(initial_params)
    input_infidelity = 1 - float(fidelity_type(gate, pulse, params_jax, tol))

    if not awesome_state.imitate_rydopt:
        if is_ask_tell(method):
            raise NotImplementedError('Ask-Tell Optimizers are not implemented on AwesomeRydopt yet')
        result = custom_optimize_gradient(gate, pulse, initial_params, fixed_initial_params,
                                          min_initial_params, max_initial_params,
                                          num_steps=num_steps, learning_rate=learning_rate, tol=tol,
                                          return_history=return_history, verbose=verbose, method=method,
                                          fidelity_type=fidelity_type, apply_bounds=apply_bounds,
                                          return_best=return_best)
    else:
        result = ro_optimize(gate, pulse, initial_params, fixed_initial_params,
                             num_steps=num_steps, learning_rate=learning_rate, tol=tol,
                             return_history=return_history, verbose=verbose)

    # Логируем шаг
    awesome_state.log_step(
        method='optimize',
        input_params=initial_params,
        output_params=result.params,
        input_infidelity=input_infidelity,
        output_infidelity=float(result.infidelity),
        extra={
            'num_steps': num_steps,
            'learning_rate': learning_rate,
            'tol': tol,

        }
    )
    return result


def multi_start_optimize(
        gate: GateSystem,
        pulse: PulseAnsatz,
        min_initial_params: PulseParams,
        max_initial_params: PulseParams,
        fixed_initial_params: FixedPulseParams | None = None,
        *,
        num_steps: int = 1000,
        learning_rate: float = 0.05,
        tol: float = 1e-7,
        num_initializations: int = 8,
        min_converged_initializations: int | None = None,
        num_processes: int | None = None,
        seed: int | None = None,
        return_history: bool = False,
        return_all: bool = False,
        verbose: bool = False,
        method: Callable = None,
        go_optimizer_settings: dict = None,
        fidelity_type: Callable = None,
        apply_bounds: bool = True,
        return_list_results: bool = False
) -> OptimizationResult | List[OptimizationResult]:
    if fidelity_type is None:
        fidelity_type = process_fidelity
    if method is None:
        method = optax.adam

    if awesome_state.imitate_rydopt:
        if is_ask_tell(method):
            raise NotImplementedError('Ask-Tell Optimizers are not implemented on AwesomeRydopt yet')
        else:
            return custom_multi_start_optimize_gradient(gate, pulse, min_initial_params, max_initial_params,
                                                        fixed_initial_params, num_steps=num_steps,
                                                        learning_rate=learning_rate, tol=tol,
                                                        return_history=return_history, verbose=verbose,
                                                        num_processes=num_processes,
                                                        num_initializations=num_initializations, return_all=return_all,
                                                        seed=seed,
                                                        min_converged_initializations=min_converged_initializations,
                                                        method=method, fidelity_type=fidelity_type,
                                                        apply_bounds=apply_bounds,
                                                        return_list_results=return_list_results)
    else:
        return ro_multi_start_optimize(gate, pulse, min_initial_params, max_initial_params,
                                       fixed_initial_params, num_steps=num_steps,
                                       learning_rate=learning_rate, tol=tol,
                                       return_history=return_history, verbose=verbose, num_processes=num_processes,
                                       num_initializations=num_initializations, return_all=return_all, seed=seed,
                                       min_converged_initializations=min_converged_initializations)


def sequence_optimize(
        gate: GateSystem,
        pulse: PulseAnsatz,
        initial_params: PulseParams,
        fixed_initial_params: FixedPulseParams | None = None,
        min_initial_params: PulseParams = None,
        max_initial_params: PulseParams = None,
        *,
        num_steps: int | Iterable[int] = 1000,
        learning_rate: float | Iterable[float] = 0.005,
        tol: float | Iterable[float] = 1e-7,
        return_history: bool = False,
        verbose: bool = False,
        method: Callable | Iterable[Callable] = None,
        fidelity_type: Callable | Iterable[Callable] = None,
        apply_bounds: bool = True,
        return_best: bool = True,
) -> OptimizationResult[PulseParams, float, np.ndarray | None]:
    if fidelity_type is None:
        fidelity_type = process_fidelity
    if method is None:
        method = optax.adam

    specs = make_full_specs(num_steps, learning_rate, tol,
                            method, fidelity_type)
    num_steps_specs, learning_rate_specs, tol_specs, method_specs, fidelity_type_specs = specs

    point = initial_params
    opt_result = None
    #print(f'Optimize sequence is started with {len(learning_rate_specs)} steps')

    for s, l, t, m, f in zip(num_steps_specs, learning_rate_specs, tol_specs, method_specs, fidelity_type_specs):
      #  print('\n\n\n')
       # print('＼(°O°)／' * 20)
       # print(f'STEP STARTED WITH:')
       # print(f'Steps Amount {s}')
       # print(f'Learning Rate: {l}')
       # print(f'Tolerance: {t}')
       # print(f'Method: {m.__name__}')
       # print(f'Fidelity Type: {f.__name__}')
       # print('＼(°O°)／' * 20)

        opt_result = optimize(gate, pulse, point, fixed_initial_params, min_initial_params, max_initial_params,
                              num_steps=s, learning_rate=l, tol=t, return_history=return_history,
                              verbose=verbose, method=m, fidelity_type=f, apply_bounds=apply_bounds,
                              return_best=return_best)
        point = opt_result.params

    return opt_result


def population_optimize(
        gate: GateSystem,
        pulse: PulseAnsatz,
        initial_params: PulseParams,
        min_initial_params: PulseParams,
        max_initial_params: PulseParams,
        fixed_initial_params: FixedPulseParams | None = None,
        *,
        num_generations: int = 200,
        population_size: int | None = None,
        optimizer_class: type = None,
        optimizer_kwargs: dict | None = None,
        tol: float = 1e-7,
        return_history: bool = False,
        verbose: bool = False,
        fidelity_type: Callable = None,
        apply_bounds: bool = True,
        return_best: bool = True,
) -> OptimizationResult:

    from src.optimization.optimize import population_optimize as _population_optimize

    if fidelity_type is None:
        fidelity_type = process_fidelity
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    # Вычисляем начальную infidelity
    params_jax = _to_jax_params(initial_params)
    input_infidelity = 1 - float(fidelity_type(gate, pulse, params_jax, tol))

    result = _population_optimize(
        gate=gate,
        pulse=pulse,
        initial_params=initial_params,
        min_initial_params=min_initial_params,
        max_initial_params=max_initial_params,
        fixed_initial_params=fixed_initial_params,
        num_generations=num_generations,
        population_size=population_size,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        tol=tol,
        return_history=return_history,
        verbose=verbose,
        fidelity_type=fidelity_type,
        apply_bounds=apply_bounds,
        return_best=return_best
    )

    awesome_state.log_step(
        method='population_optimize',
        input_params=initial_params,
        output_params=result.params,
        input_infidelity=input_infidelity,
        output_infidelity=float(result.infidelity),
        extra={
            'num_generations': num_generations,
            'population_size': population_size,
            'tol': tol,
            'optimizer': optimizer_class.__name__ if optimizer_class else 'CMAOptimizer'
        }
    )
    return result