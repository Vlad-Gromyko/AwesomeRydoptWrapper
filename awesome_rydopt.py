import optax

from src.optimization.optimize import *

from rydopt.optimization.optimize import optimize as ro_optimize
from rydopt.optimization.optimize import multi_start_optimize as ro_multi_start_optimize
from rydopt.simulation.fidelity import process_fidelity

from typing import Union, List, Tuple, Callable, Iterable, Optional

from rydopt.protocols import GateSystem, PulseAnsatzLike
from rydopt.simulation.evolve import evolve
from rydopt.types import PulseParams

from itertools import cycle, islice
from src.container_tools import sort_by_fidelity_function, filter_by_fidelity_function


def check_fidelity(fidelity_func, gate: GateSystem, pulse: PulseAnsatzLike, params: PulseParams, tol: float = 1e-7):
    return fidelity_func(gate, pulse, params, tol)


class AwesomeState:
    def __init__(self):
        self.__imitate_rydopt = True

    @property
    def imitate_rydopt(self):
        return self.__imitate_rydopt

    @imitate_rydopt.setter
    def imitate_rydopt(self, value: bool = True):
        self.__imitate_rydopt = value


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


from itertools import cycle, islice


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

    if not awesome_state.imitate_rydopt:
        if is_ask_tell(method):
            raise NotImplementedError('Ask-Tell Optimizers are not implemented on AwesomeRydopt yet')
        else:
            return custom_optimize_gradient(gate, pulse, initial_params, fixed_initial_params,
                                            min_initial_params,
                                            max_initial_params,
                                            num_steps=num_steps,
                                            learning_rate=learning_rate, tol=tol,
                                            return_history=return_history, verbose=verbose, method=method,
                                            fidelity_type=fidelity_type,
                                            apply_bounds=apply_bounds,
                                            return_best=return_best)

    else:
        return ro_optimize(gate, pulse, initial_params, fixed_initial_params, num_steps=num_steps,
                           learning_rate=learning_rate, tol=tol,
                           return_history=return_history, verbose=verbose)


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
    print(f'Optimize sequence is started with {len(learning_rate_specs)} steps')

    for s, l, t, m, f in zip(num_steps_specs, learning_rate_specs, tol_specs, method_specs, fidelity_type_specs):
        print('\n\n\n')
        print('＼(°O°)／' * 20)

        print(f'STEP STARTED WITH:')
        print(f'Steps Amount {s}')
        print(f'Learning Rate: {l}')
        print(f'Tolerance: {t}')
        print(f'Method: {m.__name__}')
        print(f'Fidelity Type: {f.__name__}')

        print('＼(°O°)／' * 20)

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

    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    return _population_optimize(
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
