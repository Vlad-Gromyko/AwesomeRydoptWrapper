import multiprocessing as mp
import sys
import threading
import time
from collections.abc import Callable, Sized
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from queue import SimpleQueue
from types import TracebackType
from typing import Any, Generic, Literal, Protocol, TypeVar, overload, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm.auto import tqdm

from rydopt.protocols import GateSystem
from rydopt.pulses.pulse_ansatz import PulseAnsatz
from rydopt.simulation.fidelity import process_fidelity
from rydopt.types import FixedPulseParams, PulseParams




from typing import Union, List

tqdm.monitor_interval = 0

# Portal colors using ANSI codes
PORTAL_BLUE = "\033[94m"      # ярко-синий
PORTAL_ORANGE = "\033[38;5;208m"    # ярко-жёлтый (ближе к оранжевому)
PORTAL_RESET = "\033[0m"

ParamsType = TypeVar("ParamsType", covariant=True)
ValueType = TypeVar("ValueType", covariant=True)
HistoryType = TypeVar("HistoryType", covariant=True)


@dataclass
class OptimizationResult(Generic[ParamsType, ValueType, HistoryType]):
    r"""Data class that stores the results of a gate pulse optimization.

    Attributes:
        params: Final pulse parameters.
        infidelity: Final cost function evaluation.
        duration: Final duration
        infidelity_history: Cost function evaluations during the optimization.
        duration_history: Durations during the optimization.
        grad_norm_history: Norm of the parameter gradient during the optimization.
        num_steps: Number of optimization steps.
        tol: Target gate infidelity.
        runtime_in_sec: Runtime of the optimization in seconds.

    """

    params: ParamsType  # type: ignore[misc]
    infidelity: ValueType  # type: ignore[misc]
    duration: ValueType  # type: ignore[misc]
    infidelity_history: HistoryType  # type: ignore[misc]
    duration_history: HistoryType  # type: ignore[misc]
    grad_norm_history: HistoryType  # type: ignore[misc]
    num_steps: int
    tol: float
    runtime_in_sec: float


def _to_jax_params(params: PulseParams) -> PulseParams:
    return tuple(jnp.asarray(p) if isinstance(p, (np.ndarray, list, tuple)) else p for p in params)


def custom_optimize_gradient(
        gate: GateSystem,
        pulse: PulseAnsatz,
        initial_params: PulseParams,
        fixed_initial_params: FixedPulseParams | None = None,
        min_initial_params: PulseParams = None,
        max_initial_params: PulseParams = None,
        *,
        num_steps: int = 1000,
        learning_rate: float = 0.05,
        tol: float = 1e-7,
        return_history: bool = False,
        verbose: bool = False,
        method: Callable = None,
        fidelity_type: Callable = None,
        apply_bounds: bool = False,
        return_best: bool = True,
) -> OptimizationResult[PulseParams, float, np.ndarray | None]:

    flat_min = None
    flat_max = None

    split_indices = _spec(initial_params)

    params_full = _ravel(initial_params)

    if apply_bounds:
        if min_initial_params is not None and max_initial_params is not None:
            flat_min = _ravel(min_initial_params)
            flat_max = _ravel(max_initial_params)
        else:
            print('No bounds specified')

    if fixed_initial_params is None:
        trainable_mask = np.ones_like(params_full, dtype=bool)
    else:
        trainable_mask = ~_ravel(fixed_initial_params).astype(bool)
    trainable_indices = np.nonzero(trainable_mask)[0]

    params_trainable = params_full[trainable_indices]

    # --- Optimize parameters ---

    #print("Started optimization using 1 process\n")
    print(f"{PORTAL_BLUE}◧ Gradient Optimization ◧{PORTAL_RESET}")

    t0 = time.perf_counter()
    with _ProgressBar(
            num_processes=1, num_steps=num_steps, min_converged_initializations=1, enable=not verbose
    ) as progress_queue:
        final_params_trainable, final_infidelity, infidelity_history, duration_history, grad_norm_history = (
            _adam_optimize(
                gate,
                pulse,
                params_full,
                params_trainable,
                trainable_indices,
                split_indices,
                num_steps,
                1,
                learning_rate,
                tol,
                0,
                None,
                progress_queue,
                return_history,
                method,
                fidelity_type,
                flat_min,
                flat_max,
            )
        )
    runtime = time.perf_counter() - t0

    if apply_bounds:
        # Применяем клиппинг к финальным обучаемым параметрам
        final_params_trainable = np.clip(final_params_trainable,
                                         flat_min[trainable_indices],
                                         flat_max[trainable_indices])

    final_full = params_full.copy()
    final_full[trainable_indices] = final_params_trainable

    final_params = _unravel(final_full, split_indices)
    num_converged = 1 if final_infidelity <= tol else 0

    # --- Logging ---
    if return_best:
        params_to_check = _to_jax_params(initial_params)
        check_initial_infidelity = 1 - float(fidelity_type(gate, pulse, params_to_check, tol))


        if check_initial_infidelity < float(final_infidelity):
            return OptimizationResult(
                params=initial_params,
                infidelity=float(check_initial_infidelity),
                duration=initial_params[0],
                infidelity_history=infidelity_history,
                duration_history=duration_history,
                grad_norm_history=grad_norm_history,
                num_steps=num_steps,
                tol=tol,
                runtime_in_sec=runtime,
            )
    _print_gate("Optimized gate:", final_params, float(final_infidelity), tol)

    return OptimizationResult(
        params=final_params,
        infidelity=float(final_infidelity),
        duration=final_params[0],
        infidelity_history=infidelity_history,
        duration_history=duration_history,
        grad_norm_history=grad_norm_history,
        num_steps=num_steps,
        tol=tol,
        runtime_in_sec=runtime,
    )


def custom_multi_start_optimize_gradient(
        gate: GateSystem,
        pulse: PulseAnsatz,
        min_initial_params: PulseParams,
        max_initial_params: PulseParams,
        fixed_initial_params: FixedPulseParams | None = None,
        *,
        num_steps: int = 1000,
        learning_rate: float = 0.05,
        tol: float = 1e-7,
        num_initializations: int = 10,
        min_converged_initializations: int | None = None,
        num_processes: int | None = None,
        seed: int | None = None,
        return_history: bool = False,
        return_all: bool = False,
        verbose: bool = False,
        method: Callable = None,
        fidelity_type: Callable = None,
        apply_bounds: bool = False,
        return_list_results: bool = False,


) -> OptimizationResult[PulseParams | list[PulseParams], float | np.ndarray, np.ndarray | None]:
    split_indices = _spec(min_initial_params)
    flat_min = _ravel(min_initial_params)
    flat_max = _ravel(max_initial_params)
    params_full = flat_min.copy()

    if fixed_initial_params is None:
        trainable_mask = np.ones_like(flat_min, dtype=bool)
    else:
        trainable_mask = ~_ravel(fixed_initial_params).astype(bool)
        if not np.allclose(flat_min[~trainable_mask], flat_max[~trainable_mask]):
            raise ValueError(
                "For fixed parameters, min_initial_params and max_initial_params must have identical values."
            )
    trainable_indices = np.nonzero(trainable_mask)[0]

    use_one_process_per_device = len(jax.devices()) > 1 or jax.devices()[0].platform != "cpu"
    if num_processes is None:
        num_processes = (
            len(jax.devices()) if use_one_process_per_device else max(1, mp.cpu_count() // 2)
        )  # the division by 2 avoids oversubscription
    elif use_one_process_per_device and num_processes > len(jax.devices()):
        raise ValueError(
            "If multiple devices or a GPU device is visible, num_processes must be smaller or equal "
            "to the number of devices."
        )

    # Pad the number of initial parameter samples to be a multiple of the number of processes
    pad = (-num_initializations) % num_processes
    padded_num_initializations = num_initializations + pad
    if pad != 0:
        print(
            f"Padding num_initializations from {num_initializations} to "
            f"{padded_num_initializations} to be a multiple of num_processes={num_processes}."
        )

    if min_converged_initializations is None:
        min_converged_initializations = padded_num_initializations

    # Initial parameter samples
    rng = np.random.default_rng(seed)
    params_trainable = flat_min[trainable_indices] + (
            flat_max[trainable_indices] - flat_min[trainable_indices]
    ) * rng.random(size=(padded_num_initializations, trainable_indices.size))

    # --- Optimize parameters ---

    print(f"Started optimization using {num_processes} {'processes' if num_processes > 1 else 'process'}\n")

    t0 = time.perf_counter()

    min_converged_initializations_local = (min_converged_initializations + num_processes - 1) // num_processes

    if not apply_bounds:
        flat_min = None
        flat_max = None

    if num_processes == 1:
        # Run optimization in main process
        with _ProgressBar(
                num_processes=num_processes,
                num_steps=num_steps,
                min_converged_initializations=min_converged_initializations_local,
                enable=not verbose,
        ) as progress_queue:
            final_params_trainable, final_infidelities, infidelity_history, duration_history, grad_norm_history = (
                _adam_optimize(
                    gate,
                    pulse,
                    params_full,
                    params_trainable,
                    trainable_indices,
                    split_indices,
                    num_steps,
                    min_converged_initializations_local,
                    learning_rate,
                    tol,
                    0,
                    None,
                    progress_queue,
                    return_history,
                    method,
                    fidelity_type,
                    flat_min,
                    flat_max
                )
            )

    else:
        # Run optimization in spawned processes
        chunks = np.array_split(params_trainable, num_processes, axis=0)

        ctx = mp.get_context("spawn")
        with (
            ctx.Manager() as manager,
            _ProgressBar(
                num_processes=num_processes,
                num_steps=num_steps,
                min_converged_initializations=min_converged_initializations_local,
                queue=manager.Queue(),
                enable=not verbose,
            ) as progress_queue,
            ctx.Pool(processes=num_processes) as pool,
        ):
            results = pool.starmap(
                _adam_optimize,
                [
                    (
                        gate,
                        pulse,
                        params_full,
                        p,
                        trainable_indices,
                        split_indices,
                        num_steps,
                        min_converged_initializations_local,
                        learning_rate,
                        tol,
                        device_idx,
                        device_idx if use_one_process_per_device else None,
                        progress_queue,
                        return_history,
                        method,
                        fidelity_type,
                        flat_min,
                        flat_max
                    )
                    for device_idx, p in enumerate(chunks)
                ],
            )

            # Concatenate results from all processes
            (
                final_params_trainable_list,
                final_infidelities_list,
                infidelity_history_list,
                duration_history_list,
                grad_norm_history_list,
            ) = zip(*results)
            final_params_trainable = np.concatenate(final_params_trainable_list, axis=0)
            final_infidelities = np.concatenate(final_infidelities_list, axis=0)

            if return_history:
                infidelity_history = np.concatenate(infidelity_history_list, axis=1)
                duration_history = np.concatenate(duration_history_list, axis=1)
                grad_norm_history = np.concatenate(grad_norm_history_list, axis=1)
            else:
                infidelity_history = None
                duration_history = None
                grad_norm_history = None

    runtime = time.perf_counter() - t0

    if apply_bounds:
        # Применяем клиппинг к финальным обучаемым параметрам
        final_params_trainable = np.clip(final_params_trainable,
                                         flat_min[trainable_indices],
                                         flat_max[trainable_indices])


    final_full = np.tile(params_full, (final_params_trainable.shape[0], 1))
    final_full[:, trainable_indices] = final_params_trainable

    converged = np.where(final_infidelities <= tol)[0]
    num_converged = len(converged)
    if num_converged == 0:
        converged = np.array([np.argmin(final_infidelities)])
    durations_converged = final_full[converged][:, 0]

    # --- Logging ---

    _print_summary(f"multi-start", runtime, tol, num_converged)

    fastest_idx = converged[np.argmin(durations_converged)]
    fastest_infidelity = final_infidelities[fastest_idx]
    fastest_params = _unravel(final_full[fastest_idx], split_indices)

    if num_converged > 1:
        # If multiple parameter sets converged, show slowest and fastest gate
        slowest_idx = converged[np.argmax(durations_converged)]
        slowest_infidelity = final_infidelities[slowest_idx]
        slowest_params = _unravel(final_full[slowest_idx], split_indices)

        _print_gate("Slowest gate:", slowest_params, slowest_infidelity, tol)
        _print_gate("Fastest gate:", fastest_params, fastest_infidelity, tol)

        idx = rng.integers(0, num_converged, size=(1024, num_converged))
        mins = np.asarray(durations_converged)[idx].min(axis=1)
        err = mins.std()
        print(f"> one-sided bootstrap error on duration: {err:.1g}")
    else:
        # Otherwise, show the gate with the smallest infidelity
        _print_gate("Best gate:", fastest_params, fastest_infidelity, tol)

    # --- Return value(s) ---

    if return_list_results:
        # Возвращаем список всех результатов (каждый запуск отдельно)
        results = []
        for i in range(final_full.shape[0]):
            params_i = _unravel(final_full[i], split_indices)
            inf_i = final_infidelities[i]
            if return_history:
                inf_hist = infidelity_history[:, i] if infidelity_history is not None else None
                dur_hist = duration_history[:, i] if duration_history is not None else None
                grad_hist = grad_norm_history[:, i] if grad_norm_history is not None else None
            else:
                inf_hist = dur_hist = grad_hist = None
            results.append(OptimizationResult(
                params=params_i,
                infidelity=float(inf_i),
                duration=float(params_i[0]),
                infidelity_history=inf_hist,
                duration_history=dur_hist,
                grad_norm_history=grad_hist,
                num_steps=num_steps,
                tol=tol,
                runtime_in_sec=0.0,  # можно передать runtime, если нужно
            ))
        return results
    elif return_all:
        sorter = np.argsort(final_infidelities)
        final_full_sorted = final_full[sorter]
        infidelity_history_out = infidelity_history[:, sorter] if infidelity_history is not None else None
        duration_history_out = duration_history[:, sorter] if duration_history is not None else None
        grad_norm_history_out = grad_norm_history[:, sorter] if grad_norm_history is not None else None
        return OptimizationResult(
            params=[_unravel(p, split_indices) for p in final_full_sorted],
            infidelity=final_infidelities[sorter],
            duration=final_full_sorted[:, 0],
            infidelity_history=infidelity_history_out,
            duration_history=duration_history_out,
            grad_norm_history=grad_norm_history_out,
            num_steps=num_steps,
            tol=tol,
            runtime_in_sec=runtime,
        )
    else:
        fastest_idx = np.argmin(final_infidelities)
        infidelity_history_out = infidelity_history[:, fastest_idx] if infidelity_history is not None else None
        duration_history_out = duration_history[:, fastest_idx] if duration_history is not None else None
        grad_norm_history_out = grad_norm_history[:, fastest_idx] if grad_norm_history is not None else None
        return OptimizationResult(
            params=_unravel(final_full[fastest_idx], split_indices),
            infidelity=final_infidelities[fastest_idx],
            duration=_unravel(final_full[fastest_idx], split_indices)[0],
            infidelity_history=infidelity_history_out,
            duration_history=duration_history_out,
            grad_norm_history=grad_norm_history_out,
            num_steps=num_steps,
            tol=tol,
            runtime_in_sec=runtime,
        )

# -----------------------------------------------------------------------------
# Progress bar
# -----------------------------------------------------------------------------


class _ProgressQueue(Protocol):
    def put(self, item: Any) -> None: ...

    def get(self) -> Any: ...


class _ProgressBar:
    def __init__(
            self,
            num_processes: int,
            num_steps: int,
            min_converged_initializations: int,
            queue: _ProgressQueue | None = None,
            enable: bool = True,
    ) -> None:
        self._num_processes = num_processes
        self._num_steps = num_steps
        self._min_converged_initializations = min_converged_initializations
        self._external_queue = queue
        self._queue: _ProgressQueue = queue or SimpleQueue()
        self._listener: threading.Thread | None = None
        self._enable = enable

    def __enter__(self) -> _ProgressQueue | None:
        if not self._enable:
            return None
        self._listener = threading.Thread(
            target=self._progress_listener,
            daemon=True,
        )
        self._listener.start()
        return self._queue

    def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
    ) -> None:
        if not self._enable:
            return
        for proc_idx in range(self._num_processes):
            self._queue.put(("done", proc_idx, 0, 0, 0))
        if self._listener is not None:
            self._listener.join()

    @staticmethod
    def make_progress_hook(
            queue: _ProgressQueue | None,
    ) -> Callable[[tuple[int, int, float, int]], None] | None:
        if queue is None:
            return None

        def progress_hook(args: tuple[int, int, float, int]) -> None:
            process_idx, step, infidelity, converged = args
            queue.put(
                (
                    "update",
                    int(process_idx),
                    int(step),
                    float(infidelity),
                    int(converged),
                )
            )

        return progress_hook

    def _progress_listener(self) -> None:
        bars: dict[int, tqdm] = {}
        finished: set[int] = set()

        while len(finished) < self._num_processes:
            kind, proc_idx, step, min_inf, converged = self._queue.get()

            if kind == "update":
                bar = bars.get(proc_idx)
                if bar is None:
                    bar = tqdm(
                        total=self._num_steps,
                        desc=f"{PORTAL_BLUE}proc{proc_idx:02d}{PORTAL_RESET}",
                        position=proc_idx,
                        file=sys.stdout,
                        dynamic_ncols=True,
                    )
                    bars[proc_idx] = bar

                bar.n = step + 1
                bar.set_postfix_str(
                    f"{PORTAL_ORANGE}infidelity={min_inf:.2e}, converged={converged}/{self._min_converged_initializations}{PORTAL_RESET}",
                    refresh=False
                )
                bar.refresh()

            elif kind == "done":
                finished.add(proc_idx)
                bar = bars.pop(proc_idx, None)
                if bar is not None:
                    if bar.n < self._num_steps:
                        bar.n = self._num_steps
                    bar.close()


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def _spec(nested: PulseParams | FixedPulseParams) -> tuple[int, ...]:
    return tuple(np.cumsum([len(p) if isinstance(p, Sized) else 1 for p in nested])[:-1].tolist())


def _ravel(nested: PulseParams | FixedPulseParams) -> np.ndarray:
    first, *rest = nested
    return np.concatenate([(first,), *list(rest)])


def _unravel(flat: np.ndarray, split_indices: tuple[int, ...]) -> PulseParams | FixedPulseParams:
    parts = np.split(flat, split_indices)
    return (parts[0][0], *tuple(parts[1:]))  # type: ignore[return-value]


def _unravel_jax(flat: jnp.ndarray, split_indices: tuple[int, ...]) -> PulseParams | FixedPulseParams:
    parts = jnp.split(flat, split_indices)
    return (parts[0][0], *tuple(parts[1:]))  # type: ignore[return-value]


def _make_infidelity(
        gate: GateSystem,
        pulse: PulseAnsatz,
        params_full: np.ndarray,
        params_trainable_indices: np.ndarray,
        params_split_indices: tuple,
        tol: float,
        fidelity_type: Callable,
        flat_min = None,
        flat_max = None,
):
    full = jnp.asarray(params_full)
    trainable_indices = jnp.asarray(params_trainable_indices)
    if flat_min is not None:
        # Границы для обучаемых параметров
        min_trainable = jnp.asarray(flat_min[trainable_indices])
        max_trainable = jnp.asarray(flat_max[trainable_indices])
    else:
        min_trainable = max_trainable = None

    def bounded_infidelity(params_trainable):
        if min_trainable is not None:
            params_trainable = jnp.clip(params_trainable, min_trainable, max_trainable)
        params = full.at[trainable_indices].set(params_trainable)
        params_tuple = _unravel_jax(params, params_split_indices)
        return jnp.abs(1 - fidelity_type(gate, pulse, params_tuple, tol))

    def infidelity(params_trainable):
        params = full.at[trainable_indices].set(params_trainable)
        params_tuple = _unravel_jax(params, params_split_indices)
        return jnp.abs(1 - fidelity_type(gate, pulse, params_tuple, tol))

    return bounded_infidelity if flat_min is not None else infidelity



def _print_gate(title: str, params, infidelity: float, tol: float):
    print(f"\n{title}")
    if abs(float(infidelity)) < tol:
        print("> infidelity <= tol")
    else:
        print(f"> infidelity = {infidelity:.6e}")
    print(f"> parameters = ({', '.join(str(p) for p in params)})")
    print(f"> duration = {params[0]}")


def _print_summary(method_name: str, runtime: float, tol: float, num_converged: int):
    print(f"\n=== Optimization finished using {method_name} ===\n")
    print(f"Runtime: {runtime:.3f} seconds")
    print(f"Gates with infidelity below tol={tol:.1e}: {num_converged}")


# -----------------------------------------------------------------------------
# Internal jax.jit-ed Adam optimization scan loop
# -----------------------------------------------------------------------------

@partial(
    jax.jit,
    static_argnames=[
        "infidelity_and_grad",
        "optimizer",
        "num_steps",
        "min_converged_initializations",
        "progress_hook",
        "return_history",
        "is_batched",
    ],
    donate_argnames=["params_trainable"],
)
def _adam_scan(
        infidelity_and_grad,
        optimizer: optax.GradientTransformation,
        params_trainable,
        num_steps: int,
        min_converged_initializations: int,
        process_idx: int,
        tol: float | jnp.ndarray,
        progress_hook,
        return_history: bool,
        is_batched: bool = False,
):
    opt_state0 = optimizer.init(params_trainable)

    if is_batched:
        batch_size = params_trainable.shape[0]
        zeros_like_tol = jnp.zeros((batch_size,))
        best_params = params_trainable
        best_inf = jnp.full((batch_size,), jnp.inf)
    else:
        batch_size = 1
        zeros_like_tol = jnp.zeros_like(tol)
        best_params = params_trainable
        best_inf = jnp.full_like(tol, jnp.inf)

    carry0 = (params_trainable, params_trainable, zeros_like_tol, opt_state0, 0, zeros_like_tol, best_params, best_inf)

    def body(carry, step):
        (params, new_params, inf, opt_state, prev_converged, grad_norm, best_params_cur, best_inf_cur) = carry

        def do_step(carry):
            (_, params, _, opt_state, _, _, best_params_cur, best_inf_cur) = carry

            infidelity, grads = infidelity_and_grad(params)
            converged_initializations = jnp.sum(infidelity <= tol)

            updates, opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            grad_norm = jnp.linalg.norm(grads, axis=-1) if return_history else jnp.zeros_like(infidelity)

            return (params, new_params, infidelity, opt_state, converged_initializations, grad_norm, best_params_cur, best_inf_cur)

        was_not_done = prev_converged < min_converged_initializations
        carry = jax.lax.cond(was_not_done, do_step, lambda c: c, operand=carry)

        (params, new_params, inf, opt_state, converged, grad_norm, best_params_cur, best_inf_cur) = carry

        if is_batched:
            improved = inf < best_inf_cur
            new_best_params = jnp.where(improved[:, None], params, best_params_cur)
            new_best_inf = jnp.where(improved, inf, best_inf_cur)
        else:
            improved = inf < best_inf_cur
            new_best_params = jnp.where(improved, params, best_params_cur)
            new_best_inf = jnp.where(improved, inf, best_inf_cur)

        is_done_now = converged >= min_converged_initializations
        is_distinct = (step % 20 == 0) | (step == num_steps - 1)
        should_log = was_not_done & (is_done_now | is_distinct)

        if progress_hook is not None:
            jax.lax.cond(
                should_log,
                lambda args: jax.debug.callback(progress_hook, args),
                lambda _: None,
                operand=(process_idx, step, jnp.min(inf), converged),
            )
        else:
            jax.lax.cond(
                should_log,
                lambda args: jax.debug.print(
                    "Step {step} [proc{process_idx}]: infidelity = {min_infidelity}, "
                    "converged = {converged} / {min_converged_initializations}",
                    step=args[0],
                    process_idx=args[1],
                    min_infidelity=args[2],
                    converged=args[3],
                    min_converged_initializations=args[4],
                ),
                lambda _: None,
                operand=(
                    step,
                    process_idx,
                    jnp.min(inf),
                    converged,
                    min_converged_initializations,
                ),
            )



        new_carry = (params, new_params, inf, opt_state, converged, grad_norm, new_best_params, new_best_inf)
        if return_history:
            return new_carry, (inf, params[..., 0], grad_norm)
        else:
            return new_carry, None

    (_, _, _, _, _, _, final_best_params, final_best_inf), history = jax.lax.scan(
        body,
        carry0,
        jnp.arange(num_steps),
    )

    return (final_best_params, final_best_inf, history)


# -----------------------------------------------------------------------------
# Internal Adam optimization helper
# -----------------------------------------------------------------------------


def _adam_optimize(
        gate: GateSystem,
        pulse: PulseAnsatz,
        params_full: np.ndarray,
        params_trainable: np.ndarray,
        params_trainable_indices: np.ndarray,
        params_split_indices: tuple,
        num_steps: int,
        min_converged_initializations: int,
        learning_rate: float,
        tol: float,
        process_idx: int,
        device_idx: int | None,
        progress_queue: _ProgressQueue | None,
        return_history: bool,
        method: Callable,
        fidelity_type: Callable,
        flat_min = None,
        flat_max= None,

) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    device_ctx = nullcontext() if device_idx is None else jax.default_device(jax.devices()[device_idx])

    progress_hook = _ProgressBar.make_progress_hook(progress_queue)

    with device_ctx:
        trainable = jnp.asarray(params_trainable)
        optimizer = method(learning_rate)
        infidelity = _make_infidelity(
            gate,
            pulse,
            params_full,
            params_trainable_indices,
            params_split_indices,
            tol,
            fidelity_type,
            flat_min,
            flat_max
        )

        if trainable.ndim == 1:
            infidelity_and_grad = jax.value_and_grad(infidelity)
            tol_arg: float | jnp.ndarray = tol
        else:
            infidelity_and_grad = jax.vmap(jax.value_and_grad(infidelity))
            tol_arg = jnp.full((trainable.shape[0],), tol)

        final_params, final_infidelities, history = _adam_scan(
            infidelity_and_grad=infidelity_and_grad,
            optimizer=optimizer,
            params_trainable=trainable,
            num_steps=num_steps,
            min_converged_initializations=min_converged_initializations,
            process_idx=process_idx,
            tol=tol_arg,
            progress_hook=progress_hook,
            return_history=return_history,
            is_batched=(trainable.ndim > 1),

        )

        if return_history:
            infidelity_history = np.array(history[0])
            duration_history = np.array(history[1])
            grad_norm_history = np.array(history[2])
        else:
            infidelity_history = None
            duration_history = None
            grad_norm_history = None

        return (
            np.array(final_params),
            np.array(final_infidelities),
            infidelity_history,
            duration_history,
            grad_norm_history,
        )


def population_optimize(
        gate: GateSystem,
        pulse: PulseAnsatz,
        initial_params: PulseParams,  # НОВЫЙ ПАРАМЕТР - хорошее начальное решение
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

) -> OptimizationResult[PulseParams, float, np.ndarray | None]:

    from src.optimization.population.cma_optimizer import CMAOptimizer

    if fidelity_type is None:
        fidelity_type = process_fidelity
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    if optimizer_class is None:
        optimizer_class = CMAOptimizer

    # Подготовка параметров
    split_indices = _spec(min_initial_params)
    flat_min = _ravel(min_initial_params)
    flat_max = _ravel(max_initial_params)

    # Маска обучаемых параметров
    if fixed_initial_params is None:
        trainable_mask = np.ones_like(flat_min, dtype=bool)
    else:
        flat_fixed = _ravel(fixed_initial_params)
        trainable_mask = ~flat_fixed.astype(bool)
    trainable_indices = np.nonzero(trainable_mask)[0]

    dimension = len(trainable_indices)

    # Границы
    bounds = None
    flat_min_jax = jnp.asarray(flat_min)
    flat_max_jax = jnp.asarray(flat_max)

    if apply_bounds:
        bounds = (flat_min[trainable_indices], flat_max[trainable_indices])

    # Подготовка начального среднего (если передано)
    initial_mean_flat = None
    if initial_params is not None:
        # Конвертируем PulseParams в плоский массив
        initial_mean_flat_np = _ravel(initial_params)
        initial_mean_flat = initial_mean_flat_np[trainable_indices]
        if verbose:
            print(f"Using provided initial mean for optimizer")
            # Вычисляем infidelity начального решения
            try:
                # Создаем JAX массив для вычисления
                initial_mean_jax = jnp.asarray(initial_mean_flat)
                # Временно создаем функцию для вычисления
                full = flat_min_jax
                trainable_indices_jax = jnp.asarray(trainable_indices)
                if apply_bounds:
                    min_trainable = flat_min_jax[trainable_indices_jax]
                    max_trainable = flat_max_jax[trainable_indices_jax]
                else:
                    min_trainable = max_trainable = None

                def temp_inf(params):
                    if min_trainable is not None:
                        params = jnp.clip(params, min_trainable, max_trainable)
                    p = full.at[trainable_indices_jax].set(params)
                    pt = _unravel_jax(p, split_indices)
                    return jnp.abs(1 - fidelity_type(gate, pulse, pt, tol))

                init_inf = float(temp_inf(initial_mean_jax))
                print(f"  Initial mean infidelity: {init_inf:.6e}")
            except:
                pass

    # Создаем словарь параметров для конструктора оптимизатора
    init_kwargs = optimizer_kwargs.copy()
    if population_size is not None:
        init_kwargs['population_size'] = population_size
    if bounds is not None:
        init_kwargs['bounds'] = bounds
    if initial_mean_flat is not None:
        init_kwargs['mean'] = initial_mean_flat

    optimizer = optimizer_class(dimension=dimension, **init_kwargs)

    # Создаем JAX функцию infidelity
    full = flat_min_jax
    trainable_indices_jax = jnp.asarray(trainable_indices)

    if apply_bounds:
        min_trainable = flat_min_jax[trainable_indices_jax]
        max_trainable = flat_max_jax[trainable_indices_jax]
    else:
        min_trainable = max_trainable = None

    def infidelity_func(params_trainable: jnp.ndarray) -> jnp.ndarray:
        if min_trainable is not None:
            params_trainable = jnp.clip(params_trainable, min_trainable, max_trainable)
        params = full.at[trainable_indices_jax].set(params_trainable)
        params_tuple = _unravel_jax(params, split_indices)
        return jnp.abs(1 - fidelity_type(gate, pulse, params_tuple, tol))

    infidelity_vmap = jax.vmap(infidelity_func)

    if verbose:
        print(f"\n=== Population Optimization ===\n")
        print(f"Optimizer: {optimizer.__class__.__name__}")
        print(f"Dimension: {dimension}")
        print(f"Population size: {optimizer.population_size}")
        print(f"Generations: {num_generations}")
        if initial_mean_flat is not None:
            print(f"Initial mean provided: Yes (first few values: {initial_mean_flat[:3]}...)")
        print()

    # Прогресс-бар
    from tqdm.auto import tqdm

    t0 = time.perf_counter()
    best_inf_history = [] if return_history else None
    best_inf = float('inf')
    best_params = None

    pbar = tqdm(
        total=num_generations,
        desc=f"{PORTAL_BLUE}Global Optimization{PORTAL_RESET}",
        disable=not verbose,
        dynamic_ncols=True,
    )

    for generation in range(num_generations):
        population_np = np.array([optimizer.ask() for _ in range(optimizer.population_size)])
        population_jax = jnp.asarray(population_np)
        infidelities_jax = infidelity_vmap(population_jax)
        infidelities_np = np.array(infidelities_jax)

        solutions = list(zip(population_np, infidelities_np))
        optimizer.tell(solutions)

        current_best_params, current_best_inf = optimizer.result()

        if current_best_inf < best_inf:
            best_inf = current_best_inf
            best_params = current_best_params.copy()

        if return_history:
            best_inf_history.append(best_inf)

        pbar.set_postfix_str(f"{PORTAL_ORANGE}infidelity={best_inf:.2e}{PORTAL_RESET}")
        pbar.update(1)

        if best_inf <= tol:
            tqdm.write(f"{PORTAL_ORANGE}✓ Target infidelity {tol:.1e} reached at generation {generation}{PORTAL_RESET}")
            break

    pbar.close()
    runtime = time.perf_counter() - t0

    # Финальный результат
    if best_params is None:
        best_params, best_inf = optimizer.result()

    final_full_jax = full.at[trainable_indices_jax].set(jnp.asarray(best_params))
    final_params = _unravel_jax(final_full_jax, split_indices)

    final_params_converted = []
    for p in final_params:
        if hasattr(p, 'shape') and p.shape == ():
            final_params_converted.append(float(p))
        elif hasattr(p, 'tolist'):
            final_params_converted.append(p.tolist())
        else:
            final_params_converted.append(p)
    final_params = tuple(final_params_converted)

    if verbose:
        print(f"\n=== Optimization finished using {optimizer.__class__.__name__} ===\n")
        print(f"Runtime: {runtime:.3f} seconds")
        if best_inf <= tol:
            print(f"✓ Gate infidelity below tol={tol:.1e}")
        else:
            print(f"⚠️  Final infidelity = {best_inf:.6e} (tol={tol:.1e} not reached)")
        print(f"Final duration: {final_params[0]:.4f}")

    return_history_actual = return_history

    if return_best:
        params_to_check = _to_jax_params(initial_params)
        check_initial_infidelity = 1 - float(fidelity_type(gate, pulse, params_to_check, tol))

        if check_initial_infidelity < float(best_inf):
            return OptimizationResult(
                params=initial_params,
                infidelity=float(check_initial_infidelity),
                duration=initial_params[0],
                infidelity_history=None,
                duration_history=None,
                grad_norm_history=None,
                num_steps=0,
                tol=tol,
                runtime_in_sec=runtime,
            )

    return OptimizationResult(
        params=final_params,
        infidelity=float(best_inf),
        duration=float(final_params[0]),
        infidelity_history=np.array(best_inf_history) if return_history_actual else None,
        duration_history=None,
        grad_norm_history=None,
        num_steps=len(best_inf_history) if best_inf_history else generation + 1,
        tol=tol,
        runtime_in_sec=runtime,
    )