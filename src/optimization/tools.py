
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
from typing import Any, Generic, Literal, Protocol, TypeVar, overload

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm.auto import tqdm

from rydopt.protocols import GateSystem
from rydopt.pulses.pulse_ansatz import PulseAnsatz
from rydopt.simulation.fidelity import process_fidelity
from rydopt.types import FixedPulseParams, PulseParams

tqdm.monitor_interval = 0

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
                        desc=f"proc{proc_idx:02d}",
                        position=proc_idx,
                        file=sys.stdout,
                        dynamic_ncols=True,
                    )
                    bars[proc_idx] = bar

                bar.n = step + 1
                bar.set_postfix(
                    {
                        "infidelity": f"{min_inf:.2e}",
                        "converged": f"{converged}/{self._min_converged_initializations}",
                    },
                    refresh=False,
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
):
    full = jnp.asarray(params_full)
    trainable_indices = jnp.asarray(params_trainable_indices)

    def infidelity(params_trainable):
        params = full.at[trainable_indices].set(params_trainable)
        params_tuple = _unravel_jax(params, params_split_indices)
        return jnp.abs(1 - process_fidelity(gate, pulse, params_tuple, tol))

    return infidelity


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
):
    opt_state0 = optimizer.init(params_trainable)

    def body(carry, step):
        _, _, _, _, prev_converged_initializations, _ = carry

        # Do an gradient descent step if the optimization was not yet done. Note that 'params' and
        # not 'new_params' contains the parameters that correspond to the 'infidelity'.
        def do_step(carry):
            _, params, _, opt_state, _, _ = carry

            infidelity, grads = infidelity_and_grad(params)
            converged_initializations = jnp.sum(infidelity <= tol)

            updates, opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            grad_norm = jnp.linalg.norm(grads, axis=-1) if return_history else jnp.zeros_like(tol)

            return (
                params,
                new_params,
                infidelity,
                opt_state,
                converged_initializations,
                grad_norm,
            )

        was_not_done = prev_converged_initializations < min_converged_initializations
        carry = jax.lax.cond(was_not_done, do_step, lambda carry: carry, operand=carry)

        params, _, infidelity, _, converged_initializations, grad_norm = carry

        # Log intermediate results at distinct steps
        is_done_now = converged_initializations >= min_converged_initializations
        is_distinct = (step % 20 == 0) | (step == num_steps - 1)
        should_log = was_not_done & (is_done_now | is_distinct)

        if progress_hook is not None:
            jax.lax.cond(
                should_log,
                lambda args: jax.debug.callback(progress_hook, args),
                lambda _: None,
                operand=(process_idx, step, jnp.min(infidelity), converged_initializations),
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
                    jnp.min(infidelity),
                    converged_initializations,
                    min_converged_initializations,
                ),
            )

        if return_history:
            return carry, (infidelity, params[..., 0], grad_norm)
        return carry, None

    (final_params, _, final_infidelity, _, _, _), history = jax.lax.scan(
        body,
        (params_trainable, params_trainable, jnp.zeros_like(tol), opt_state0, 0, jnp.zeros_like(tol)),
        jnp.arange(num_steps),
    )

    return (final_params, final_infidelity, history)


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    device_ctx = nullcontext() if device_idx is None else jax.default_device(jax.devices()[device_idx])

    progress_hook = _ProgressBar.make_progress_hook(progress_queue)

    with device_ctx:
        trainable = jnp.asarray(params_trainable)
        optimizer = optax.adam(learning_rate)
        infidelity = _make_infidelity(
            gate,
            pulse,
            params_full,
            params_trainable_indices,
            params_split_indices,
            tol,
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
