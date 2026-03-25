from rydopt.protocols import GateSystem
from rydopt.pulses.pulse_ansatz import PulseAnsatz
from rydopt.simulation.evolve import evolve
from rydopt.types import PulseParams
import jax.numpy as jnp
import jax


def custom_robust_shift(gate: GateSystem, pulse: PulseAnsatz, params: PulseParams, tol: float = 1e-10) -> jnp.ndarray:
    items = jnp.array([0.99, 0.995, 1.0, 1.005, 1.01])


    def compute_single_fidelity(item):

        shifted_rabi = params[3].at[0].set(params[3][0] * item)
        shifted_params = (params[0], params[1], params[2], shifted_rabi)


        final_state = evolve(gate, pulse, shifted_params, tol)

        return gate.process_fidelity(final_state)

    fidelities = jax.vmap(compute_single_fidelity)(items)

    F_var = jnp.var(fidelities)

    F_mean = jnp.sum(fidelities) / 5

    slope_penalty = jnp.sum(jnp.abs(fidelities[1:] - fidelities[:-1]))

    alpha = 0.3
    beta = 0.1

    return F_mean - alpha * F_var - beta * slope_penalty
