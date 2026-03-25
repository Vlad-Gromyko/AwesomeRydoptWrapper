import rydopt as ro
import numpy as np
import matplotlib.pyplot as plt
import optax

if __name__ == '__main__':
    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=float("inf"), decay=0.0)
    pulse_ansatz = ro.pulses.PulseAnsatz(
        detuning_ansatz=ro.pulses.const,
        phase_ansatz=ro.pulses.lin_sin_cos_crab,
        rabi_ansatz=ro.pulses.const
    )

    initial_params = (15.563866534286548, [-0.6475107],
                      [0.53430758, -0.30142903, 0.83574059, -1., -0.02144667, -0.36510733,
                       -1., 0.12886407, 1., 0.59720888, 0.68612173, -0.19252638,
                       -0.50734043, 1., -0.27862481, 0.12556319, -0.18442565], [1.])

    final_states = ro.simulation.evolve(gate, pulse_ansatz, initial_params,  tol=1e-10,)

    print(gate.process_fidelity(final_states))
