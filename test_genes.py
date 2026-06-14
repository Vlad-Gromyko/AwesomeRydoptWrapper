import matplotlib.pyplot as plt
import rydopt as ro
import numpy as np
import optax
from awesome_rydopt import (awesome_state, optimize, sequence_optimize, multi_start_optimize, imitate_rydopt,
                            population_optimize)
from awesome_rydopt import filter_by_fidelity_function, sort_by_fidelity_function
from optimization.population.cma_optimizer import CMAOptimizer
from optimization.population.de import DEOptimizer

from src.fidelities.intensity import custom_robust_shift
from rydopt.simulation.fidelity import process_fidelity

from src.pulses import genetic_sum_ansatz, genes_to_sympy_expr

if __name__ == '__main__':
    imitate_rydopt(False)

    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=float("inf"), decay=0.0)
    pulse_ansatz = ro.pulses.PulseAnsatz(
        detuning_ansatz=ro.pulses.const, phase_ansatz=genetic_sum_ansatz, rabi_ansatz=ro.pulses.const)

    initial_params = (15.560089695727132,
                      [-0.66202403],
                      [0 for i in range(4 * 10)],
                      [1.0])
    min_initial_params = (10.0,
                          [-1.0],
                          [-1.0 for i in range(len(initial_params[2]))],
                          [1.0])
    max_initial_params = (25.0,
                          [1.0],
                          [1.0 for i in range(len(initial_params[2]))],
                          [1.0])

    bounds = (min_initial_params, max_initial_params,)

    random_initial_params = (np.random.uniform(min_initial_params[0], max_initial_params[0]),
                             np.random.uniform(min_initial_params[1], max_initial_params[1]),
                             np.random.uniform(min_initial_params[2], max_initial_params[2]),
                             np.random.uniform(min_initial_params[3], max_initial_params[3]),)

    fixed_params = (False,
                    [False],
                    [False for i in range(len(initial_params[2]))],
                    [True])


    opt_result = optimize(gate, pulse_ansatz, random_initial_params, num_steps=1000, tol=1e-10,
                              min_initial_params=min_initial_params,  # Границы
                              max_initial_params=max_initial_params,
                              method=optax.nadam,  # Выбор оптимизатора
                              fidelity_type=custom_robust_shift,  # Выбор фиделити
                              apply_bounds=True,  # Применение границ
                              return_best=True)




    awesome_state.plot_history(save_path='optimization_flow.png')

    optimized_params = opt_result.params

    # print(genes_to_sympy_expr(optimized_params[2]))


    ro.characterization.plot_pulse(pulse_ansatz, optimized_params)

    plt.show()
