import matplotlib.pyplot as plt
import rydopt as ro
import numpy as np
import optax
from awesome_rydopt import (awesome_state, optimize, sequence_optimize, multi_start_optimize, imitate_rydopt,
                            population_optimize)
from awesome_rydopt import filter_by_fidelity_function, sort_by_fidelity_function
from optimization.population.cma_optimizer import CMAOptimizer

from src.fidelities.intensity import custom_robust_shift
from rydopt.simulation.fidelity import process_fidelity

if __name__ == '__main__':
    imitate_rydopt(False)

    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=float("inf"), decay=0.0)
    pulse_ansatz = ro.pulses.PulseAnsatz(
        detuning_ansatz=ro.pulses.const, phase_ansatz=ro.pulses.lin_sin_cos_crab, rabi_ansatz=ro.pulses.const)

    initial_params = (15.560089695727132,
                      [-0.66202403],
                      [0.51990003, -0.2527502, 0.8859972, -1.17347235, -0.08802893,
                       -0.327713, -1.00445164, 0.10583959, 1.20175343, 0.62878523,
                       0.65908966, -0.25679209, -0.56651597, 1.10736928, -0.32141791,
                       0.14352779, -0.2186016],
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

    if True:
        opt_result = optimize(gate, pulse_ansatz, random_initial_params, num_steps=10, tol=1e-10,
                              min_initial_params=min_initial_params,  # Границы
                              max_initial_params=max_initial_params,
                              method=optax.adam,  # Выбор оптимизатора
                              fidelity_type=custom_robust_shift,  # Выбор фиделити
                              apply_bounds=True,  # Применение границ
                              return_best=True)

    if False:
        opt_result = sequence_optimize(gate, pulse_ansatz, opt_result.params,
                                       min_initial_params=min_initial_params,
                                       max_initial_params=max_initial_params,
                                       fixed_initial_params=fixed_params,
                                       learning_rate=[0.01, 0.0005, 0.0001],  # -> [0.01, 0.0005, 0.0001]
                                       num_steps=[20, 10],  # -> [20, 1000, 20]
                                       tol=1e-10,  # -> [1e-10, 1e-10, 1e-10]
                                       method=[optax.adam],  # -> [optax.adam, optax.adam, optax.adam]
                                       fidelity_type=[custom_robust_shift, custom_robust_shift],
                                       # -> [process_fidelity, custom_robust_shift, process_fidelity]
                                       apply_bounds=False,  # Применение границ
                                       return_best=True)

    # После градиентной оптимизации используем её результат как начальное среднее для CMA-ES
    if True:
        # Затем CMA-ES с начальным средним из результата градиентной оптимизации
        opt_result = population_optimize(
            gate, pulse_ansatz, opt_result.params,
            min_initial_params, max_initial_params,
            fixed_params,
            num_generations=5,
            population_size=15,
            optimizer_class=CMAOptimizer,
            optimizer_kwargs={'sigma0': 0.2, 'seed': 42},  # Меньший sigma0 т.к. уже близко
            tol=1e-8,
            fidelity_type=custom_robust_shift,
            verbose=True,
            apply_bounds=True,
            return_history=True,
            return_best=True)

    optimized_params = opt_result.params
    ro.characterization.plot_pulse(pulse_ansatz, optimized_params)

    plt.show()
