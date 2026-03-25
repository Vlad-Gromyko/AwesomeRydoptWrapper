import matplotlib.pyplot as plt
import rydopt as ro
import numpy as np
import optax
from awesome_rydopt import imitate_rydopt, optimize, sequence_optimize, multi_start_optimize
from awesome_rydopt import filter_by_fidelity_function, sort_by_fidelity_function

from src.fidelities.intensity import custom_robust_shift
from rydopt.simulation.fidelity import process_fidelity

if __name__ == '__main__':
    imitate_rydopt(True)

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
        opt_result = optimize(gate, pulse_ansatz, initial_params, num_steps=1000, tol=1e-10,
                              min_initial_params=min_initial_params,  # Границы
                              max_initial_params=max_initial_params,
                              method=optax.lion,  # Выбор оптимизатор
                              fidelity_type=custom_robust_shift,  # Выбор фиделити
                              apply_bounds=True  # Применение границ
                              )

    if True:
        opt_result = sequence_optimize(gate, pulse_ansatz, random_initial_params,
                                       min_initial_params=min_initial_params,
                                       max_initial_params=max_initial_params,
                                       fixed_initial_params=fixed_params,
                                       learning_rate=[0.01, 0.0005, 0.0001],  # -> [0.01, 0.0005, 0.0001]
                                       num_steps=[20, 1000],  # -> [20, 1000, 20]
                                       tol=1e-10,  # -> [1e-10, 1e-10, 1e-10]
                                       method=[optax.adam],  # -> [optax.adam, optax.adam, optax.adam]
                                       fidelity_type=[process_fidelity, custom_robust_shift],
                                       # -> [process_fidelity, custom_robust_shift, process_fidelity]
                                       apply_bounds=False)

    if True:
        opt_result = multi_start_optimize(gate, pulse_ansatz,
                                          min_initial_params,
                                          max_initial_params,
                                          fixed_params,
                                          learning_rate=0.001,
                                          num_steps=100,
                                          tol=1e-10,
                                          method=optax.nadamw,
                                          fidelity_type=process_fidelity,
                                          num_processes=8,
                                          apply_bounds=False,
                                          return_list_results=True)  # Отдает не один OptimizationResult, а все списком

        # Сортировочка
        sorted_ = sort_by_fidelity_function(gate, pulse_ansatz, opt_result, custom_robust_shift, tol=1e-7)

        # Отбор лучшего
        opt_result = filter_by_fidelity_function(gate, pulse_ansatz, opt_result, custom_robust_shift, tol=1e-10)

    optimized_params = opt_result.params
    ro.characterization.plot_pulse(pulse_ansatz, optimized_params)

    plt.show()
