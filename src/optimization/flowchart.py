"""
Визуализация прогресса оптимизации - только infidelity.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Optional


def plot_optimization_flow(steps: List[Dict[str, Any]], save_path: Optional[str] = None):
    """
    График прогресса infidelity.
    """
    if not steps:
        print("No steps to visualize")
        return

    print(f"Total steps: {len(steps)}")

    # Собираем данные
    step_nums = []
    infidelities = []
    methods = []
    improvements = []

    prev_inf = None

    for i, step in enumerate(steps):
        step_nums.append(i + 1)
        metadata = step.get('metadata', {})

        print(f"Step {i+1}: metadata keys = {list(metadata.keys())}")

        # Infidelity
        inf = metadata.get('output_infidelity', None)
        if inf is None:
            print(f"  WARNING: No output_infidelity in step {i+1}")
            continue

        infidelities.append(inf)
        print(f"  infidelity = {inf:.6e}")

        # Название метода
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

        # Улучшение
        if prev_inf is not None:
            improvements.append(prev_inf - inf)
        else:
            improvements.append(0)

        prev_inf = inf

    if not infidelities:
        print("No infidelity data available - nothing to plot")
        return

    print(f"\nPlotting {len(infidelities)} points...")

    # Цвета для разных методов
    color_map = {
        'Gradient': '#1f77b4',
        'CMA-ES': '#2ca02c',
        'Multi-start': '#ff7f0e',
        'Sequence': '#d62728',
    }
    colors = [color_map.get(m, '#888888') for m in methods]

    # Создаем график
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Линия и точки
    ax.semilogy(step_nums[:len(infidelities)], infidelities, 'o-', linewidth=2, markersize=10,
                color='#333333', zorder=1, alpha=0.5)

    # Подписываем точки
    for i, (x, y, method, imp) in enumerate(zip(step_nums[:len(infidelities)], infidelities, methods, improvements)):
        # Цветная точка
        ax.plot(x, y, 'o', markersize=12, color=colors[i], zorder=2,
                markeredgecolor='black', markeredgewidth=1.5)

        # Подпись: метод
        ax.annotate(method, (x, y), xytext=(0, 12), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    color=colors[i])

        # Значение infidelity
        inf_str = f'{y:.1e}'
        ax.annotate(inf_str, (x, y), xytext=(8, 0), textcoords='offset points',
                    ha='left', va='center', fontsize=8, fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        # Улучшение (стрелка вниз)
        if imp > 0 and i > 0:
            ax.annotate(f'▼ {imp:.1e}', (x, y), xytext=(0, -20), textcoords='offset points',
                        ha='center', va='top', fontsize=7, color='#2e7d32')

    # Настройки графика
    ax.set_xlabel('Optimization Step', fontsize=12)
    ax.set_ylabel('Infidelity (log scale)', fontsize=12)
    ax.set_title('Optimization Progress - Infidelity', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.set_xlim(0.5, len(step_nums) + 0.5)

    # Таблица с шагами
    table_data = [[
        f"{i+1}",
        methods[i],
        f"{infidelities[i]:.1e}",
        f"{improvements[i]:.1e}" if improvements[i] > 0 else '-'
    ] for i in range(len(infidelities))]

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
    print("Graph should be displayed now")